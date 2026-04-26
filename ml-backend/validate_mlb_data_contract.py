"""
Validate ESPN MLB data contracts used by fetch_mlb_stats.py.

This script is a schema sentinel for upstream API drift. It checks that the
minimum required keys/structures still exist for:
  1) scoreboard events
  2) competition odds -> propBets reference
  3) team roster payload
  4) player gamelog payload

Exit codes:
  0 = pass (or no games for target date)
  1 = failed contract checks
"""

from __future__ import annotations

import argparse
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests

ESPN_BASE = "https://site.web.api.espn.com/apis"
ESPN_CORE_BASE = "https://sports.core.api.espn.com/v2/sports/baseball/leagues/mlb"
SCOREBOARD_URL = f"{ESPN_BASE}/site/v2/sports/baseball/mlb/scoreboard"
TEAM_ROSTER_URL = f"{ESPN_BASE}/site/v2/sports/baseball/mlb/teams/{{team_id}}/roster"
PLAYER_GAMELOG_URL = f"{ESPN_BASE}/common/v3/sports/baseball/mlb/athletes/{{player_id}}/gamelog"

# One team is enough for contract checks.
SAMPLE_TEAM_ID = 29  # ARI


@dataclass
class CheckResult:
    name: str
    ok: bool
    message: str


def fetch_with_retry(url: str, retries: int = 2, delay: float = 0.75) -> Optional[Dict[str, Any]]:
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=15)
            if response.status_code == 200:
                return response.json()
        except Exception:
            pass

        if attempt < retries - 1:
            time.sleep(delay)
    return None


def parse_competition_id_from_ref(ref_url: Optional[str]) -> Optional[str]:
    if not ref_url:
        return None
    m = re.search(r"/competitions/(\d+)", str(ref_url))
    return m.group(1) if m else None


def parse_athlete_id_from_ref(ref_url: Optional[str]) -> Optional[str]:
    if not ref_url:
        return None
    m = re.search(r"/athletes/(\d+)", str(ref_url))
    return m.group(1) if m else None


def check_scoreboard(date_yyyymmdd: Optional[str]) -> Tuple[List[CheckResult], List[str]]:
    url = SCOREBOARD_URL if not date_yyyymmdd else f"{SCOREBOARD_URL}?dates={date_yyyymmdd}"
    data = fetch_with_retry(url)
    results: List[CheckResult] = []

    if not data:
        return [CheckResult("scoreboard_fetch", False, "Failed to fetch scoreboard payload")], []

    events = data.get("events")
    if not isinstance(events, list):
        return [CheckResult("scoreboard_events", False, "Missing/invalid 'events' array on scoreboard payload")], []

    event_ids = [str(e.get("id")) for e in events if isinstance(e, dict) and e.get("id")]
    results.append(CheckResult("scoreboard_events", True, f"Found events array with {len(event_ids)} event id(s)"))

    if not events:
        results.append(CheckResult("scoreboard_no_games", True, "No games for target date; skipping game-dependent contract checks"))
        return results, []

    first = events[0] if isinstance(events[0], dict) else {}
    comps = first.get("competitions")
    if not isinstance(comps, list) or not comps:
        results.append(CheckResult("scoreboard_competitions", False, "Missing competitions list on first event"))
    else:
        results.append(CheckResult("scoreboard_competitions", True, "First event includes competitions list"))

    return results, event_ids


def check_event_odds_contract(event_id: str, require_props: bool) -> List[CheckResult]:
    results: List[CheckResult] = []

    odds_url = f"{ESPN_CORE_BASE}/events/{event_id}/competitions/{event_id}/odds"
    odds_data = fetch_with_retry(odds_url)
    if not odds_data:
        return [CheckResult("odds_fetch", False, f"Failed to fetch odds payload for event {event_id}")]

    items = odds_data.get("items")
    if not isinstance(items, list):
        return [CheckResult("odds_items", False, "Missing/invalid 'items' in odds payload")]

    results.append(CheckResult("odds_items", True, f"Odds payload has items ({len(items)})"))

    if not items:
        if require_props:
            results.append(CheckResult("odds_items_nonempty", False, "Odds items empty while --require-props is enabled"))
        else:
            results.append(CheckResult("odds_items_nonempty", True, "Odds items empty; tolerated without --require-props"))
        return results

    first = items[0] if isinstance(items[0], dict) else {}
    prop_ref = (first.get("propBets") or {}).get("$ref")
    if not prop_ref:
        if require_props:
            results.append(CheckResult("odds_propbets_ref", False, "Missing propBets.$ref on first odds item"))
        else:
            results.append(CheckResult("odds_propbets_ref", True, "Missing propBets.$ref tolerated without --require-props"))
        return results

    results.append(CheckResult("odds_propbets_ref", True, "Found propBets.$ref on first odds item"))

    prop_url = str(prop_ref).replace("http://", "https://")
    prop_data = fetch_with_retry(prop_url)
    if not prop_data:
        return results + [CheckResult("propbets_fetch", False, "Failed to fetch propBets payload from $ref")]

    prop_items = prop_data.get("items")
    if not isinstance(prop_items, list):
        return results + [CheckResult("propbets_items", False, "Missing/invalid 'items' in propBets payload")]

    results.append(CheckResult("propbets_items", True, f"propBets payload has items ({len(prop_items)})"))
    return results


def _find_first_roster_athlete_id(roster_data: Dict[str, Any]) -> Optional[str]:
    groups = roster_data.get("athletes")
    if not isinstance(groups, list):
        return None

    for group in groups:
        for item in (group or {}).get("items", []):
            if not isinstance(item, dict):
                continue

            # ESPN roster responses may either wrap athlete under item['athlete']
            # or provide athlete fields directly on the item object.
            direct_id = item.get("id")
            if direct_id:
                return str(direct_id)

            athlete = item.get("athlete") or {}
            athlete_id = athlete.get("id")
            if athlete_id:
                return str(athlete_id)
    return None


def check_roster_and_gamelog_contract() -> List[CheckResult]:
    results: List[CheckResult] = []

    roster_data = fetch_with_retry(TEAM_ROSTER_URL.format(team_id=SAMPLE_TEAM_ID))
    if not roster_data:
        return [CheckResult("roster_fetch", False, "Failed to fetch sample team roster payload")]

    athletes = roster_data.get("athletes")
    if not isinstance(athletes, list):
        return [CheckResult("roster_athletes", False, "Missing/invalid 'athletes' in roster payload")]

    results.append(CheckResult("roster_athletes", True, "Roster payload includes athletes list"))

    athlete_id = _find_first_roster_athlete_id(roster_data)
    if not athlete_id:
        return results + [CheckResult("roster_sample_athlete", False, "Could not resolve sample athlete id from roster")]

    results.append(CheckResult("roster_sample_athlete", True, f"Resolved sample athlete id {athlete_id}"))

    gamelog_data = fetch_with_retry(PLAYER_GAMELOG_URL.format(player_id=athlete_id))
    if not gamelog_data:
        return results + [CheckResult("gamelog_fetch", False, f"Failed to fetch gamelog for athlete {athlete_id}")]

    if not isinstance(gamelog_data.get("seasonTypes"), list):
        return results + [CheckResult("gamelog_seasontypes", False, "Missing/invalid seasonTypes in gamelog payload")]

    results.append(CheckResult("gamelog_seasontypes", True, "Gamelog payload includes seasonTypes"))

    # Ensure at least one event carries stats-compatible keys used by parser.
    found_event = False
    found_stats = False
    for st in gamelog_data.get("seasonTypes", []):
        for cat in (st or {}).get("categories", []):
            for event in (cat or {}).get("events", []):
                if not isinstance(event, dict):
                    continue
                found_event = True

                stats = event.get("stats")
                if isinstance(stats, list):
                    found_stats = True
                    break

                # Tolerate alternate structure if upstream shifts key naming.
                alt_stats = event.get("statistics")
                if isinstance(alt_stats, list):
                    found_stats = True
                    break
            if found_stats:
                break
        if found_stats:
            break

    if not found_event:
        results.append(CheckResult("gamelog_events", False, "No events found in gamelog seasonTypes/categories"))
    else:
        results.append(CheckResult("gamelog_events", True, "Found at least one gamelog event"))

    if not found_stats:
        results.append(CheckResult("gamelog_statistics", False, "No stats/statistics list found on gamelog events"))
    else:
        results.append(CheckResult("gamelog_statistics", True, "Found stats-compatible list on gamelog events"))

    return results


def summarize_and_exit(results: List[CheckResult]) -> int:
    failed = [r for r in results if not r.ok]
    passed = [r for r in results if r.ok]

    print("MLB data contract check:")
    for r in results:
        state = "PASS" if r.ok else "FAIL"
        print(f"  [{state}] {r.name}: {r.message}")

    print(f"\nSummary: {len(passed)} passed, {len(failed)} failed")
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate ESPN MLB data contracts")
    parser.add_argument("--date", help="Optional scoreboard date YYYYMMDD", default=None)
    parser.add_argument(
        "--require-props",
        action="store_true",
        help="Require odds items and propBets ref to be present when games exist",
    )
    args = parser.parse_args()

    checks: List[CheckResult] = []
    scoreboard_checks, event_ids = check_scoreboard(args.date)
    checks.extend(scoreboard_checks)

    # If no games, we intentionally pass. Contract drift on game-day endpoints
    # can still be detected in other scheduled runs.
    if event_ids:
        checks.extend(check_event_odds_contract(event_ids[0], require_props=args.require_props))

    checks.extend(check_roster_and_gamelog_contract())

    return summarize_and_exit(checks)


if __name__ == "__main__":
    raise SystemExit(main())
