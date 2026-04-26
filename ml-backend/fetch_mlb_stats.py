"""
Fetch MLB player statistics from ESPN API - 2024 Season (2025 season hasn't started yet)
Fetches hitters and pitchers from all 30 MLB teams
Focus: Hits, Home Runs, RBIs for hitters; Strikeouts, Hits Allowed, Earned Runs for pitchers
"""

import requests
import json
import time
import re
import argparse
from datetime import datetime, timezone

# ESPN API endpoints
ESPN_BASE = "https://site.web.api.espn.com/apis"
ESPN_CORE_BASE = "https://sports.core.api.espn.com/v2/sports/baseball/leagues/mlb"
PLAYER_GAMELOG_URL = f"{ESPN_BASE}/common/v3/sports/baseball/mlb/athletes/{{player_id}}/gamelog"
TEAM_ROSTER_URL = f"{ESPN_BASE}/site/v2/sports/baseball/mlb/teams/{{team_id}}/roster"
SCOREBOARD_URL = f"{ESPN_BASE}/site/v2/sports/baseball/mlb/scoreboard"
ODDS_HISTORY_FILE = "mlb_prop_odds_history.json"

# MLB Team IDs for ESPN API
MLB_TEAMS = {
    'ARI': 29, 'ATL': 15, 'BAL': 1, 'BOS': 2, 'CHC': 16, 'CWS': 4,
    'CIN': 17, 'CLE': 5, 'COL': 27, 'DET': 6, 'HOU': 18, 'KC': 7,
    'LAA': 3, 'LAD': 19, 'MIA': 28, 'MIL': 8, 'MIN': 9, 'NYM': 21,
    'NYY': 10, 'OAK': 11, 'PHI': 22, 'PIT': 23, 'SD': 25, 'SF': 26,
    'SEA': 12, 'STL': 24, 'TB': 30, 'TEX': 13, 'TOR': 14, 'WSH': 20
}


def normalize_stat_name(name):
    """Normalize ESPN stat names to lowercase alphanumerics (e.g. homeRuns -> homeruns)."""
    return re.sub(r'[^a-z0-9]', '', (name or '').lower())


def parse_pitching_innings(value):
    """
    Parse MLB innings notation.

    ESPN may return:
    - decimal string (e.g. '5.2' meaning 5 and 2/3 innings)
    - int/float values
    This converts .1 -> 1/3 and .2 -> 2/3 for consistent numeric modeling.
    """
    s = str(value or '0').strip()
    if not s:
        return 0.0

    if '.' not in s:
        try:
            return float(s)
        except Exception:
            return 0.0

    whole, frac = s.split('.', 1)
    try:
        whole_int = int(whole)
    except Exception:
        whole_int = 0

    frac = frac[:1]
    if frac == '1':
        return whole_int + (1.0 / 3.0)
    if frac == '2':
        return whole_int + (2.0 / 3.0)

    try:
        return float(s)
    except Exception:
        return float(whole_int)

def fetch_with_retry(url, retries=2, delay=1):
    """Fetch URL with retry logic"""
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=15)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
        
        if attempt < retries - 1:
            time.sleep(delay)
    
    return None


def parse_athlete_id_from_ref(ref_url):
    """Extract athlete id from ESPN $ref URL."""
    if not ref_url:
        return None
    m = re.search(r"/athletes/(\d+)", str(ref_url))
    return m.group(1) if m else None


def parse_competition_id_from_ref(ref_url):
    """Extract competition/event id from ESPN competition $ref URL."""
    if not ref_url:
        return None
    m = re.search(r"/competitions/(\d+)", str(ref_url))
    return m.group(1) if m else None


def parse_american_odds_value(v):
    """Parse American odds string like '+105' / '-120' into int."""
    if v is None:
        return None
    try:
        return int(str(v).strip())
    except Exception:
        return None


def map_mlb_prop_name(prop_name):
    """Map ESPN prop names to internal target names where possible."""
    p = (prop_name or "").strip().lower()
    if p in ["total hits", "hits"]:
        return "hits"
    if p in ["total home runs", "home runs", "homeruns"]:
        return "home_runs"
    if p in ["total rbis", "total rbi", "rbis", "rbi"]:
        return "rbis"
    if p in ["total strikeouts", "strikeouts"]:
        return "strikeouts"
    if p in ["hits allowed", "total hits allowed"]:
        return "hits_allowed"
    if p in ["earned runs", "total earned runs"]:
        return "earned_runs"
    return None


def load_odds_history_records():
    """Load odds history file into a record dictionary."""
    try:
        with open(ODDS_HISTORY_FILE, 'r', encoding='utf-8') as f:
            payload = json.load(f)
            return payload.get('records', {}) if isinstance(payload, dict) else {}
    except Exception:
        return {}


def save_odds_history_records(records):
    payload = {
        'updated_at_utc': datetime.now(timezone.utc).isoformat(),
        'records': records,
    }
    with open(ODDS_HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)


def update_odds_history(parsed_markets):
    """
    Upsert live parsed markets into historical cache.
    Key format: game_id|player_id|target
    """
    records = load_odds_history_records()
    now = datetime.now(timezone.utc).isoformat()
    upserts = 0

    for m in parsed_markets:
        key = f"{m['game_id']}|{m['player_id']}|{m['target']}"
        prev = records.get(key)
        if prev:
            first_seen = prev.get('first_seen_utc', now)
        else:
            first_seen = now

        records[key] = {
            'game_id': m['game_id'],
            'player_id': m['player_id'],
            'target': m['target'],
            'line': m.get('line'),
            'over': m.get('over'),
            'under': m.get('under'),
            'source': m.get('source', 'ESPN_DRAFTKINGS_LIVE'),
            'side_assignment': m.get('side_assignment', 'feed_order_guess'),
            'first_seen_utc': first_seen,
            'last_seen_utc': now,
        }
        upserts += 1

    save_odds_history_records(records)
    return records, upserts


def build_odds_index(records):
    """Build lookup index: game_id -> player_id -> target -> market."""
    index = {}
    for rec in records.values():
        game_id = str(rec.get('game_id') or '')
        player_id = str(rec.get('player_id') or '')
        target = rec.get('target')
        if not game_id or not player_id or not target:
            continue
        index.setdefault(game_id, {}).setdefault(player_id, {})[target] = {
            'line': rec.get('line'),
            'over': rec.get('over'),
            'under': rec.get('under'),
            'source': rec.get('source'),
            'side_assignment': rec.get('side_assignment'),
        }
    return index


def american_to_implied_prob(odds):
    if odds is None:
        return None
    o = int(odds)
    if o > 0:
        return round(100.0 / (o + 100.0), 6)
    return round(abs(o) / (abs(o) + 100.0), 6)


def enrich_gamelog_with_market_odds(gamelog, player_id):
    """
    Attach matched historical market odds to each game row when available.
    Stored at game['market_odds'][target] for training-time odds features.
    """
    records = load_odds_history_records()
    idx = build_odds_index(records)
    pid = str(player_id)

    for game in gamelog:
        gid = str(game.get('game_id') or '')
        if not gid:
            continue
        market = (idx.get(gid, {}).get(pid, {}) or {})
        if not market:
            continue

        game_market = {}
        for target, m in market.items():
            game_market[target] = {
                'line': m.get('line'),
                'over': m.get('over'),
                'under': m.get('under'),
                'implied_over': american_to_implied_prob(m.get('over')),
                'implied_under': american_to_implied_prob(m.get('under')),
                'source': m.get('source'),
            }
        game['market_odds'] = game_market


def fetch_todays_mlb_event_ids(date_yyyymmdd=None):
    """Get MLB event ids from scoreboard for a given date (or ESPN default day)."""
    url = SCOREBOARD_URL
    if date_yyyymmdd:
        url = f"{SCOREBOARD_URL}?dates={date_yyyymmdd}"

    data = fetch_with_retry(url, retries=2, delay=1)
    if not data:
        return []
    return [e.get('id') for e in data.get('events', []) if e.get('id')]


def fetch_event_prop_items(event_id):
    """
    Fetch raw prop bet items for an event via competition odds -> propBets $ref.
    """
    odds_url = f"{ESPN_CORE_BASE}/events/{event_id}/competitions/{event_id}/odds"
    odds_data = fetch_with_retry(odds_url, retries=2, delay=0.5)
    if not odds_data:
        return []

    items = odds_data.get('items', [])
    if not items:
        return []

    first = items[0] if isinstance(items[0], dict) else {}
    prop_ref = (first.get('propBets') or {}).get('$ref')
    if not prop_ref:
        return []

    # ESPN often returns http refs; upgrade to https for requests.
    prop_ref = prop_ref.replace('http://', 'https://')
    prop_data = fetch_with_retry(prop_ref, retries=2, delay=0.5)
    if not prop_data:
        return []

    return prop_data.get('items', [])


def parse_live_prop_odds(prop_items):
    """
    Parse raw prop rows into per-athlete markets.

    MLB prop feed usually emits two rows per market (one per side) without explicit
    over/under labels. We preserve both prices and assign side order heuristically
    by feed order for EV support.
    """
    grouped = {}

    for item in prop_items:
        try:
            athlete_ref = (item.get('athlete') or {}).get('$ref', '')
            athlete_id = parse_athlete_id_from_ref(athlete_ref)
            if not athlete_id:
                continue

            prop_name = (item.get('type') or {}).get('name', '')
            mapped = map_mlb_prop_name(prop_name)
            if not mapped:
                continue

            current = item.get('current') or {}
            target = current.get('target') or {}
            line_val = target.get('value')
            if line_val is None:
                continue

            odds_obj = item.get('odds') or {}
            american_obj = odds_obj.get('american') or {}
            price = parse_american_odds_value(american_obj.get('value'))
            if price is None:
                continue

            key = (athlete_id, mapped, float(line_val))
            grouped.setdefault(key, []).append(price)
        except Exception:
            continue

    per_player = {}
    parsed_markets = []
    for (athlete_id, mapped, line_val), prices in grouped.items():
        # Deduplicate while preserving order.
        dedup = []
        for p in prices:
            if p not in dedup:
                dedup.append(p)

        over_odds = dedup[0] if len(dedup) >= 1 else None
        under_odds = dedup[1] if len(dedup) >= 2 else None

        per_player.setdefault(athlete_id, {})[mapped] = {
            'line': float(line_val),
            'over': over_odds,
            'under': under_odds,
            'source': 'ESPN_DRAFTKINGS_LIVE',
            'side_assignment': 'feed_order_guess'
        }

    # Re-parse with game_id attached for historical cache upserts.
    detailed_grouped = {}
    for item in prop_items:
        try:
            athlete_ref = (item.get('athlete') or {}).get('$ref', '')
            athlete_id = parse_athlete_id_from_ref(athlete_ref)
            if not athlete_id:
                continue

            comp_ref = (item.get('competition') or {}).get('$ref', '')
            game_id = parse_competition_id_from_ref(comp_ref)
            if not game_id:
                continue

            prop_name = (item.get('type') or {}).get('name', '')
            mapped = map_mlb_prop_name(prop_name)
            if not mapped:
                continue

            target_obj = (item.get('current') or {}).get('target') or {}
            line_val = target_obj.get('value')
            if line_val is None:
                continue

            american_val = ((item.get('odds') or {}).get('american') or {}).get('value')
            price = parse_american_odds_value(american_val)
            if price is None:
                continue

            key = (str(game_id), str(athlete_id), mapped, float(line_val))
            detailed_grouped.setdefault(key, []).append(price)
        except Exception:
            continue

    for (game_id, athlete_id, mapped, line_val), prices in detailed_grouped.items():
        dedup = []
        for p in prices:
            if p not in dedup:
                dedup.append(p)
        parsed_markets.append({
            'game_id': game_id,
            'player_id': athlete_id,
            'target': mapped,
            'line': float(line_val),
            'over': dedup[0] if len(dedup) >= 1 else None,
            'under': dedup[1] if len(dedup) >= 2 else None,
            'source': 'ESPN_DRAFTKINGS_LIVE',
            'side_assignment': 'feed_order_guess',
        })

    return per_player, parsed_markets


def fetch_live_mlb_prop_odds(date_yyyymmdd=None):
    """Fetch and aggregate live MLB player prop odds for a given date (or ESPN default day)."""
    event_ids = fetch_todays_mlb_event_ids(date_yyyymmdd=date_yyyymmdd)
    if not event_ids:
        print("No MLB events found for live odds")
        return {}

    print(f"Fetching live prop odds from {len(event_ids)} MLB games...")

    combined_items = []
    for event_id in event_ids:
        items = fetch_event_prop_items(event_id)
        if items:
            combined_items.extend(items)
        time.sleep(0.1)

    live_odds, parsed_markets = parse_live_prop_odds(combined_items)
    _, upserts = update_odds_history(parsed_markets)
    print(f"Live odds parsed for {len(live_odds)} players")
    print(f"Odds history upserts: {upserts}")
    return live_odds


def run_odds_only_update(date_yyyymmdd=None):
    """Fast mode: only refresh live odds and historical odds cache."""
    print("=" * 60)
    print("MLB LIVE ODDS UPDATER")
    print("=" * 60)
    if date_yyyymmdd:
        print(f"Date override: {date_yyyymmdd}")

    live = fetch_live_mlb_prop_odds(date_yyyymmdd=date_yyyymmdd)
    records = load_odds_history_records()

    print(f"\nLive players with odds: {len(live)}")
    print(f"Historical odds records: {len(records)}")
    print("=" * 60)

def get_team_roster(team_code):
    """Fetch all players from team roster"""
    team_id = MLB_TEAMS.get(team_code)
    if not team_id:
        return []
    
    url = TEAM_ROSTER_URL.format(team_id=team_id)
    data = fetch_with_retry(url)
    
    if not data or 'athletes' not in data:
        return []
    
    players = []
    for group in data.get('athletes', []):
        for athlete in group.get('items', []):
            try:
                position_data = athlete.get('position', {})
                position_abbr = position_data.get('abbreviation', 'P')
                
                # Include all positions
                players.append({
                    'id': athlete.get('id'),
                    'name': athlete.get('displayName', 'Unknown'),
                    'team': team_code,
                    'position': position_abbr
                })
            except Exception as e:
                print(f"Error parsing player: {e}")
    
    return players

def get_player_gamelog(player_id, position):
    """
    Fetch player game log for 2024 season
    
    For Hitters: Hits, Home Runs, RBIs, Strikeouts, Walks
    For Pitchers: Strikeouts, Hits Allowed, Earned Runs, Innings Pitched, Walks
    """
    url = PLAYER_GAMELOG_URL.format(player_id=player_id)
    data = fetch_with_retry(url, retries=2, delay=0.5)
    
    if not data:
        return []
    
    try:
        stat_names = data.get('names', [])
        if not stat_names:
            return []
        
        is_pitcher = position in ['P', 'SP', 'RP']
        
        # Find indices for key stats
        hits_idx = None
        hr_idx = None
        rbi_idx = None
        k_idx = None
        bb_idx = None
        er_idx = None
        ip_idx = None
        
        for i, name in enumerate(stat_names):
            name_norm = normalize_stat_name(name)
            if not is_pitcher:
                # Hitter stats
                if name_norm in ['h', 'hits']:
                    hits_idx = i
                elif name_norm in ['hr', 'homerun', 'homeruns']:
                    hr_idx = i
                elif name_norm in ['rbi', 'rbis']:
                    rbi_idx = i
                elif name_norm in ['k', 'so', 'strikeouts']:
                    k_idx = i
                elif name_norm in ['bb', 'walk', 'walks']:
                    bb_idx = i
            else:
                # Pitcher stats
                if name_norm in ['k', 'so', 'strikeouts']:
                    k_idx = i
                elif name_norm in ['h', 'hits']:
                    hits_idx = i
                elif name_norm in ['er', 'earnedrun', 'earnedruns']:
                    er_idx = i
                elif name_norm in ['ip', 'inning', 'innings', 'inningspitched']:
                    ip_idx = i
                elif name_norm in ['bb', 'walk', 'walks']:
                    bb_idx = i
        
        gamelog = []
        
        # Events are stored in seasonTypes -> categories -> events (same as NBA/NHL)
        season_types = data.get('seasonTypes', [])
        for season in season_types:
            if 'Regular Season' not in season.get('displayName', ''):
                continue
                
            categories = season.get('categories', [])
            for category in categories:
                events_list = category.get('events', [])
                
                for event in events_list:
                    try:
                        event_id = event.get('eventId')
                        stat_values = event.get('stats', [])
                        
                        if not stat_values:
                            continue
                        
                        if is_pitcher:
                            game_stats = {
                                'game_id': event_id,
                                'strikeouts': float(stat_values[k_idx]) if k_idx is not None and k_idx < len(stat_values) else 0.0,
                                'hits_allowed': float(stat_values[hits_idx]) if hits_idx is not None and hits_idx < len(stat_values) else 0.0,
                                'earned_runs': float(stat_values[er_idx]) if er_idx is not None and er_idx < len(stat_values) else 0.0,
                                'innings_pitched': parse_pitching_innings(stat_values[ip_idx]) if ip_idx is not None and ip_idx < len(stat_values) else 0.0
                            }
                        else:
                            game_stats = {
                                'game_id': event_id,
                                'hits': float(stat_values[hits_idx]) if hits_idx is not None and hits_idx < len(stat_values) else 0.0,
                                'home_runs': float(stat_values[hr_idx]) if hr_idx is not None and hr_idx < len(stat_values) else 0.0,
                                'rbis': float(stat_values[rbi_idx]) if rbi_idx is not None and rbi_idx < len(stat_values) else 0.0,
                                'strikeouts': float(stat_values[k_idx]) if k_idx is not None and k_idx < len(stat_values) else 0.0
                            }
                        
                        gamelog.append(game_stats)
                        
                    except Exception as e:
                        print(f"Error parsing game: {e}")
                        continue
        
        return gamelog
        
    except Exception as e:
        print(f"Error processing gamelog: {e}")
        return []

def calculate_hit_probability(gamelog, threshold=1.5):
    """Calculate probability of getting 2+ hits"""
    if not gamelog:
        return {
            'games_played': 0,
            'games_with_2plus_hits': 0,
            'probability_2plus_hits': 0.0,
            'total_hits': 0.0,
            'avg_hits_per_game': 0.0
        }
    
    games_played = len(gamelog)
    hits = [g.get('hits', 0.0) for g in gamelog]
    games_with_2plus = sum(1 for h in hits if h >= 2)
    total_hits = sum(hits)
    
    return {
        'games_played': games_played,
        'games_with_2plus_hits': games_with_2plus,
        'probability_2plus_hits': round((games_with_2plus / games_played) * 100, 1) if games_played > 0 else 0.0,
        'total_hits': total_hits,
        'avg_hits_per_game': round(total_hits / games_played, 2) if games_played > 0 else 0.0
    }

def fetch_all_mlb_players(date_yyyymmdd=None):
    """Fetch all MLB players from all teams"""
    all_players = []

    live_prop_odds = fetch_live_mlb_prop_odds(date_yyyymmdd=date_yyyymmdd)
    
    print("Fetching MLB rosters from all 30 teams...")
    print("NOTE: Using 2024 season data (2025 season hasn't started yet)")
    
    for team_code, team_id in MLB_TEAMS.items():
        print(f"Fetching {team_code}...")
        roster = get_team_roster(team_code)
        
        for player in roster:
            player_id = player['id']
            position = player['position']
            print(f"  {player['name']} ({position})...", end=' ')
            
            gamelog = get_player_gamelog(player_id, position)
            
            if gamelog:
                enrich_gamelog_with_market_odds(gamelog, player_id)
                games_played = len(gamelog)
                is_pitcher = position in ['P', 'SP', 'RP']
                
                if is_pitcher:
                    # Pitcher stats
                    strikeouts = [g.get('strikeouts', 0) for g in gamelog]
                    hits_allowed = [g.get('hits_allowed', 0) for g in gamelog]
                    earned_runs = [g.get('earned_runs', 0) for g in gamelog]
                    
                    player_data = {
                        'id': player_id,
                        'player_name': player['name'],
                        'team': team_code,
                        'position': position,
                        'gamelog': gamelog,
                        'prop_odds': {},
                        'season_stats': {
                            'games_played': games_played,
                            'total_strikeouts': sum(strikeouts),
                            'avg_strikeouts': round(sum(strikeouts) / games_played, 1) if games_played > 0 else 0.0,
                            'avg_hits_allowed': round(sum(hits_allowed) / games_played, 1) if games_played > 0 else 0.0,
                            'avg_earned_runs': round(sum(earned_runs) / games_played, 2) if games_played > 0 else 0.0
                        }
                    }
                else:
                    # Hitter stats
                    hits = [g.get('hits', 0) for g in gamelog]
                    hrs = [g.get('home_runs', 0) for g in gamelog]
                    rbis = [g.get('rbis', 0) for g in gamelog]
                    
                    hit_prob = calculate_hit_probability(gamelog)
                    
                    player_data = {
                        'id': player_id,
                        'player_name': player['name'],
                        'team': team_code,
                        'position': position,
                        'gamelog': gamelog,
                        'prop_odds': {},
                        'hit_stats': hit_prob,
                        'season_stats': {
                            'games_played': games_played,
                            'total_hits': sum(hits),
                            'total_home_runs': sum(hrs),
                            'total_rbis': sum(rbis),
                            'avg_hits': round(sum(hits) / games_played, 2) if games_played > 0 else 0.0,
                            'probability_2plus_hits': hit_prob['probability_2plus_hits']
                        }
                    }

                # Attach live prop odds if available.
                player_market = live_prop_odds.get(str(player_id), {})
                if player_market:
                    if is_pitcher:
                        if 'strikeouts' in player_market:
                            player_data['prop_odds']['strikeouts'] = player_market['strikeouts']
                    else:
                        if 'hits' in player_market:
                            player_data['prop_odds']['hits'] = player_market['hits']
                        if 'home_runs' in player_market:
                            player_data['prop_odds']['home_runs'] = player_market['home_runs']
                        if 'rbis' in player_market:
                            player_data['prop_odds']['rbis'] = player_market['rbis']
                        # For hitters, strikeouts market maps to batter_strikeouts target.
                        if 'strikeouts' in player_market:
                            player_data['prop_odds']['batter_strikeouts'] = player_market['strikeouts']
                
                all_players.append(player_data)
                print(f"OK {games_played} games")
            else:
                print("No data")
            
            time.sleep(0.3)  # Rate limiting
    
    return all_players

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MLB player stats and live odds fetcher")
    parser.add_argument(
        "--odds-only",
        action="store_true",
        help="Only update live odds history cache (fast mode, no full roster scrape)",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Optional scoreboard date override in YYYYMMDD format (used by odds fetch)",
    )
    args = parser.parse_args()

    if args.odds_only:
        run_odds_only_update(date_yyyymmdd=args.date)
    else:
        print("=" * 60)
        print("MLB PLAYER STATS FETCHER - 2024 SEASON")
        print("=" * 60)

        players = fetch_all_mlb_players(date_yyyymmdd=args.date)

        # Save to JSON
        output_file = 'mlb_player_data.json'
        with open(output_file, 'w') as f:
            json.dump(players, f, indent=2)

        print(f"\n{'=' * 60}")
        print(f"COMPLETE: {len(players)} players saved to {output_file}")
        print(f"{'=' * 60}")

        # Print summary
        pitchers = [p for p in players if p['position'] in ['P', 'SP', 'RP']]
        hitters = [p for p in players if p['position'] not in ['P', 'SP', 'RP']]

        print(f"\nSummary:")
        print(f"  Total Players: {len(players)}")
        print(f"  Pitchers: {len(pitchers)}")
        print(f"  Hitters: {len(hitters)}")
