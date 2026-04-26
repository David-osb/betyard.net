"""
Generate a weekly MLB model health report from existing artifacts.

Inputs:
  - mlb_ensemble_metrics.json
  - mlb_prop_predictions_ml.json

Outputs:
  - reports/mlb_model_health_report.md
  - reports/mlb_model_health_report.json

Exit code:
  - 0: Healthy or warnings only
  - 1: One or more alert thresholds breached
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


ROOT = Path(__file__).parent
METRICS_FILE = ROOT / "mlb_ensemble_metrics.json"
PREDICTIONS_FILE = ROOT / "mlb_prop_predictions_ml.json"
REPORTS_DIR = ROOT / "reports"
REPORT_MD = REPORTS_DIR / "mlb_model_health_report.md"
REPORT_JSON = REPORTS_DIR / "mlb_model_health_report.json"

# Alert thresholds (fail job when breached)
MIN_RECOMMENDATION_HIT_RATE = 0.60
MAX_BRIER = 0.24
MIN_HOME_RUN_AUC = 0.70
MIN_TOTAL_BET_RECOMMENDATIONS = 20

# Warning thresholds (report only)
WARN_MIN_ODDS_COVERAGE = 0.10
WARN_MIN_EV_SAMPLES_PER_TARGET = 10


@dataclass
class Alert:
    severity: str  # "ALERT" or "WARN"
    message: str


def _safe_float(v: object) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _iter_targets(metrics: dict) -> Iterable[Tuple[str, dict]]:
    for group in ("hitter_targets", "pitcher_targets"):
        group_data = metrics.get(group, {})
        if not isinstance(group_data, dict):
            continue
        for target, details in group_data.items():
            if isinstance(details, dict):
                yield target, details


def _summarize_training_metrics(metrics: dict) -> Tuple[List[dict], dict]:
    rows: List[dict] = []
    weighted_hit_rate_numer = 0.0
    weighted_hit_rate_denom = 0.0
    weighted_brier_numer = 0.0
    weighted_brier_denom = 0.0

    for target, details in _iter_targets(metrics):
        test_samples = int(details.get("test_samples") or 0)
        rec_hit_rate = _safe_float(details.get("recommendation_hit_rate"))
        brier = _safe_float(details.get("brier"))
        auc = _safe_float(details.get("auc"))
        rec_rate = _safe_float(details.get("recommendation_rate"))

        rows.append(
            {
                "target": target,
                "status": details.get("status", "unknown"),
                "test_samples": test_samples,
                "recommendation_hit_rate": rec_hit_rate,
                "brier": brier,
                "auc": auc,
                "recommendation_rate": rec_rate,
            }
        )

        if rec_hit_rate is not None and test_samples > 0:
            weighted_hit_rate_numer += rec_hit_rate * test_samples
            weighted_hit_rate_denom += test_samples

        if brier is not None and test_samples > 0:
            weighted_brier_numer += brier * test_samples
            weighted_brier_denom += test_samples

    aggregates = {
        "weighted_recommendation_hit_rate": (
            weighted_hit_rate_numer / weighted_hit_rate_denom if weighted_hit_rate_denom else None
        ),
        "weighted_brier": (
            weighted_brier_numer / weighted_brier_denom if weighted_brier_denom else None
        ),
        "targets_evaluated": len(rows),
    }
    return rows, aggregates


def _summarize_prediction_markets(predictions: dict) -> dict:
    per_target: Dict[str, dict] = defaultdict(
        lambda: {
            "samples": 0,
            "odds_available": 0,
            "ev_samples": 0,
            "ev_positive": 0,
            "bet_recommendations": 0,
            "edge_sum": 0.0,
            "edge_n": 0,
            "ev_sum": 0.0,
            "ev_n": 0,
        }
    )

    total_props = 0
    total_props_with_odds = 0

    for side in ("hitters", "pitchers"):
        for player in predictions.get(side, []) or []:
            props = player.get("props", {})
            if not isinstance(props, dict):
                continue

            for target, prop in props.items():
                if not isinstance(prop, dict):
                    continue

                total_props += 1
                item = per_target[target]
                item["samples"] += 1

                over_odds = prop.get("odds_over")
                under_odds = prop.get("odds_under")
                odds_available = over_odds is not None and under_odds is not None
                if odds_available:
                    total_props_with_odds += 1
                    item["odds_available"] += 1

                rec = str(prop.get("recommendation") or "").upper()
                if rec in {"OVER", "UNDER"}:
                    item["bet_recommendations"] += 1

                ev = _safe_float(prop.get("ev_per_unit"))
                if ev is not None:
                    item["ev_samples"] += 1
                    item["ev_sum"] += ev
                    item["ev_n"] += 1
                    if ev > 0:
                        item["ev_positive"] += 1

                edge = _safe_float(prop.get("edge"))
                if edge is not None:
                    item["edge_sum"] += edge
                    item["edge_n"] += 1

    target_rows: List[dict] = []
    for target, d in sorted(per_target.items()):
        samples = d["samples"]
        target_rows.append(
            {
                "target": target,
                "samples": samples,
                "odds_coverage": (d["odds_available"] / samples) if samples else None,
                "ev_samples": d["ev_samples"],
                "ev_positive_rate": (d["ev_positive"] / d["ev_samples"]) if d["ev_samples"] else None,
                "avg_edge": (d["edge_sum"] / d["edge_n"]) if d["edge_n"] else None,
                "avg_ev_per_unit": (d["ev_sum"] / d["ev_n"]) if d["ev_n"] else None,
                "bet_recommendations": d["bet_recommendations"],
            }
        )

    return {
        "totals": {
            "total_props": total_props,
            "props_with_odds": total_props_with_odds,
            "overall_odds_coverage": (total_props_with_odds / total_props) if total_props else None,
            "total_bet_recommendations": sum(r["bet_recommendations"] for r in target_rows),
        },
        "targets": target_rows,
    }


def _evaluate_alerts(training_rows: List[dict], aggregates: dict, market_summary: dict) -> List[Alert]:
    alerts: List[Alert] = []

    for row in training_rows:
        target = row["target"]
        status = str(row.get("status") or "unknown").lower()
        if status != "ok":
            alerts.append(Alert("ALERT", f"{target}: status is '{status}', expected 'ok'."))

    weighted_hit = _safe_float(aggregates.get("weighted_recommendation_hit_rate"))
    if weighted_hit is not None and weighted_hit < MIN_RECOMMENDATION_HIT_RATE:
        alerts.append(
            Alert(
                "ALERT",
                f"Weighted recommendation hit rate {weighted_hit:.3f} is below {MIN_RECOMMENDATION_HIT_RATE:.3f}.",
            )
        )

    weighted_brier = _safe_float(aggregates.get("weighted_brier"))
    if weighted_brier is not None and weighted_brier > MAX_BRIER:
        alerts.append(
            Alert("ALERT", f"Weighted brier {weighted_brier:.3f} is above {MAX_BRIER:.3f}.")
        )

    hr_row = next((r for r in training_rows if r["target"] == "home_runs"), None)
    if hr_row is not None:
        hr_auc = _safe_float(hr_row.get("auc"))
        if hr_auc is None:
            alerts.append(Alert("ALERT", "home_runs: missing AUC metric."))
        elif hr_auc < MIN_HOME_RUN_AUC:
            alerts.append(
                Alert("ALERT", f"home_runs AUC {hr_auc:.3f} is below {MIN_HOME_RUN_AUC:.3f}.")
            )

    total_bets = int((market_summary.get("totals") or {}).get("total_bet_recommendations") or 0)
    if total_bets < MIN_TOTAL_BET_RECOMMENDATIONS:
        alerts.append(
            Alert(
                "ALERT",
                f"Total actionable recommendations {total_bets} is below {MIN_TOTAL_BET_RECOMMENDATIONS}.",
            )
        )

    overall_odds_coverage = _safe_float((market_summary.get("totals") or {}).get("overall_odds_coverage"))
    if overall_odds_coverage is not None and overall_odds_coverage < WARN_MIN_ODDS_COVERAGE:
        alerts.append(
            Alert(
                "WARN",
                f"Overall odds coverage {overall_odds_coverage:.3f} is below warning threshold {WARN_MIN_ODDS_COVERAGE:.3f}.",
            )
        )

    for row in market_summary.get("targets", []) or []:
        target = row.get("target")
        ev_samples = int(row.get("ev_samples") or 0)
        if ev_samples < WARN_MIN_EV_SAMPLES_PER_TARGET:
            alerts.append(
                Alert(
                    "WARN",
                    f"{target}: EV samples {ev_samples} below warning threshold {WARN_MIN_EV_SAMPLES_PER_TARGET}.",
                )
            )

    return alerts


def _fmt_pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


def _fmt_num(x: Optional[float], decimals: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{decimals}f}"


def _render_markdown(training_rows: List[dict], aggregates: dict, market_summary: dict, alerts: List[Alert]) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    blocking_alerts = [a for a in alerts if a.severity == "ALERT"]
    warn_alerts = [a for a in alerts if a.severity == "WARN"]

    lines: List[str] = []
    lines.append("# MLB Weekly Model Health Report")
    lines.append("")
    lines.append(f"Generated: {now}")
    lines.append("")

    if blocking_alerts:
        lines.append("## Status")
        lines.append("")
        lines.append("FAIL (one or more alert thresholds breached)")
        lines.append("")
    else:
        lines.append("## Status")
        lines.append("")
        lines.append("PASS (no alert-threshold breaches)")
        lines.append("")

    lines.append("## Aggregate Metrics")
    lines.append("")
    lines.append(f"- Weighted recommendation hit rate: {_fmt_pct(_safe_float(aggregates.get('weighted_recommendation_hit_rate')))}")
    lines.append(f"- Weighted brier score: {_fmt_num(_safe_float(aggregates.get('weighted_brier')))}")
    lines.append(f"- Targets evaluated: {int(aggregates.get('targets_evaluated') or 0)}")
    lines.append("")

    lines.append("## Training Metrics by Target")
    lines.append("")
    lines.append("| Target | Status | Test Samples | Rec Hit Rate | Brier | AUC | Rec Rate |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in training_rows:
        lines.append(
            "| {target} | {status} | {test_samples} | {rec_hit} | {brier} | {auc} | {rec_rate} |".format(
                target=row["target"],
                status=row["status"],
                test_samples=int(row["test_samples"]),
                rec_hit=_fmt_pct(_safe_float(row.get("recommendation_hit_rate"))),
                brier=_fmt_num(_safe_float(row.get("brier"))),
                auc=_fmt_num(_safe_float(row.get("auc"))),
                rec_rate=_fmt_pct(_safe_float(row.get("recommendation_rate"))),
            )
        )
    lines.append("")

    totals = market_summary.get("totals", {})
    lines.append("## Market and EV Coverage")
    lines.append("")
    lines.append(f"- Total props scored: {int(totals.get('total_props') or 0)}")
    lines.append(f"- Props with live odds: {int(totals.get('props_with_odds') or 0)} ({_fmt_pct(_safe_float(totals.get('overall_odds_coverage')))})")
    lines.append(f"- Total actionable recommendations: {int(totals.get('total_bet_recommendations') or 0)}")
    lines.append("")

    lines.append("| Market Target | Samples | Odds Coverage | EV Samples | EV Positive Rate | Avg Edge | Avg EV / Unit | Actionable Bets |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in market_summary.get("targets", []) or []:
        lines.append(
            "| {target} | {samples} | {odds_cov} | {ev_samples} | {ev_pos_rate} | {avg_edge} | {avg_ev} | {bets} |".format(
                target=row.get("target"),
                samples=int(row.get("samples") or 0),
                odds_cov=_fmt_pct(_safe_float(row.get("odds_coverage"))),
                ev_samples=int(row.get("ev_samples") or 0),
                ev_pos_rate=_fmt_pct(_safe_float(row.get("ev_positive_rate"))),
                avg_edge=_fmt_num(_safe_float(row.get("avg_edge"))),
                avg_ev=_fmt_num(_safe_float(row.get("avg_ev_per_unit"))),
                bets=int(row.get("bet_recommendations") or 0),
            )
        )
    lines.append("")

    lines.append("## Alerts")
    lines.append("")
    if not alerts:
        lines.append("- None")
    else:
        for a in alerts:
            lines.append(f"- [{a.severity}] {a.message}")

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate MLB model health report")
    parser.add_argument(
        "--no-fail-on-alert",
        action="store_true",
        help="Do not return non-zero exit code on alert-threshold breaches",
    )
    args = parser.parse_args()

    missing = [str(p) for p in (METRICS_FILE, PREDICTIONS_FILE) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Required input file(s) missing: {', '.join(missing)}")

    metrics = _load_json(METRICS_FILE)
    predictions = _load_json(PREDICTIONS_FILE)

    training_rows, aggregates = _summarize_training_metrics(metrics)
    market_summary = _summarize_prediction_markets(predictions)
    alerts = _evaluate_alerts(training_rows, aggregates, market_summary)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    report_md = _render_markdown(training_rows, aggregates, market_summary, alerts)
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write(report_md)

    report_json = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "fail" if any(a.severity == "ALERT" for a in alerts) else "pass",
        "thresholds": {
            "min_recommendation_hit_rate": MIN_RECOMMENDATION_HIT_RATE,
            "max_brier": MAX_BRIER,
            "min_home_run_auc": MIN_HOME_RUN_AUC,
            "min_total_bet_recommendations": MIN_TOTAL_BET_RECOMMENDATIONS,
            "warn_min_odds_coverage": WARN_MIN_ODDS_COVERAGE,
            "warn_min_ev_samples_per_target": WARN_MIN_EV_SAMPLES_PER_TARGET,
        },
        "aggregate_metrics": aggregates,
        "training_by_target": training_rows,
        "market_summary": market_summary,
        "alerts": [a.__dict__ for a in alerts],
    }
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report_json, f, indent=2)

    print(f"Report written: {REPORT_MD}")
    print(f"Report data  : {REPORT_JSON}")

    has_alert = any(a.severity == "ALERT" for a in alerts)
    if has_alert:
        print("\nALERT thresholds breached:")
        for a in alerts:
            if a.severity == "ALERT":
                print(f"  - {a.message}")

    if has_alert and not args.no_fail_on_alert:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
