"""
Train ensemble MLB prop models — hitters and pitchers separately.

Architecture mirrors the NBA ensemble:
  Stacked Ensemble (XGBoost + RandomForest + ExtraTrees -> Ridge meta-learner)
  Walk-forward cross-validation per player
  Residual sigma calibration via normal CDF for over-probability

Hitter targets  : hits, home_runs, rbis, batter_strikeouts
Pitcher targets : strikeouts, hits_allowed, earned_runs

Outputs:
  mlb_models/mlb_hitter_<target>_ensemble.joblib
  mlb_models/mlb_pitcher_<target>_ensemble.joblib
  mlb_ensemble_metrics.json
  mlb_prop_predictions_ml.json

NOTE: Model accuracy improves significantly once pitcher-matchup and
      platoon-split features are added to mlb_player_data.json.
      Current features are purely player-history based.
"""

import json
import math
from pathlib import Path
from statistics import mean

import numpy as np
from joblib import dump
from scipy.stats import norm
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    StackingClassifier,
    StackingRegressor,
)
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from xgboost import XGBClassifier, XGBRegressor

ROOT = Path(__file__).parent
DATA_FILE = ROOT / "mlb_player_data.json"
MODELS_DIR = ROOT / "mlb_models"
METRICS_FILE = ROOT / "mlb_ensemble_metrics.json"
PREDICTIONS_FILE = ROOT / "mlb_prop_predictions_ml.json"

# Baseball needs longer windows — higher per-game variance than basketball
WINDOWS = [5, 10, 20]
MIN_GAMES_HITTER = 15
MIN_GAMES_PITCHER = 8   # SPs have fewer games per season
TEST_FRAC = 0.2

HITTER_TARGETS = ["hits", "home_runs", "rbis", "batter_strikeouts"]
PITCHER_TARGETS = ["strikeouts", "hits_allowed", "earned_runs"]

# Season stat keys that map to each target
HITTER_SEASON_KEYS = {
    "hits": "avg_hits",
    "home_runs": "avg_home_runs",
    "rbis": "avg_rbis",
    "batter_strikeouts": "avg_batter_strikeouts",
}
PITCHER_SEASON_KEYS = {
    "strikeouts": "avg_strikeouts",
    "hits_allowed": "avg_hits_allowed",
    "earned_runs": "avg_earned_runs",
}

# Target-specific thresholds and EV guardrails.
TARGET_THRESHOLDS = {
    "default": (0.55, 0.45),
    "home_runs": (0.62, 0.38),
}
MIN_EDGE_BY_TARGET = {
    "default": 0.03,
    "home_runs": 0.05,
}

MARKET_FEATURE_NAMES = [
    "mkt_has_line",
    "mkt_line",
    "mkt_over_odds",
    "mkt_under_odds",
    "mkt_implied_over",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _float(v):
    return float(v or 0.0)


def _avg(values):
    return float(sum(values) / len(values)) if values else 0.0


def _std(values):
    return float(np.std(values)) if len(values) >= 2 else 0.0


def history_slice(values, n):
    return values[-n:] if len(values) >= n else values


def load_data():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def is_pitcher(position):
    return (position or "").upper() in {"P", "SP", "RP"}


def realistic_line_hitter(avg, target):
    """Conservative line relative to average — mirrors sportsbook tendencies."""
    if target == "home_runs":
        # HR props are always 0.5 (anytime HR yes/no)
        return 0.5
    if target == "hits":
        # Most common: 0.5 or 1.5
        return 0.5 if avg < 1.0 else 1.5
    if target == "rbis":
        return 0.5
    if target == "batter_strikeouts":
        return 0.5 if avg < 0.8 else 1.5
    return max(0.5, round(avg * 0.8 * 2) / 2)


def realistic_line_pitcher(avg, target):
    if target == "strikeouts":
        # Common lines: 4.5, 5.5, 6.5, 7.5 for starters; 0.5-2.5 for relievers
        if avg < 2.0:
            return 0.5
        return max(0.5, round((avg * 0.85) * 2) / 2)
    if target == "hits_allowed":
        return max(0.5, round((avg * 0.90) * 2) / 2)
    if target == "earned_runs":
        return 0.5 if avg < 1.0 else 1.5
    return max(0.5, round(avg * 2) / 2)


def get_thresholds(target):
    return TARGET_THRESHOLDS.get(target, TARGET_THRESHOLDS["default"])


def recommendation(prob, target="default"):
    over_th, under_th = get_thresholds(target)
    if prob > over_th:
        return "OVER"
    if prob < under_th:
        return "UNDER"
    return "NO BET"


def american_to_implied_prob(odds):
    """Convert American odds to implied probability (0..1)."""
    o = int(odds)
    if o > 0:
        return 100.0 / (o + 100.0)
    return abs(o) / (abs(o) + 100.0)


def expected_value_per_unit(win_prob, american_odds):
    """Expected value per 1.0 unit stake at given American odds."""
    o = int(american_odds)
    payout = (o / 100.0) if o > 0 else (100.0 / abs(o))
    lose_prob = 1.0 - float(win_prob)
    return float(win_prob) * payout - lose_prob


def get_target_min_edge(target):
    return float(MIN_EDGE_BY_TARGET.get(target, MIN_EDGE_BY_TARGET["default"]))


def coerce_american_odds(value):
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def build_market_feature_vector(market_for_target):
    """Build target-specific market features from market_odds/prop_odds payload."""
    m = market_for_target if isinstance(market_for_target, dict) else {}
    line = m.get("line")
    over = m.get("over")
    under = m.get("under")
    implied_over = m.get("implied_over")

    over_i = coerce_american_odds(over)
    under_i = coerce_american_odds(under)
    has_line = 1.0 if line is not None else 0.0

    if implied_over is None and over_i is not None:
        implied_over = american_to_implied_prob(over_i)

    return [
        has_line,
        _float(line) if line is not None else 0.0,
        _float(over_i) if over_i is not None else 0.0,
        _float(under_i) if under_i is not None else 0.0,
        _float(implied_over) if implied_over is not None else 0.0,
    ]


def get_prop_odds_for_target(player, target):
    """
    Optional odds lookup.
    Supports either:
      player['prop_odds'][target] = {'over': -110, 'under': -110}
      player['odds'][target] = {'over': -110, 'under': -110}
    """
    for top_key in ["prop_odds", "odds"]:
        top = player.get(top_key, {})
        if not isinstance(top, dict):
            continue
        target_odds = top.get(target)
        if not isinstance(target_odds, dict):
            continue
        return {
            "line": _float(target_odds.get("line")) if target_odds.get("line") is not None else None,
            "over": coerce_american_odds(target_odds.get("over")),
            "under": coerce_american_odds(target_odds.get("under")),
        }
    return {"line": None, "over": None, "under": None}


def apply_ev_filter(prob_over, target, raw_rec, over_odds, under_odds):
    """
    Apply edge/EV filter to a threshold-based recommendation.
    If odds are unavailable, leaves recommendation unchanged but marks filter as not applied.
    """
    result = {
        "raw_recommendation": raw_rec,
        "recommendation": raw_rec,
        "ev_filter_applied": False,
        "edge": None,
        "ev_per_unit": None,
        "odds_used": None,
        "min_edge_required": get_target_min_edge(target),
    }

    if raw_rec not in {"OVER", "UNDER"}:
        return result

    side_prob = float(prob_over) if raw_rec == "OVER" else float(1.0 - prob_over)
    side_odds = over_odds if raw_rec == "OVER" else under_odds
    if side_odds is None:
        return result

    implied = float(american_to_implied_prob(side_odds))
    edge = side_prob - implied
    ev = expected_value_per_unit(side_prob, side_odds)
    min_edge = get_target_min_edge(target)

    result["ev_filter_applied"] = True
    result["edge"] = round(edge, 4)
    result["ev_per_unit"] = round(ev, 4)
    result["odds_used"] = side_odds

    if edge < min_edge or ev <= 0.0:
        result["recommendation"] = "NO BET"

    return result


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def hitter_position_flags(pos):
    p = (pos or "").upper()
    return {
        "is_catcher":   1.0 if p == "C"  else 0.0,
        "is_infield":   1.0 if p in {"1B", "2B", "3B", "SS"} else 0.0,
        "is_outfield":  1.0 if "OF" in p or p in {"LF", "CF", "RF"} else 0.0,
        "is_dh":        1.0 if p == "DH" else 0.0,
    }


def pitcher_position_flags(pos):
    p = (pos or "").upper()
    return {
        "is_starter":  1.0 if p == "SP" else 0.0,
        "is_reliever": 1.0 if p == "RP" else 0.0,
    }


def extract_matchup_features(context):
    """
    Matchup/context scaffold.
    Uses defaults when fields are unavailable; this keeps schema stable while
    allowing immediate gains once context data is populated.
    """
    ctx = context if isinstance(context, dict) else {}
    return {
        "is_home": 1.0 if bool(ctx.get("is_home", False)) else 0.0,
        "park_factor": _float(ctx.get("park_factor", 1.0)),
        "opponent_starter_era": _float(ctx.get("opponent_starter_era", 4.2)),
        "opponent_starter_k9": _float(ctx.get("opponent_starter_k9", 8.5)),
        "opponent_team_k_rate": _float(ctx.get("opponent_team_k_rate", 0.22)),
        "temperature_f": _float(ctx.get("temperature_f", 70.0)),
        "wind_out_mph": _float(ctx.get("wind_out_mph", 0.0)),
    }


def make_hitter_feature_names():
    stats = HITTER_TARGETS  # hits, home_runs, rbis, batter_strikeouts
    names = ["games_in_history", "is_catcher", "is_infield", "is_outfield", "is_dh"]
    for stat in stats:
        names.append(f"{stat}_last1")
        for w in WINDOWS:
            names.extend([
                f"{stat}_avg_{w}",
                f"{stat}_std_{w}",
                f"{stat}_max_{w}",
                f"{stat}_min_{w}",
            ])
        names.append(f"{stat}_trend_5v20")
    for target in HITTER_TARGETS:
        names.append(f"season_{target}")
    names.extend([
        "ctx_is_home",
        "ctx_park_factor",
        "ctx_opp_starter_era",
        "ctx_opp_starter_k9",
        "ctx_opp_team_k_rate",
        "ctx_temperature_f",
        "ctx_wind_out_mph",
    ])
    return names


def make_pitcher_feature_names():
    stats = PITCHER_TARGETS  # strikeouts, hits_allowed, earned_runs
    names = ["games_in_history", "is_starter", "is_reliever"]
    for stat in stats:
        names.append(f"{stat}_last1")
        for w in WINDOWS:
            names.extend([
                f"{stat}_avg_{w}",
                f"{stat}_std_{w}",
                f"{stat}_max_{w}",
                f"{stat}_min_{w}",
            ])
        names.append(f"{stat}_trend_5v20")
    for target in PITCHER_TARGETS:
        names.append(f"season_{target}")
    names.extend([
        "ctx_is_home",
        "ctx_park_factor",
        "ctx_opp_starter_era",
        "ctx_opp_starter_k9",
        "ctx_opp_team_k_rate",
        "ctx_temperature_f",
        "ctx_wind_out_mph",
    ])
    return names


def build_hitter_features(history_games, season_stats, position, context=None):
    flags = hitter_position_flags(position)
    feats = [
        float(len(history_games)),
        flags["is_catcher"],
        flags["is_infield"],
        flags["is_outfield"],
        flags["is_dh"],
    ]

    # Map gamelog field 'strikeouts' to 'batter_strikeouts' for hitters
    def get_stat(game, stat):
        if stat == "batter_strikeouts":
            return _float(game.get("strikeouts", 0))
        return _float(game.get(stat, 0))

    for stat in HITTER_TARGETS:
        vals = [get_stat(g, stat) for g in history_games]
        feats.append(vals[-1] if vals else 0.0)
        for w in WINDOWS:
            s = history_slice(vals, w)
            feats.extend([_avg(s), _std(s), max(s) if s else 0.0, min(s) if s else 0.0])
        short = history_slice(vals, 5)
        long = history_slice(vals, 20)
        feats.append(_avg(short) - _avg(long))

    # Season averages — derive avg_home_runs / avg_rbis / avg_batter_strikeouts if missing
    total_hr = _float(season_stats.get("total_home_runs", 0))
    total_rbis = _float(season_stats.get("total_rbis", 0))
    gp = max(1.0, _float(season_stats.get("games_played", 1)))

    derived = {
        "avg_hits":              _float(season_stats.get("avg_hits", 0)),
        "avg_home_runs":         total_hr / gp,
        "avg_rbis":              total_rbis / gp,
        "avg_batter_strikeouts": 0.0,  # not in season_stats yet; use 0 until data is enriched
    }
    for target in HITTER_TARGETS:
        feats.append(derived[HITTER_SEASON_KEYS[target]])

    m = extract_matchup_features(context)
    feats.extend([
        m["is_home"],
        m["park_factor"],
        m["opponent_starter_era"],
        m["opponent_starter_k9"],
        m["opponent_team_k_rate"],
        m["temperature_f"],
        m["wind_out_mph"],
    ])

    return feats


def build_pitcher_features(history_games, season_stats, position, context=None):
    flags = pitcher_position_flags(position)
    feats = [
        float(len(history_games)),
        flags["is_starter"],
        flags["is_reliever"],
    ]

    for stat in PITCHER_TARGETS:
        vals = [_float(g.get(stat, 0)) for g in history_games]
        feats.append(vals[-1] if vals else 0.0)
        for w in WINDOWS:
            s = history_slice(vals, w)
            feats.extend([_avg(s), _std(s), max(s) if s else 0.0, min(s) if s else 0.0])
        short = history_slice(vals, 5)
        long = history_slice(vals, 20)
        feats.append(_avg(short) - _avg(long))

    for target in PITCHER_TARGETS:
        key = PITCHER_SEASON_KEYS[target]
        feats.append(_float(season_stats.get(key, 0)))

    m = extract_matchup_features(context)
    feats.extend([
        m["is_home"],
        m["park_factor"],
        m["opponent_starter_era"],
        m["opponent_starter_k9"],
        m["opponent_team_k_rate"],
        m["temperature_f"],
        m["wind_out_mph"],
    ])

    return feats


# ---------------------------------------------------------------------------
# Dataset builder (walk-forward, same pattern as NBA)
# ---------------------------------------------------------------------------

def build_dataset(players):
    hitter_feature_names = make_hitter_feature_names()
    pitcher_feature_names = make_pitcher_feature_names()

    hitter_data = {t: [] for t in HITTER_TARGETS}
    pitcher_data = {t: [] for t in PITCHER_TARGETS}

    for player in players:
        gamelog = player.get("gamelog", [])
        position = player.get("position", "")
        season_stats = player.get("season_stats", {})
        pitcher = is_pitcher(position)

        min_games = MIN_GAMES_PITCHER if pitcher else MIN_GAMES_HITTER
        if len(gamelog) < min_games:
            continue

        # Oldest → newest for walk-forward
        games = list(reversed(gamelog))
        n = len(games)
        test_cut = max(1, int(math.ceil(n * TEST_FRAC)))

        targets = PITCHER_TARGETS if pitcher else HITTER_TARGETS
        data_dict = pitcher_data if pitcher else hitter_data
        start_idx = 5 if pitcher else 8  # minimum history before first sample

        for i in range(start_idx, n):
            history = games[:i]
            current = games[i]
            in_test = i >= (n - test_cut)

            # Matchup/context scaffold from gamelog row if available.
            context = current.get("matchup_context", {}) if isinstance(current, dict) else {}

            if pitcher:
                feats = build_pitcher_features(history, season_stats, position, context)
            else:
                feats = build_hitter_features(history, season_stats, position, context)

            for target in targets:
                if target == "batter_strikeouts":
                    y_val = _float(current.get("strikeouts", 0))
                else:
                    y_val = _float(current.get(target, 0))

                hist_vals = []
                for g in history:
                    if target == "batter_strikeouts":
                        hist_vals.append(_float(g.get("strikeouts", 0)))
                    else:
                        hist_vals.append(_float(g.get(target, 0)))

                data_dict[target].append({
                    "x": feats,
                    "x_market": build_market_feature_vector((current.get("market_odds") or {}).get(target, {})),
                    "y": y_val,
                    "is_test": in_test,
                    "player_name": player.get("player_name"),
                    "team": player.get("team"),
                    "position": position,
                    "game_id": current.get("game_id"),
                    "hist_avg": _avg(hist_vals),
                    "season_stats": season_stats,
                })

    for target in HITTER_TARGETS:
        hitter_data[target].sort(key=lambda r: (r["player_name"] or "", r["game_id"] or ""))
    for target in PITCHER_TARGETS:
        pitcher_data[target].sort(key=lambda r: (r["player_name"] or "", r["game_id"] or ""))

    return hitter_data, pitcher_data, hitter_feature_names, pitcher_feature_names


# ---------------------------------------------------------------------------
# Model definition (same architecture as NBA)
# ---------------------------------------------------------------------------

def build_model(random_state=42):
    xgb = XGBRegressor(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=random_state,
        n_jobs=4,
    )
    rf = RandomForestRegressor(
        n_estimators=450,
        max_depth=14,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=4,
    )
    et = ExtraTreesRegressor(
        n_estimators=450,
        max_depth=16,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=4,
    )
    final = RidgeCV(alphas=np.logspace(-3, 3, 20))

    return StackingRegressor(
        estimators=[("xgb", xgb), ("rf", rf), ("et", et)],
        final_estimator=final,
        passthrough=True,
        n_jobs=4,
    )


def build_home_run_model(random_state=42):
    """Binary ensemble tuned for HR over 0.5 probability."""
    xgb = XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=4,
    )

    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=16,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=4,
    )

    et = ExtraTreesClassifier(
        n_estimators=500,
        max_depth=18,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=4,
    )

    final = LogisticRegression(max_iter=1200, class_weight="balanced")

    return StackingClassifier(
        estimators=[("xgb", xgb), ("rf", rf), ("et", et)],
        final_estimator=final,
        passthrough=True,
        stack_method="predict_proba",
        n_jobs=4,
    )


# ---------------------------------------------------------------------------
# Calibration helpers (identical to NBA)
# ---------------------------------------------------------------------------

def eval_calibration(probs, actuals):
    buckets = {}
    for p, a in zip(probs, actuals):
        idx = int(min(0.9999, max(0.0, float(p))) * 10)
        lo = idx / 10
        key = f"{lo:.1f}-{lo+0.1:.1f}"
        if key not in buckets:
            buckets[key] = {"n": 0, "pred": 0.0, "actual": 0.0}
        buckets[key]["n"] += 1
        buckets[key]["pred"] += float(p)
        buckets[key]["actual"] += int(a)

    out = []
    for k in sorted(buckets):
        b = buckets[k]
        out.append({
            "bucket": k,
            "n": b["n"],
            "avg_pred": round(b["pred"] / b["n"], 4),
            "actual_rate": round(b["actual"] / b["n"], 4),
        })
    return out


# ---------------------------------------------------------------------------
# Train + evaluate one group (hitters or pitchers)
# ---------------------------------------------------------------------------

def train_group(data_dict, feature_names, group_label, line_fn):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    targets = list(data_dict.keys())
    metrics = {}
    fitted = {}

    for target in targets:
        rows = data_dict[target]
        train_rows = [r for r in rows if not r["is_test"]]
        test_rows  = [r for r in rows if r["is_test"]]

        print(f"  [{group_label}] {target}: train={len(train_rows)} test={len(test_rows)}")

        if len(train_rows) < 100 or len(test_rows) < 30:
            print(f"    ⚠  Skipping {target} — not enough samples")
            continue

        x_train = np.array([r["x"] + r.get("x_market", [0.0] * len(MARKET_FEATURE_NAMES)) for r in train_rows], dtype=np.float32)
        y_train = np.array([r["y"] for r in train_rows], dtype=np.float32)
        x_test  = np.array([r["x"] + r.get("x_market", [0.0] * len(MARKET_FEATURE_NAMES)) for r in test_rows],  dtype=np.float32)
        y_test  = np.array([r["y"] for r in test_rows],  dtype=np.float32)

        # Coverage of rows with attached market line (odds-informed training signal).
        train_market_rows = int(np.sum(np.array([r.get("x_market", [0.0])[0] for r in train_rows], dtype=np.float32) > 0))
        test_market_rows = int(np.sum(np.array([r.get("x_market", [0.0])[0] for r in test_rows], dtype=np.float32) > 0))
        train_market_rate = float(train_market_rows / len(train_rows)) if train_rows else 0.0
        test_market_rate = float(test_market_rows / len(test_rows)) if test_rows else 0.0

        train_nonzero = int(np.sum(y_train > 0))
        test_nonzero = int(np.sum(y_test > 0))
        train_nonzero_rate = float(train_nonzero / len(y_train))
        test_nonzero_rate = float(test_nonzero / len(y_test))

        print(
            f"    Label balance: train_nonzero={train_nonzero}/{len(y_train)} ({train_nonzero_rate:.3f}) "
            f"test_nonzero={test_nonzero}/{len(y_test)} ({test_nonzero_rate:.3f})"
        )
        print(
            f"    Market coverage: train={train_market_rows}/{len(train_rows)} ({train_market_rate:.3f}) "
            f"test={test_market_rows}/{len(test_rows)} ({test_market_rate:.3f})"
        )

        # Guardrail: skip degenerate all-zero / constant targets.
        train_unique = np.unique(y_train)
        test_unique = np.unique(y_test)
        if len(train_unique) <= 1 or len(test_unique) <= 1:
            reason = (
                f"degenerate target distribution (train_unique={train_unique.tolist()}, "
                f"test_unique={test_unique.tolist()})"
            )
            print(f"    ⚠  Skipping {target} — {reason}")
            metrics[target] = {
                "status": "skipped",
                "skip_reason": reason,
                "train_samples": int(len(train_rows)),
                "test_samples": int(len(test_rows)),
                "train_nonzero": train_nonzero,
                "test_nonzero": test_nonzero,
                "train_nonzero_rate": round(train_nonzero_rate, 4),
                "test_nonzero_rate": round(test_nonzero_rate, 4),
                "train_market_rows": train_market_rows,
                "test_market_rows": test_market_rows,
                "train_market_rate": round(train_market_rate, 4),
                "test_market_rate": round(test_market_rate, 4),
            }
            continue

        # HR-specific path: train directly on binary HR event (line 0.5).
        if group_label == "hitter" and target == "home_runs":
            y_train_bin = (y_train > 0).astype(int)
            y_test_bin = (y_test > 0).astype(int)

            # Guardrail: require sufficient positives to train a meaningful binary model.
            if int(np.sum(y_train_bin)) < 50 or int(np.sum(y_test_bin)) < 10:
                reason = (
                    "insufficient positive HR samples for binary ensemble "
                    f"(train_pos={int(np.sum(y_train_bin))}, test_pos={int(np.sum(y_test_bin))})"
                )
                print(f"    ⚠  Skipping {target} — {reason}")
                metrics[target] = {
                    "status": "skipped",
                    "skip_reason": reason,
                    "train_samples": int(len(train_rows)),
                    "test_samples": int(len(test_rows)),
                    "train_nonzero": train_nonzero,
                    "test_nonzero": test_nonzero,
                    "train_nonzero_rate": round(train_nonzero_rate, 4),
                    "test_nonzero_rate": round(test_nonzero_rate, 4),
                    "train_market_rows": train_market_rows,
                    "test_market_rows": test_market_rows,
                    "train_market_rate": round(train_market_rate, 4),
                    "test_market_rate": round(test_market_rate, 4),
                }
                continue

            model = build_home_run_model(42)

            # Upweight positive HR events to fight severe class imbalance.
            pos_count = max(1, int(np.sum(y_train_bin == 1)))
            neg_count = max(1, int(np.sum(y_train_bin == 0)))
            pos_weight = float(neg_count / pos_count)
            sample_weight = np.where(y_train_bin == 1, pos_weight, 1.0).astype(np.float32)

            model.fit(x_train, y_train_bin, sample_weight=sample_weight)

            probs_test = model.predict_proba(x_test)[:, 1]
            dir_preds = (probs_test >= 0.5).astype(int)

            directional_accuracy = float(np.mean(dir_preds == y_test_bin))
            recs = np.array([recommendation(p, target) for p in probs_test])
            rec_mask = recs != "NO BET"
            rec_hit_rate = (
                float(np.mean(np.where(recs[rec_mask] == "OVER", 1, 0) == y_test_bin[rec_mask]))
                if np.any(rec_mask)
                else None
            )

            brier = float(np.mean((probs_test - y_test_bin) ** 2))
            prob_mae = float(np.mean(np.abs(probs_test - y_test_bin)))
            auc = float(roc_auc_score(y_test_bin, probs_test))

            metrics[target] = {
                "status": "ok",
                "mode": "binary_hr_classifier",
                "train_samples": int(len(train_rows)),
                "test_samples": int(len(test_rows)),
                "train_nonzero": train_nonzero,
                "test_nonzero": test_nonzero,
                "train_nonzero_rate": round(train_nonzero_rate, 4),
                "test_nonzero_rate": round(test_nonzero_rate, 4),
                "train_market_rows": train_market_rows,
                "test_market_rows": test_market_rows,
                "train_market_rate": round(train_market_rate, 4),
                "test_market_rate": round(test_market_rate, 4),
                "directional_accuracy": round(directional_accuracy, 4),
                "recommendation_hit_rate": round(rec_hit_rate, 4) if rec_hit_rate is not None else None,
                "recommendation_rate": round(float(np.mean(rec_mask)), 4),
                "probability_mae": round(prob_mae, 4),
                "brier": round(brier, 4),
                "auc": round(auc, 4),
                "calibration": eval_calibration(probs_test, y_test_bin),
                "line": 0.5,
            }

            artifact = {
                "model": model,
                "target": target,
                "feature_names": feature_names + MARKET_FEATURE_NAMES,
                "group": group_label,
                "prediction_mode": "binary_over_0_5",
            }
            model_path = MODELS_DIR / f"mlb_{group_label}_{target}_ensemble.joblib"
            dump(artifact, model_path)
            fitted[target] = artifact

            rec_hit_rate_display = f"{rec_hit_rate:.3f}" if rec_hit_rate is not None else "N/A"
            print(
                f"    Mode=HR_BINARY  AUC={auc:.3f}  DirAcc={directional_accuracy:.3f}  "
                f"Brier={brier:.3f}  RecHitRate={rec_hit_rate_display}"
            )
            continue

        model = build_model(42)
        model.fit(x_train, y_train)

        pred_train = model.predict(x_train)
        pred_test  = model.predict(x_test)

        residual_sigma = float(np.std(y_train - pred_train))
        residual_sigma = max(0.5, residual_sigma)

        mae  = float(mean_absolute_error(y_test, pred_test))
        rmse = float(mean_squared_error(y_test, pred_test) ** 0.5)
        r2   = float(r2_score(y_test, pred_test))

        lines = np.array(
            [line_fn(r["hist_avg"], target) for r in test_rows], dtype=np.float32
        )
        actual_over = (y_test > lines).astype(int)

        z = (lines - pred_test) / residual_sigma
        probs_over = 1.0 - norm.cdf(z)

        dir_preds = (probs_over >= 0.5).astype(int)
        directional_accuracy = float(np.mean(dir_preds == actual_over))

        recs = np.array([recommendation(p, target) for p in probs_over])
        rec_mask = recs != "NO BET"
        rec_hit_rate = (
            float(np.mean(
                np.where(recs[rec_mask] == "OVER", 1, 0) == actual_over[rec_mask]
            )) if np.any(rec_mask) else None
        )

        brier = float(np.mean((probs_over - actual_over) ** 2))
        prob_mae = float(np.mean(np.abs(probs_over - actual_over)))

        metrics[target] = {
            "status": "ok",
            "mode": "regression_plus_residual_calibration",
            "train_samples": int(len(train_rows)),
            "test_samples":  int(len(test_rows)),
            "train_nonzero": train_nonzero,
            "test_nonzero": test_nonzero,
            "train_nonzero_rate": round(train_nonzero_rate, 4),
            "test_nonzero_rate": round(test_nonzero_rate, 4),
            "train_market_rows": train_market_rows,
            "test_market_rows": test_market_rows,
            "train_market_rate": round(train_market_rate, 4),
            "test_market_rate": round(test_market_rate, 4),
            "mae":  round(mae,  4),
            "rmse": round(rmse, 4),
            "r2":   round(r2,   4),
            "residual_sigma": round(residual_sigma, 4),
            "directional_accuracy": round(directional_accuracy, 4),
            "recommendation_hit_rate": round(rec_hit_rate, 4) if rec_hit_rate is not None else None,
            "recommendation_rate": round(float(np.mean(rec_mask)), 4),
            "probability_mae": round(prob_mae, 4),
            "brier": round(brier, 4),
            "calibration": eval_calibration(probs_over, actual_over),
        }

        artifact = {
            "model": model,
            "target": target,
            "feature_names": feature_names + MARKET_FEATURE_NAMES,
            "residual_sigma": residual_sigma,
            "group": group_label,
            "prediction_mode": "regression",
        }
        model_path = MODELS_DIR / f"mlb_{group_label}_{target}_ensemble.joblib"
        dump(artifact, model_path)
        fitted[target] = artifact

        rec_hit_rate_display = f"{rec_hit_rate:.3f}" if rec_hit_rate is not None else "N/A"

        print(f"    MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.3f}  "
              f"DirAcc={directional_accuracy:.3f}  "
              f"RecHitRate={rec_hit_rate_display}")

    return fitted, metrics


# ---------------------------------------------------------------------------
# Prediction generation
# ---------------------------------------------------------------------------

def predict_for_players(players, hitter_fitted, pitcher_fitted,
                        hitter_feature_names, pitcher_feature_names):
    hitter_predictions = []
    pitcher_predictions = []

    for player in players:
        gamelog = player.get("gamelog", [])
        position = player.get("position", "")
        season_stats = player.get("season_stats", {})
        pitcher = is_pitcher(position)
        player_prop_odds = player.get("prop_odds", {}) if isinstance(player.get("prop_odds", {}), dict) else {}
        has_any_live_odds = bool(player_prop_odds)

        min_games = MIN_GAMES_PITCHER if pitcher else MIN_GAMES_HITTER
        # If live odds exist, allow lower-history predictions so EV pipeline can run.
        # Keep a floor of 3 games to avoid purely empty-history outputs.
        min_games_for_prediction = 3 if has_any_live_odds else min_games
        if len(gamelog) < min_games_for_prediction:
            continue

        games = list(reversed(gamelog))
        history = games  # use all games as history for final prediction
        next_context = player.get("next_game_context", {})

        if pitcher:
            feats = build_pitcher_features(history, season_stats, position, next_context)
            fitted = pitcher_fitted
            targets = PITCHER_TARGETS
            line_fn = realistic_line_pitcher
        else:
            feats = build_hitter_features(history, season_stats, position, next_context)
            fitted = hitter_fitted
            targets = HITTER_TARGETS
            line_fn = realistic_line_hitter

        props = {}

        for target in targets:
            if target not in fitted:
                continue

            artifact = fitted[target]
            model = artifact["model"]
            prediction_mode = artifact.get("prediction_mode", "regression")
            odds = get_prop_odds_for_target(player, target)
            x_market = build_market_feature_vector(odds)
            x_target = np.array([feats + x_market], dtype=np.float32)

            if prediction_mode == "binary_over_0_5":
                prob_over = float(model.predict_proba(x_target)[0, 1])
                pred = prob_over
                # HR line is always 0.5 in this binary mode.
                line = 0.5
                hist_avg = _avg([_float(g.get("home_runs", 0)) for g in history])
            else:
                sigma = artifact["residual_sigma"]
                pred = float(model.predict(x_target)[0])
                pred = max(0.0, pred)  # stats can't be negative

                # History average for line
                if target == "batter_strikeouts":
                    hist_vals = [_float(g.get("strikeouts", 0)) for g in history]
                else:
                    hist_vals = [_float(g.get(target, 0)) for g in history]

                hist_avg = _avg(hist_vals)
                line = line_fn(hist_avg, target)

            if odds.get("line") is not None:
                line = float(odds.get("line"))

            if prediction_mode == "binary_over_0_5":
                # Binary HR model already outputs P(over 0.5).
                pass
            else:
                z = (line - pred) / sigma
                prob_over = float(1.0 - norm.cdf(z))

            raw_rec = recommendation(prob_over, target)
            ev_info = apply_ev_filter(
                prob_over,
                target,
                raw_rec,
                odds.get("over"),
                odds.get("under"),
            )

            props[target] = {
                "line": line,
                "predicted": round(pred, 2),
                "history_avg": round(hist_avg, 2),
                "over_probability": round(prob_over * 100, 1),
                "under_probability": round((1 - prob_over) * 100, 1),
                "raw_recommendation": ev_info["raw_recommendation"],
                "recommendation": ev_info["recommendation"],
                "odds_over": odds.get("over"),
                "odds_under": odds.get("under"),
                "edge": ev_info["edge"],
                "ev_per_unit": ev_info["ev_per_unit"],
                "ev_filter_applied": ev_info["ev_filter_applied"],
                "min_edge_required": ev_info["min_edge_required"],
            }

        entry = {
            "id": player.get("id"),
            "player_name": player.get("player_name"),
            "team": player.get("team"),
            "position": position,
            "games_used": len(gamelog),
            "live_odds_available": has_any_live_odds,
            "below_standard_min_games": len(gamelog) < min_games,
            "props": props,
        }

        if pitcher:
            pitcher_predictions.append(entry)
        else:
            hitter_predictions.append(entry)

    return hitter_predictions, pitcher_predictions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("MLB ENSEMBLE PROP MODEL TRAINING")
    print("Architecture: XGBoost + RandomForest + ExtraTrees → Ridge")
    print("=" * 65)

    players = load_data()
    print(f"\nLoaded {len(players)} MLB players")

    hitters = [p for p in players if not is_pitcher(p.get("position", ""))]
    pitchers = [p for p in players if is_pitcher(p.get("position", ""))]
    print(f"  Hitters: {len(hitters)}  |  Pitchers: {len(pitchers)}")

    print("\nBuilding walk-forward datasets...")
    hitter_data, pitcher_data, hitter_fn, pitcher_fn = build_dataset(players)

    print("\nTraining hitter models...")
    hitter_fitted, hitter_metrics = train_group(
        hitter_data, hitter_fn, "hitter", realistic_line_hitter
    )

    print("\nTraining pitcher models...")
    pitcher_fitted, pitcher_metrics = train_group(
        pitcher_data, pitcher_fn, "pitcher", realistic_line_pitcher
    )

    # Save metrics
    metrics_out = {
        "model_type": "Stacked Ensemble (XGBoost + RandomForest + ExtraTrees -> Ridge)",
        "windows": WINDOWS,
        "hitter_features": hitter_fn,
        "pitcher_features": pitcher_fn,
        "market_features": MARKET_FEATURE_NAMES,
        "hitter_targets": hitter_metrics,
        "pitcher_targets": pitcher_metrics,
    }
    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"\nMetrics saved → {METRICS_FILE}")

    # Generate predictions
    print("\nGenerating predictions for all players...")
    hitter_preds, pitcher_preds = predict_for_players(
        players, hitter_fitted, pitcher_fitted, hitter_fn, pitcher_fn
    )

    predictions_out = {
        "hitters": hitter_preds,
        "pitchers": pitcher_preds,
    }
    with open(PREDICTIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(predictions_out, f, indent=2)

    print(f"\n{'=' * 65}")
    print("COMPLETE")
    print(f"  Hitter predictions : {len(hitter_preds)}")
    print(f"  Pitcher predictions: {len(pitcher_preds)}")
    print(f"  Predictions saved  → {PREDICTIONS_FILE}")
    print(f"  Metrics saved      → {METRICS_FILE}")
    print(f"  Models saved       → {MODELS_DIR}/")
    print("=" * 65)

    # Quick sample output
    if hitter_preds:
        s = hitter_preds[0]
        print(f"\nSample hitter — {s['player_name']} ({s['team']}):")
        for prop, v in s["props"].items():
            print(f"  {prop:20s} line={v['line']}  pred={v['predicted']}  "
                  f"OVER {v['over_probability']}%  → {v['recommendation']}")

    if pitcher_preds:
        s = pitcher_preds[0]
        print(f"\nSample pitcher — {s['player_name']} ({s['team']}):")
        for prop, v in s["props"].items():
            print(f"  {prop:20s} line={v['line']}  pred={v['predicted']}  "
                  f"OVER {v['over_probability']}%  → {v['recommendation']}")
