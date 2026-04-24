"""
Train ensemble NBA prop models (points, rebounds, assists, threes_made).

This pipeline builds a real ML model per prop using player game logs and
exports:
- model artifacts (*.joblib)
- evaluation report (nba_ensemble_metrics.json)
- production predictions (nba_prop_predictions_ml.json)
"""

import json
import math
from pathlib import Path
from statistics import mean

import numpy as np
from joblib import dump
from scipy.stats import norm
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

ROOT = Path(__file__).parent
DATA_FILE = ROOT / "nba_player_data.json"
MODELS_DIR = ROOT / "nba_models"
METRICS_FILE = ROOT / "nba_ensemble_metrics.json"
PREDICTIONS_FILE = ROOT / "nba_prop_predictions_ml.json"

TARGETS = ["points", "rebounds", "assists", "threes_made"]
WINDOWS = [3, 5, 10]
MIN_GAMES = 12
TEST_FRAC = 0.2


def load_data():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _float(v):
    return float(v or 0.0)


def _avg(values):
    return float(sum(values) / len(values)) if values else 0.0


def _std(values):
    if len(values) < 2:
        return 0.0
    return float(np.std(values))


def realistic_line(avg):
    if avg < 5:
        line = round((avg * 0.75) * 2) / 2
        return max(0.5, line)
    if avg < 15:
        line = round((avg * 0.65) * 2) / 2
        return max(0.5, line)
    line = round((avg * 0.60) * 2) / 2
    return max(0.5, line)


def recommendation(prob):
    if prob > 0.55:
        return "OVER"
    if prob < 0.45:
        return "UNDER"
    return "NO BET"


def season_key_for_target(target):
    mapping = {
        "points": "ppg",
        "rebounds": "rpg",
        "assists": "apg",
        "threes_made": "three_pm",
    }
    return mapping[target]


def position_flags(pos):
    p = (pos or "").upper()
    return {
        "is_guard": 1.0 if "G" in p else 0.0,
        "is_forward": 1.0 if "F" in p else 0.0,
        "is_center": 1.0 if "C" in p else 0.0,
    }


def make_feature_names():
    names = [
        "games_in_history",
        "is_guard",
        "is_forward",
        "is_center",
    ]
    for stat in TARGETS:
        names.append(f"{stat}_last1")
        for w in WINDOWS:
            names.extend(
                [
                    f"{stat}_avg_{w}",
                    f"{stat}_std_{w}",
                    f"{stat}_max_{w}",
                    f"{stat}_min_{w}",
                ]
            )
        names.append(f"{stat}_trend_3v10")
    for target in TARGETS:
        names.append(f"season_{target}")
    return names


def history_slice(values, n):
    return values[-n:] if len(values) >= n else values


def build_features(history_games, season_stats, position):
    feats = []
    flags = position_flags(position)
    feats.append(float(len(history_games)))
    feats.extend([flags["is_guard"], flags["is_forward"], flags["is_center"]])

    by_stat = {stat: [_float(g.get(stat, 0)) for g in history_games] for stat in TARGETS}

    for stat in TARGETS:
        vals = by_stat[stat]
        feats.append(vals[-1] if vals else 0.0)
        for w in WINDOWS:
            s = history_slice(vals, w)
            feats.extend([_avg(s), _std(s), max(s) if s else 0.0, min(s) if s else 0.0])
        short = history_slice(vals, 3)
        long = history_slice(vals, 10)
        feats.append(_avg(short) - _avg(long))

    for target in TARGETS:
        key = season_key_for_target(target)
        feats.append(_float(season_stats.get(key, 0)))

    return feats


def build_dataset(players):
    feature_names = make_feature_names()
    data = {target: [] for target in TARGETS}

    for player in players:
        gamelog = player.get("gamelog", [])
        if len(gamelog) < MIN_GAMES:
            continue

        # Convert to oldest -> newest for walk-forward samples.
        games = list(reversed(gamelog))
        n = len(games)
        test_cut = max(1, int(math.ceil(n * TEST_FRAC)))

        season_stats = player.get("season_stats", {})
        position = player.get("position", "")

        for i in range(10, n):
            history = games[:i]
            current = games[i]
            feats = build_features(history, season_stats, position)
            in_test = i >= (n - test_cut)

            for target in TARGETS:
                hist_target_vals = [_float(g.get(target, 0)) for g in history]
                hist_avg = _avg(hist_target_vals)
                row = {
                    "x": feats,
                    "y": _float(current.get(target, 0)),
                    "is_test": in_test,
                    "player_name": player.get("player_name"),
                    "team": player.get("team"),
                    "position": position,
                    "game_id": current.get("game_id"),
                    "hist_avg": hist_avg,
                }
                data[target].append(row)

    for target in TARGETS:
        data[target].sort(key=lambda r: (r["player_name"] or "", r["game_id"] or ""))

    return data, feature_names


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

    model = StackingRegressor(
        estimators=[("xgb", xgb), ("rf", rf), ("et", et)],
        final_estimator=final,
        passthrough=True,
        n_jobs=4,
    )
    return model


def eval_calibration(probs, actuals):
    buckets = {}
    for p, a in zip(probs, actuals):
        idx = int(min(0.9999, max(0.0, float(p))) * 10)
        lo = idx / 10
        hi = lo + 0.1
        key = f"{lo:.1f}-{hi:.1f}"
        if key not in buckets:
            buckets[key] = {"n": 0, "pred": 0.0, "actual": 0.0}
        buckets[key]["n"] += 1
        buckets[key]["pred"] += float(p)
        buckets[key]["actual"] += int(a)

    out = []
    for k in sorted(buckets.keys()):
        b = buckets[k]
        out.append(
            {
                "bucket": k,
                "n": b["n"],
                "avg_pred": round(b["pred"] / b["n"], 4),
                "actual_rate": round(b["actual"] / b["n"], 4),
            }
        )
    return out


def train_and_evaluate(dataset, feature_names):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    metrics = {
        "model_type": "Stacked Ensemble (XGBoost + RandomForest + ExtraTrees -> Ridge)",
        "features": feature_names,
        "targets": {},
    }

    fitted = {}

    for target in TARGETS:
        rows = dataset[target]
        train_rows = [r for r in rows if not r["is_test"]]
        test_rows = [r for r in rows if r["is_test"]]

        if len(train_rows) < 200 or len(test_rows) < 50:
            raise RuntimeError(f"Not enough samples for {target}: train={len(train_rows)} test={len(test_rows)}")

        x_train = np.array([r["x"] for r in train_rows], dtype=np.float32)
        y_train = np.array([r["y"] for r in train_rows], dtype=np.float32)
        x_test = np.array([r["x"] for r in test_rows], dtype=np.float32)
        y_test = np.array([r["y"] for r in test_rows], dtype=np.float32)

        model = build_model(42)
        model.fit(x_train, y_train)

        pred_train = model.predict(x_train)
        pred_test = model.predict(x_test)

        residual_sigma = float(np.std(y_train - pred_train))
        residual_sigma = max(0.8, residual_sigma)

        mae = float(mean_absolute_error(y_test, pred_test))
        rmse = float(mean_squared_error(y_test, pred_test) ** 0.5)
        r2 = float(r2_score(y_test, pred_test))

        lines = np.array([realistic_line(r["hist_avg"]) for r in test_rows], dtype=np.float32)
        actual_over = (y_test > lines).astype(int)

        z = (lines - pred_test) / residual_sigma
        probs_over = 1.0 - norm.cdf(z)

        dir_preds = (probs_over >= 0.5).astype(int)
        directional_accuracy = float(np.mean(dir_preds == actual_over))

        recs = np.array([recommendation(p) for p in probs_over])
        rec_mask = recs != "NO BET"
        if np.any(rec_mask):
            rec_preds = np.where(recs[rec_mask] == "OVER", 1, 0)
            rec_hit_rate = float(np.mean(rec_preds == actual_over[rec_mask]))
        else:
            rec_hit_rate = None

        prob_mae = float(np.mean(np.abs(probs_over - actual_over)))
        brier = float(np.mean((probs_over - actual_over) ** 2))

        metrics["targets"][target] = {
            "train_samples": int(len(train_rows)),
            "test_samples": int(len(test_rows)),
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "r2": round(r2, 4),
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
            "feature_names": feature_names,
            "residual_sigma": residual_sigma,
            "line_rule": "realistic_line(history_avg)",
        }

        model_path = MODELS_DIR / f"nba_{target}_ensemble.joblib"
        dump(artifact, model_path)

        fitted[target] = artifact

    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return fitted, metrics


def predict_for_players(players, fitted):
    predictions = []

    for player in players:
        gamelog = player.get("gamelog", [])
        if len(gamelog) < 10:
            continue

        # For current prediction, use all known games as history.
        history = list(reversed(gamelog))
        season_stats = player.get("season_stats", {})
        pos = player.get("position", "")

        feats = np.array(build_features(history, season_stats, pos), dtype=np.float32).reshape(1, -1)

        props = {}
        for target in TARGETS:
            artifact = fitted[target]
            model = artifact["model"]
            sigma = float(artifact["residual_sigma"])

            pred_stat = float(model.predict(feats)[0])
            hist_vals = [_float(g.get(target, 0)) for g in history]
            hist_avg = _avg(hist_vals)
            last_5 = _avg(hist_vals[-5:]) if hist_vals else 0.0

            line = realistic_line(hist_avg if hist_avg > 0 else pred_stat)
            prob_over = float(1.0 - norm.cdf((line - pred_stat) / sigma))
            prob_over = max(0.0, min(1.0, prob_over))

            rec = recommendation(prob_over)

            props[target] = {
                "line": round(line, 1),
                "average": round(hist_avg, 1),
                "projected": round(pred_stat, 1),
                "over_probability": round(prob_over * 100, 1),
                "under_probability": round((1.0 - prob_over) * 100, 1),
                "last_5_avg": round(last_5, 1),
                "recommendation": rec,
                "source": "ML ENSEMBLE",
            }

        predictions.append(
            {
                "id": player.get("id"),
                "player_name": player.get("player_name"),
                "team": player.get("team"),
                "position": pos,
                "games_played": len(gamelog),
                "props": props,
            }
        )

    with open(PREDICTIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2)

    return predictions


def main():
    players = load_data()
    dataset, feature_names = build_dataset(players)
    fitted, metrics = train_and_evaluate(dataset, feature_names)
    preds = predict_for_players(players, fitted)

    print("=" * 70)
    print("NBA ENSEMBLE TRAINING COMPLETE")
    print("=" * 70)
    for t in TARGETS:
        m = metrics["targets"][t]
        print(
            f"{t:12s} | MAE {m['mae']:.3f} | RMSE {m['rmse']:.3f} | R2 {m['r2']:.3f} | "
            f"DirAcc {m['directional_accuracy']:.3f} | RecHit {m['recommendation_hit_rate']}"
        )
    print(f"\nSaved models: {MODELS_DIR}")
    print(f"Saved metrics: {METRICS_FILE}")
    print(f"Saved predictions: {PREDICTIONS_FILE} ({len(preds)} players)")


if __name__ == "__main__":
    main()
