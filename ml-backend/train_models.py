"""
Train XGBoost models with 10 features using ESPN API data
This script generates synthetic training data based on NFL statistics patterns
"""

import xgboost as xgb
import numpy as np
import pandas as pd
from datetime import datetime
import json

# Team stats (same as in app.py)
TEAM_STATS = {
    # AFC East
    'BUF': {'offense': 92, 'defense': 88}, 'MIA': {'offense': 85, 'defense': 78},
    'NE': {'offense': 72, 'defense': 75}, 'NYJ': {'offense': 70, 'defense': 82},
    
    # AFC North
    'BAL': {'offense': 88, 'defense': 85}, 'CIN': {'offense': 90, 'defense': 75},
    'CLE': {'offense': 75, 'defense': 88}, 'PIT': {'offense': 78, 'defense': 90},
    
    # AFC South
    'HOU': {'offense': 85, 'defense': 72}, 'IND': {'offense': 78, 'defense': 75},
    'JAX': {'offense': 75, 'defense': 70}, 'TEN': {'offense': 68, 'defense': 72},
    
    # AFC West
    'DEN': {'offense': 75, 'defense': 75}, 'KC': {'offense': 95, 'defense': 85},
    'LV': {'offense': 70, 'defense': 72}, 'LAC': {'offense': 82, 'defense': 75},
    
    # NFC East
    'DAL': {'offense': 85, 'defense': 78}, 'NYG': {'offense': 68, 'defense': 72},
    'PHI': {'offense': 88, 'defense': 82}, 'WAS': {'offense': 90, 'defense': 75},
    
    # NFC North
    'CHI': {'offense': 75, 'defense': 72}, 'DET': {'offense': 92, 'defense': 75},
    'GB': {'offense': 82, 'defense': 78}, 'MIN': {'offense': 85, 'defense': 80},
    
    # NFC South
    'ATL': {'offense': 85, 'defense': 72}, 'CAR': {'offense': 68, 'defense': 70},
    'NO': {'offense': 78, 'defense': 75}, 'TB': {'offense': 80, 'defense': 75},
    
    # NFC West
    'ARI': {'offense': 75, 'defense': 72}, 'LAR': {'offense': 78, 'defense': 75},
    'SF': {'offense': 88, 'defense': 88}, 'SEA': {'offense': 82, 'defense': 78}
}

def generate_training_data(position, n_samples=5000):
    """
    Generate synthetic training data for a position
    
    10 Features:
    1. Team offensive rating (0-100)
    2. Team defensive rating (0-100)
    3. Opponent defensive rating (0-100)
    4. Is home game (0 or 1)
    5. Player season avg yards
    6. Player season avg TDs
    7. Player recent 3-game avg yards
    8. Weather score (0-100, higher is better)
    9. Matchup difficulty (0-100)
    10. Player health score (0-100)
    """
    
    print(f"\n{'='*60}")
    print(f"Generating {n_samples} samples for {position.upper()}")
    print(f"{'='*60}")
    
    # Position-specific baselines
    if position == 'qb':
        base_yards = 250
        base_tds = 2.0
        yards_std = 50
        td_std = 0.8
    elif position == 'rb':
        base_yards = 75
        base_tds = 0.5
        yards_std = 30
        td_std = 0.4
    elif position == 'wr':
        base_yards = 60
        base_tds = 0.5
        yards_std = 25
        td_std = 0.4
    else:  # te
        base_yards = 50
        base_tds = 0.4
        yards_std = 20
        td_std = 0.3
    
    features = []
    targets = []
    
    teams = list(TEAM_STATS.keys())
    
    for i in range(n_samples):
        # Random team matchup
        team = np.random.choice(teams)
        opponent = np.random.choice([t for t in teams if t != team])
        
        team_off = TEAM_STATS[team]['offense']
        team_def = TEAM_STATS[team]['defense']
        opp_def = TEAM_STATS[opponent]['defense']
        
        # Feature 1-3: Team stats
        f1_team_off = team_off
        f2_team_def = team_def
        f3_opp_def = opp_def
        
        # Feature 4: Is home (50% chance)
        f4_is_home = np.random.choice([0, 1])
        home_bonus = 5 if f4_is_home else 0
        
        # Feature 5-7: Player history (with noise)
        player_talent = np.random.normal(1.0, 0.2)  # 0.6 to 1.4 multiplier
        f5_season_avg = base_yards * player_talent
        f6_season_avg_tds = base_tds * player_talent
        
        # Recent form (can be hot or cold)
        recent_form = np.random.normal(1.0, 0.15)
        f7_recent_avg = f5_season_avg * recent_form
        
        # Feature 8: Weather (70-100 for good, 40-70 for moderate, 0-40 for bad)
        f8_weather = np.random.beta(8, 2) * 100  # Skewed towards good weather
        weather_impact = (f8_weather - 50) / 50  # -1 to +1
        
        # Feature 9: Matchup difficulty
        f9_matchup = max(0, min(100, opp_def - team_off + 50))
        matchup_impact = (50 - f9_matchup) / 50  # -1 to +1
        
        # Feature 10: Player health (mostly healthy)
        f10_health = np.random.beta(9, 1) * 100  # Skewed towards 100
        health_impact = (f10_health - 50) / 50  # -1 to +1
        
        # Calculate target (actual performance)
        # Base performance from player talent
        performance = base_yards * player_talent
        
        # Adjust for team offense quality
        performance *= (team_off / 80)  # 80 is average
        
        # Adjust for opponent defense
        performance *= (100 - opp_def) / 50  # Harder defense = less yards
        
        # Recent form matters
        performance *= recent_form
        
        # Home field advantage
        performance *= (1 + home_bonus / 100)
        
        # Weather impact (mostly affects passing)
        if position == 'qb':
            performance *= (1 + weather_impact * 0.15)
        
        # Matchup difficulty
        performance *= (1 + matchup_impact * 0.1)
        
        # Health impact
        performance *= (1 + health_impact * 0.1)
        
        # Add realistic noise
        performance += np.random.normal(0, yards_std * 0.3)
        
        # Ensure positive
        performance = max(10, performance)
        
        features.append([
            f1_team_off,
            f2_team_def,
            f3_opp_def,
            f4_is_home,
            f5_season_avg,
            f6_season_avg_tds,
            f7_recent_avg,
            f8_weather,
            f9_matchup,
            f10_health
        ])
        
        targets.append(performance)
        
        # Progress indicator
        if (i + 1) % 1000 == 0:
            print(f"Generated {i + 1}/{n_samples} samples...")
    
    X = np.array(features)
    y = np.array(targets)
    
    print(f"\n✅ Generated {n_samples} samples")
    print(f"Feature shape: {X.shape}")
    print(f"Target range: {y.min():.1f} to {y.max():.1f}")
    print(f"Target mean: {y.mean():.1f} ± {y.std():.1f}")
    
    return X, y

def train_xgboost_model(X, y, position):
    """Train XGBoost model with optimal parameters"""
    
    print(f"\n{'='*60}")
    print(f"Training XGBoost model for {position.upper()}")
    print(f"{'='*60}")
    
    # Split train/validation
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    # Create DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    # XGBoost parameters
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'eval_metric': 'rmse',
        'seed': 42
    }
    
    # Train
    evals = [(dtrain, 'train'), (dval, 'val')]
    print("\nTraining progress:")
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=200,
        evals=evals,
        early_stopping_rounds=20,
        verbose_eval=25
    )
    
    # Evaluate
    y_pred = model.predict(dval)
    mae = np.mean(np.abs(y_pred - y_val))
    rmse = np.sqrt(np.mean((y_pred - y_val) ** 2))
    
    print(f"\n✅ Model trained successfully!")
    print(f"Validation MAE: {mae:.2f}")
    print(f"Validation RMSE: {rmse:.2f}")
    print(f"Feature importance:")
    
    feature_names = [
        'team_offense', 'team_defense', 'opp_defense',
        'is_home', 'season_avg_yards', 'season_avg_tds',
        'recent_avg', 'weather', 'matchup_diff', 'health'
    ]
    
    importance = model.get_score(importance_type='gain')
    for fname in feature_names:
        score = importance.get(f'f{feature_names.index(fname)}', 0)
        print(f"  {fname:20s}: {score:>8.1f}")
    
    return model

def main():
    """Train all position models"""
    
    print("\n" + "="*60)
    print("🏈 NFL PLAYER PERFORMANCE MODEL TRAINER")
    print("Training XGBoost models with 10 features")
    print("="*60)
    
    positions = ['qb', 'rb', 'wr', 'te']
    
    for position in positions:
        # Generate training data
        X, y = generate_training_data(position, n_samples=5000)
        
        # Train model
        model = train_xgboost_model(X, y, position)
        
        # Save model in native XGBoost format
        model_path = f'{position}_model_v4.json'
        model.save_model(model_path)
        print(f"\n✅ Saved model to {model_path}")
    
    print("\n" + "="*60)
    print("🎉 ALL MODELS TRAINED SUCCESSFULLY!")
    print("="*60)
    print("\nTrained models:")
    for pos in positions:
        print(f"  ✅ {pos}_model_v4.json")
    
    print("\n📊 Model Statistics:")
    print("  - 10 features per prediction")
    print("  - 5,000 training samples per position")
    print("  - XGBoost regression with early stopping")
    print("  - Based on NFL team stats and player patterns")
    
    print("\n🚀 Next steps:")
    print("  1. Models are ready for deployment (v4 .json format)")
    print("  2. Upload .json files to Render ml-backend directory")
    print("  3. Backend will use 10-feature predictions")
    print("  4. JSON format ensures Render can't cache binary data!")

if __name__ == '__main__':
    main()
