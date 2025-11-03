#!/usr/bin/env python3
"""
Modern Multi-Position XGBoost Model (2015-2025)
Train XGBoost models for QB, RB, WR, TE using only recent, relevant NFL data
This replaces all synthetic models with real modern data
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score

def load_modern_position_data():
    """Load all position data from 2015-2025"""
    data_path = r'C:\Users\david\OneDrive\Workspace\Python Projects\UI.UX\ml-backend\nfl_data'
    
    datasets = {
        'QB': pd.read_csv(f'{data_path}/passing_stats_historical_2015_2025.csv'),
        'RB': pd.read_csv(f'{data_path}/rushing_stats_historical_2015_2025.csv'),
        'WR': pd.read_csv(f'{data_path}/receiving_stats_historical_2015_2025.csv'),
        'TE': pd.read_csv(f'{data_path}/receiving_stats_historical_2015_2025.csv')  # TEs also in receiving
    }
    
    return datasets

def prepare_position_data(df, position):
    """Prepare data for specific position"""
    # Clean data
    df = df.dropna(subset=['Player', 'Age', 'G', 'Yds', 'TD'])
    df = df[(df['Age'] >= 18) & (df['Age'] <= 45)]
    df = df[(df['G'] > 0) & (df['G'] <= 17)]
    
    # Create per-game stats
    df['yards_per_game'] = df['Yds'] / df['G']
    df['td_per_game'] = df['TD'] / df['G']
    df['season'] = df['Year']
    
    # Position-specific stats
    if position == 'QB' and 'Cmp' in df.columns:
        df['completions_per_game'] = df['Cmp'] / df['G']
        df['attempts_per_game'] = df.get('Att', df['Cmp'] * 1.5) / df['G']
    elif position in ['WR', 'TE'] and 'Rec' in df.columns:
        df['receptions_per_game'] = df['Rec'] / df['G']
        df['targets_per_game'] = df.get('Tgt', df['Rec'] * 1.3) / df['G']
    elif position == 'RB':
        df['attempts_per_game'] = df.get('Att', df['Yds'] / 4.2) / df['G']  # Estimate attempts
    
    # Create lag features (prevent data leakage)
    df = df.sort_values(['Player', 'season']).reset_index(drop=True)
    df_with_lags = []
    
    for player in df['Player'].unique():
        player_data = df[df['Player'] == player].copy().sort_values('season')
        
        # Previous season performance
        player_data['prev_yards_per_game'] = player_data['yards_per_game'].shift(1)
        player_data['prev_td_per_game'] = player_data['td_per_game'].shift(1)
        
        # Position-specific lag features
        if position == 'QB' and 'completions_per_game' in player_data.columns:
            player_data['prev_completions_per_game'] = player_data['completions_per_game'].shift(1)
        elif position in ['WR', 'TE'] and 'receptions_per_game' in player_data.columns:
            player_data['prev_receptions_per_game'] = player_data['receptions_per_game'].shift(1)
        elif position == 'RB' and 'attempts_per_game' in player_data.columns:
            player_data['prev_attempts_per_game'] = player_data['attempts_per_game'].shift(1)
        
        # Career stats
        player_data['career_games'] = player_data['G'].cumsum().shift(1)
        player_data['experience'] = player_data.groupby('Player').cumcount()
        
        # Age categories
        player_data['age_prime'] = ((player_data['Age'] >= 25) & (player_data['Age'] <= 30)).astype(int)
        player_data['veteran'] = (player_data['Age'] >= 32).astype(int)
        
        df_with_lags.append(player_data)
    
    df_final = pd.concat(df_with_lags, ignore_index=True)
    
    # Remove first season for each player (no lag data)
    df_final = df_final.groupby('Player', group_keys=False).apply(lambda x: x.iloc[1:])
    
    return df_final

def train_position_model(df, position):
    """Train XGBoost model for specific position"""
    print(f"\nTraining {position} model...")
    print(f"  Data: {len(df)} records from {df['season'].min():.0f}-{df['season'].max():.0f}")
    
    # Temporal split (2015-2019 train, 2020+ test)
    train_data = df[df['season'] <= 2019].copy()
    test_data = df[df['season'] >= 2020].copy()
    
    print(f"  Training: {len(train_data)} records (2015-2019)")
    print(f"  Testing: {len(test_data)} records (2020+)")
    
    if len(train_data) < 50:
        print(f"  ⚠️ Insufficient training data for {position}")
        return None
    
    # Define features based on position
    base_features = ['Age', 'experience', 'career_games', 'age_prime', 'veteran', 
                    'prev_yards_per_game', 'prev_td_per_game']
    
    if position == 'QB':
        extra_features = ['prev_completions_per_game'] if 'prev_completions_per_game' in df.columns else []
    elif position in ['WR', 'TE']:
        extra_features = ['prev_receptions_per_game'] if 'prev_receptions_per_game' in df.columns else []
    elif position == 'RB':
        extra_features = ['prev_attempts_per_game'] if 'prev_attempts_per_game' in df.columns else []
    else:
        extra_features = []
    
    features = [f for f in base_features + extra_features if f in df.columns]
    
    # Prepare training data
    X_train = train_data[features].fillna(0)
    y_train = train_data['yards_per_game']
    X_test = test_data[features].fillna(0) 
    y_test = test_data['yards_per_game']
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train XGBoost with position-optimized parameters
    if position == 'QB':
        params = {'max_depth': 4, 'learning_rate': 0.08, 'n_estimators': 150}
    elif position == 'RB':
        params = {'max_depth': 3, 'learning_rate': 0.1, 'n_estimators': 120}
    elif position in ['WR', 'TE']:
        params = {'max_depth': 3, 'learning_rate': 0.09, 'n_estimators': 140}
    else:
        params = {'max_depth': 4, 'learning_rate': 0.08, 'n_estimators': 150}
    
    model = xgb.XGBRegressor(
        **params,
        reg_alpha=0.1,
        reg_lambda=0.1,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Validate
    y_pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Betting accuracy
    line = np.median(y_test)
    betting_acc = accuracy_score(y_test > line, y_pred > line)
    
    print(f"  Performance: MAE={mae:.2f}, R²={r2:.3f}, Betting Acc={betting_acc:.1%}")
    
    return {
        'model': model,
        'scaler': scaler,
        'features': features,
        'performance': {
            'mae': mae,
            'r2': r2,
            'betting_accuracy': betting_acc,
            'training_records': len(train_data),
            'test_records': len(test_data)
        }
    }

def train_all_modern_models():
    """Train modern XGBoost models for all positions"""
    print("Training Modern Multi-Position XGBoost Models (2015-2025)")
    print("=" * 60)
    
    # Load data
    datasets = load_modern_position_data()
    models = {}
    
    for position, raw_df in datasets.items():
        if position == 'TE':
            # Filter TE from receiving data (lower targets/receptions)
            if 'Tgt' in raw_df.columns:
                te_threshold = raw_df['Tgt'].quantile(0.4)  # Bottom 40% are likely TEs
                raw_df = raw_df[raw_df['Tgt'] <= te_threshold]
            elif 'Rec' in raw_df.columns:
                rec_threshold = raw_df['Rec'].quantile(0.35)  # Bottom 35% are likely TEs  
                raw_df = raw_df[raw_df['Rec'] <= rec_threshold]
        
        # Prepare data
        prepared_df = prepare_position_data(raw_df, position)
        
        if len(prepared_df) > 0:
            # Train model
            model_data = train_position_model(prepared_df, position)
            if model_data:
                models[position] = model_data
        else:
            print(f"⚠️ No data available for {position}")
    
    # Save all models
    modern_models = {
        'models': {pos: data['model'] for pos, data in models.items()},
        'scalers': {pos: data['scaler'] for pos, data in models.items()},
        'features': {pos: data['features'] for pos, data in models.items()},
        'performance': {pos: data['performance'] for pos, data in models.items()},
        'model_type': 'Modern Multi-Position XGBoost (2015-2025)',
        'data_period': '2015-2025',
        'positions_trained': list(models.keys())
    }
    
    model_path = r'C:\Users\david\OneDrive\Workspace\Python Projects\UI.UX\ml-backend\modern_multiposition_models.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(modern_models, f)
    
    print(f"\n✅ Modern multi-position models saved!")
    print(f"📍 Location: modern_multiposition_models.pkl")
    print(f"🏈 Positions trained: {', '.join(models.keys())}")
    
    # Performance summary
    print(f"\n📊 PERFORMANCE SUMMARY:")
    for position, data in models.items():
        perf = data['performance']
        print(f"  {position}: R²={perf['r2']:.3f}, Betting Acc={perf['betting_accuracy']:.1%}, Records={perf['training_records']+perf['test_records']}")
    
    return modern_models

if __name__ == "__main__":
    models = train_all_modern_models()
    
    print(f"\n🎯 BENEFITS of Modern Multi-Position Models:")
    print("1. All positions use 2015-2025 modern NFL data")
    print("2. Position-specific features and parameters")
    print("3. Realistic betting accuracy for each position")
    print("4. Consistent temporal validation across positions")
    print("5. Ready to replace ALL synthetic models in app.py!")