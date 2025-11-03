#!/usr/bin/env python3
"""
Modern NFL Data Model (2015-2025)
Train XGBoost model using only recent, relevant NFL data
This should improve prediction accuracy for current players
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score

def train_modern_nfl_model():
    """Train model using only 2015-2025 data for better accuracy"""
    print("Training Modern NFL Model (2015-2025 data only)")
    print("=" * 50)
    
    # Load MODERN data only (2015-2025)
    data_path = r'C:\Users\david\OneDrive\Workspace\Python Projects\UI.UX\ml-backend\nfl_data'
    df = pd.read_csv(f'{data_path}/passing_stats_historical_2015_2025.csv')
    
    print(f"Loaded modern data: {len(df)} records from {df['Year'].min():.0f}-{df['Year'].max():.0f}")
    
    # Clean and prepare data
    df = df.dropna(subset=['Player', 'Age', 'G', 'Yds', 'TD'])
    df = df[(df['Age'] >= 20) & (df['Age'] <= 45)]
    df = df[(df['G'] > 0) & (df['G'] <= 17)]
    
    # Create targets
    df['yards_per_game'] = df['Yds'] / df['G']
    df['td_per_game'] = df['TD'] / df['G']
    df['season'] = df['Year']
    
    # Create lag features (prevent data leakage)
    df = df.sort_values(['Player', 'season']).reset_index(drop=True)
    df_with_lags = []
    
    for player in df['Player'].unique():
        player_data = df[df['Player'] == player].copy().sort_values('season')
        player_data['prev_yards_per_game'] = player_data['yards_per_game'].shift(1)
        player_data['prev_td_per_game'] = player_data['td_per_game'].shift(1)
        player_data['experience'] = player_data.groupby('Player').cumcount()
        df_with_lags.append(player_data)
    
    df_final = pd.concat(df_with_lags, ignore_index=True)
    df_final = df_final.groupby('Player', group_keys=False).apply(lambda x: x.iloc[1:])
    
    print(f"Prepared data: {len(df_final)} records")
    
    # Temporal split for modern era (2015-2019 train, 2020+ test)
    train_data = df_final[df_final['season'] <= 2019].copy()
    test_data = df_final[df_final['season'] >= 2020].copy()
    
    print(f"Training: {len(train_data)} records (2015-2019)")
    print(f"Testing: {len(test_data)} records (2020+)")
    
    # Train yards model with modern data
    features = ['Age', 'experience', 'prev_yards_per_game']
    X_train = train_data[features].fillna(0)
    y_train = train_data['yards_per_game']
    X_test = test_data[features].fillna(0)
    y_test = test_data['yards_per_game']
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train XGBoost with optimized parameters for modern data
    model = xgb.XGBRegressor(
        max_depth=4,
        learning_rate=0.08,  # Slightly higher for smaller dataset
        n_estimators=150,    # Fewer estimators for modern focused data
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
    
    print(f"\nMODERN MODEL PERFORMANCE:")
    print(f"  MAE: {mae:.2f}")
    print(f"  R-squared: {r2:.3f}")
    print(f"  Betting Accuracy: {betting_acc:.1%}")
    
    # Save modern model
    model_data = {
        'yards_model': model,
        'yards_scaler': scaler,
        'features': features,
        'model_type': 'Modern NFL XGBoost (2015-2025)',
        'data_period': '2015-2025',
        'performance': {
            'mae': mae,
            'r2': r2,
            'betting_accuracy': betting_acc
        },
        'training_records': len(train_data),
        'test_records': len(test_data)
    }
    
    model_path = r'C:\Users\david\OneDrive\Workspace\Python Projects\UI.UX\ml-backend\modern_nfl_models.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\nModern model saved to: modern_nfl_models.pkl")
    print("\nBENEFITS of modern data model:")
    print("1. More relevant to current NFL playing styles")
    print("2. Reflects modern rule changes and strategies")
    print("3. Better predictions for active players")
    print("4. Reduced noise from outdated historical patterns")
    
    return model_data

if __name__ == "__main__":
    model_data = train_modern_nfl_model()
    
    # Test predictions
    print("\nTesting predictions on modern model:")
    model = model_data['yards_model']
    scaler = model_data['yards_scaler']
    
    # Test typical QB values
    test_features = np.array([[28, 5, 250]])  # Age 28, 5 years exp, 250 yards/game prev
    test_scaled = scaler.transform(test_features)
    prediction = model.predict(test_scaled)[0]
    
    print(f"Sample prediction: {prediction:.1f} yards/game")
    print("\nReady to update app.py to use modern_nfl_models.pkl!")