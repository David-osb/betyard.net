#!/usr/bin/env python3
"""
Improved Real Data Loader for XGBoost Models
============================================

This replaces synthetic data generation with real NFL historical data
to improve prediction accuracy on your website.
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
import logging

logger = logging.getLogger(__name__)

class ImprovedNFLDataLoader:
    """Load and prepare real NFL historical data for improved XGBoost models"""
    
    def __init__(self):
        self.data_files = {
            'QB': r'C:\Users\david\OneDrive\Workspace\Python Projects\UI.UX\ml-backend\nfl_data\passing_stats_historical_clean.csv',
            'RB': r'C:\Users\david\OneDrive\Workspace\Python Projects\UI.UX\ml-backend\nfl_data\rushing_stats_historical_clean.csv', 
            'WR': r'C:\Users\david\OneDrive\Workspace\Python Projects\UI.UX\ml-backend\nfl_data\receiving_stats_historical_clean.csv',
            'TE': r'C:\Users\david\OneDrive\Workspace\Python Projects\UI.UX\ml-backend\nfl_data\receiving_stats_historical_clean.csv'
        }
        self.models = {}
        self.scalers = {}
        self.feature_columns = {}
    
    def load_position_data(self, position: str) -> pd.DataFrame:
        """Load real data for specific position"""
        print(f"Loading real {position} data...")
        
        filepath = self.data_files[position]
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        df = pd.read_csv(filepath)
        
        # Basic cleaning
        df = df.dropna(subset=['Player', 'Age', 'G', 'Yds'])
        df = df[(df['Age'] >= 18) & (df['Age'] <= 45)]
        df = df[(df['G'] > 0) & (df['G'] <= 20)]
        df = df[df['Yds'] >= 0]
        
        # Create per-game statistics
        df['yards_per_game'] = df['Yds'] / df['G']
        df['td_per_game'] = df['TD'] / df['G']
        
        if position == 'QB':
            # QB-specific features
            if 'Cmp' in df.columns:
                df['completions_per_game'] = df['Cmp'] / df['G']
            if 'Att' in df.columns:
                df['attempts_per_game'] = df['Att'] / df['G']
            if 'Rate' in df.columns:
                df['passer_rating'] = df['Rate']
        
        elif position == 'RB':
            # RB-specific features  
            if 'Att' in df.columns:
                df['carries_per_game'] = df['Att'] / df['G']
            if 'Rec' in df.columns:
                df['receptions_per_game'] = df['Rec'] / df['G']
        
        elif position in ['WR', 'TE']:
            # Receiving features
            if 'Rec' in df.columns:
                df['receptions_per_game'] = df['Rec'] / df['G']
            if 'Tgt' in df.columns:
                df['targets_per_game'] = df['Tgt'] / df['G']
        
        # Create experience and form features
        df['experience'] = df['Age'] - 22  # Approximate years since college
        df['veteran'] = (df['Age'] >= 30).astype(int)
        df['prime_age'] = ((df['Age'] >= 25) & (df['Age'] <= 30)).astype(int)
        
        print(f"Loaded {len(df)} {position} records")
        return df
    
    def create_features(self, df: pd.DataFrame, position: str) -> tuple:
        """Create feature matrix and target from real data"""
        
        # Base features available for all positions
        base_features = ['Age', 'G', 'experience', 'veteran', 'prime_age']
        
        # Position-specific features
        if position == 'QB':
            specific_features = []
            if 'completions_per_game' in df.columns:
                specific_features.append('completions_per_game')
            if 'attempts_per_game' in df.columns:
                specific_features.append('attempts_per_game')
            if 'passer_rating' in df.columns:
                specific_features.append('passer_rating')
                
        elif position == 'RB':
            specific_features = []
            if 'carries_per_game' in df.columns:
                specific_features.append('carries_per_game')
            if 'receptions_per_game' in df.columns:
                specific_features.append('receptions_per_game')
                
        elif position in ['WR', 'TE']:
            specific_features = []
            if 'receptions_per_game' in df.columns:
                specific_features.append('receptions_per_game')
            if 'targets_per_game' in df.columns:
                specific_features.append('targets_per_game')
        
        # Combine features
        feature_cols = base_features + specific_features
        
        # Filter to existing columns
        feature_cols = [col for col in feature_cols if col in df.columns]
        
        if not feature_cols:
            raise ValueError(f"No valid features found for {position}")
        
        X = df[feature_cols].copy()
        y = df['yards_per_game'].copy()
        
        # Remove any remaining NaN values
        mask = X.notna().all(axis=1) & y.notna() & (y > 0)
        X_clean = X[mask]
        y_clean = y[mask]
        
        self.feature_columns[position] = feature_cols
        
        print(f"Created {position} features: {feature_cols}")
        print(f"Training samples: {len(X_clean)}")
        
        return X_clean, y_clean
    
    def train_improved_model(self, position: str):
        """Train improved XGBoost model with real data"""
        print(f"\nTraining improved {position} model...")
        
        # Load real data
        df = self.load_position_data(position)
        X, y = self.create_features(df, position)
        
        if len(X) < 100:
            print(f"Warning: Only {len(X)} samples for {position}")
        
        # Split data temporally if Year column exists
        if 'Year' in df.columns:
            # Use earlier years for training, recent for testing
            train_years = df['Year'] <= 2020
            test_years = df['Year'] > 2020
            
            train_mask = train_years & (X.index.isin(df.index))
            test_mask = test_years & (X.index.isin(df.index))
            
            if train_mask.sum() > 50 and test_mask.sum() > 10:
                X_train, X_test = X[train_mask], X[test_mask]
                y_train, y_test = y[train_mask], y[test_mask]
            else:
                # Fallback to random split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
        else:
            # Random split if no year data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train improved XGBoost model
        model = xgb.XGBRegressor(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Validate
        y_pred = model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"{position} Model Performance:")
        print(f"  MAE: {mae:.2f} yards/game")
        print(f"  R²: {r2:.3f}")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Test samples: {len(X_test)}")
        
        # Store model and scaler
        self.models[position] = model
        self.scalers[position] = scaler
        
        return model, scaler
    
    def train_all_positions(self):
        """Train models for all positions"""
        print("Training improved XGBoost models with REAL NFL data...")
        print("=" * 50)
        
        for position in ['QB', 'RB', 'WR', 'TE']:
            try:
                self.train_improved_model(position)
            except Exception as e:
                print(f"Error training {position} model: {e}")
        
        print("\n" + "=" * 50)
        print("Saving improved models...")
        self.save_models()
    
    def save_models(self):
        """Save improved models for use in app.py"""
        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'feature_columns': self.feature_columns,
            'model_type': 'Improved XGBoost with Real NFL Data',
            'training_date': pd.Timestamp.now().isoformat()
        }
        
        # Save to file that app.py can load
        with open('improved_nfl_models.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        
        print("✅ Improved models saved to 'improved_nfl_models.pkl'")
        print(f"Models trained: {list(self.models.keys())}")
    
    def predict_yards(self, position: str, player_features: dict) -> float:
        """Make prediction with improved model"""
        if position not in self.models:
            raise ValueError(f"No model available for {position}")
        
        model = self.models[position]
        scaler = self.scalers[position]
        feature_cols = self.feature_columns[position]
        
        # Create feature vector
        feature_values = []
        for col in feature_cols:
            if col in player_features:
                feature_values.append(player_features[col])
            else:
                # Use reasonable defaults
                defaults = {
                    'Age': 26, 'G': 16, 'experience': 4, 
                    'veteran': 0, 'prime_age': 1,
                    'completions_per_game': 20, 'attempts_per_game': 32,
                    'passer_rating': 90, 'carries_per_game': 15,
                    'receptions_per_game': 4, 'targets_per_game': 6
                }
                feature_values.append(defaults.get(col, 0))
        
        # Scale and predict
        X = np.array(feature_values).reshape(1, -1)
        X_scaled = scaler.transform(X)
        prediction = model.predict(X_scaled)[0]
        
        return max(0, prediction)  # Ensure non-negative

def main():
    """Train improved models"""
    loader = ImprovedNFLDataLoader()
    loader.train_all_positions()

if __name__ == "__main__":
    main()