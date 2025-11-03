#!/usr/bin/env python3
"""
QB-Only XGBoost Training Pipeline
Train with complete 26-year QB dataset while RB data is resolved
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class QBTrainingPipeline:
    def __init__(self):
        self.le_team = LabelEncoder()
        self.model = None
        self.feature_names = None
        
    def load_qb_data(self, filepath='nfl_data/qb_stats_historical.csv'):
        """Load the 26-year QB historical dataset"""
        print(f"Loading QB data from {filepath}...")
        
        try:
            # Read the full dataset
            df = pd.read_csv(filepath)
            print(f"Raw data shape: {df.shape}")
            
            # Skip header rows and get actual data
            # The data starts after the multi-level headers
            data_start_row = 1  # Skip the header row
            df_clean = df.iloc[data_start_row:].copy()
            
            # Set proper column names from the second row of original headers
            # Use the second header row which has the actual column names
            if len(df.columns) >= 43:
                column_names = ['Rank', 'Player', 'Age', 'Team', 'Pos', 'G', 'GS', 'Cmp', 'Att',
                              'IAY', 'IAY_PA', 'CAY', 'CAY_Cmp', 'CAY_PA', 'YAC', 'YAC_Cmp', 
                              'Bats', 'ThAwy', 'Spikes', 'Drops', 'Drop_Pct', 'BadTh', 'Bad_Pct',
                              'OnTgt', 'OnTgt_Pct', 'PktTime', 'Bltz', 'Hrry', 'Hits', 'Prss', 
                              'Prss_Pct', 'Scrm', 'Yds_Scr', 'RPO_Plays', 'RPO_Yds', 'RPO_PassAtt',
                              'RPO_PassYds', 'RPO_RushAtt', 'RPO_RushYds', 'PA_PassAtt', 'PA_PassYds',
                              'Awards', 'PlayerID']
                              
                df_clean.columns = column_names[:len(df_clean.columns)]
            
            # Convert numeric columns
            numeric_cols = ['Age', 'G', 'GS', 'Cmp', 'Att', 'IAY', 'IAY_PA', 'CAY', 'CAY_Cmp', 
                           'CAY_PA', 'YAC', 'YAC_Cmp', 'Bats', 'ThAwy', 'Spikes', 'Drops', 
                           'Drop_Pct', 'BadTh', 'Bad_Pct', 'OnTgt', 'OnTgt_Pct', 'PktTime',
                           'Bltz', 'Hrry', 'Hits', 'Prss', 'Prss_Pct', 'Scrm', 'Yds_Scr']
                           
            for col in numeric_cols:
                if col in df_clean.columns:
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            
            # Remove rows with missing essential data
            df_clean = df_clean.dropna(subset=['Player', 'Team', 'Cmp', 'Att'])
            
            # Filter to actual QB records (remove headers and invalid data)
            df_clean = df_clean[df_clean['Att'].notna() & (df_clean['Att'] > 0)]
            
            print(f"Cleaned data shape: {df_clean.shape}")
            print(f"QB players: {df_clean['Player'].nunique()}")
            
            return df_clean
            
        except Exception as e:
            print(f"Error loading QB data: {e}")
            return None
    
    def create_features(self, df):
        """Create training features from QB data"""
        print("Creating features...")
        
        features_list = []
        
        for _, row in df.iterrows():
            try:
                # Core QB metrics
                attempts = float(row.get('Att', 0))
                completions = float(row.get('Cmp', 0))
                games = float(row.get('G', 1))
                
                if attempts == 0 or games == 0:
                    continue
                
                features = {
                    # Passing efficiency
                    'completion_rate': completions / attempts if attempts > 0 else 0,
                    'attempts_per_game': attempts / games,
                    'air_yards_per_attempt': float(row.get('IAY_PA', 0)),
                    'yac_per_completion': float(row.get('YAC_Cmp', 0)),
                    'accuracy_rate': float(row.get('OnTgt_Pct', 0)) / 100.0,
                    
                    # Pressure handling
                    'pressure_rate': float(row.get('Prss_Pct', 0)) / 100.0,
                    'pocket_time': float(row.get('PktTime', 0)),
                    'scramble_efficiency': float(row.get('Yds_Scr', 0)),
                    
                    # Volume and experience
                    'games_played': games,
                    'games_started': float(row.get('GS', 0)),
                    'age': float(row.get('Age', 25)),
                    
                    # Team encoding
                    'team_encoded': hash(str(row.get('Team', ''))) % 32,
                    
                    # Target variable (completions per game)
                    'target': completions / games
                }
                
                features_list.append(features)
                
            except Exception as e:
                continue
        
        features_df = pd.DataFrame(features_list)
        features_df = features_df.fillna(0)
        
        print(f"Created {len(features_df)} feature samples")
        print(f"Features: {list(features_df.columns)}")
        
        return features_df
    
    def train_model(self, features_df):
        """Train XGBoost model on QB data"""
        print("Training XGBoost model...")
        
        # Separate features and target
        X = features_df.drop(['target'], axis=1)
        y = features_df['target']
        
        self.feature_names = list(X.columns)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train XGBoost model
        self.model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\nModel Performance:")
        print(f"MSE: {mse:.4f}")
        print(f"R²: {r2:.4f}")
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        
        # Feature importance
        importance = self.model.feature_importances_
        feature_importance = list(zip(self.feature_names, importance))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\nTop Feature Importances:")
        for feature, importance in feature_importance[:8]:
            print(f"{feature}: {importance:.4f}")
        
        return self.model
    
    def save_model(self, filepath='qb_xgboost_model.json'):
        """Save the trained model"""
        if self.model:
            self.model.save_model(filepath)
            print(f"Model saved to {filepath}")
        return filepath

def main():
    """Main training pipeline"""
    print("=== QB-Only XGBoost Training Pipeline ===")
    print("Training with 26-year QB historical dataset")
    
    pipeline = QBTrainingPipeline()
    
    # Load QB data
    df = pipeline.load_qb_data()
    if df is None:
        print("Failed to load QB data. Exiting.")
        return
    
    # Create features
    features_df = pipeline.create_features(df)
    if features_df.empty:
        print("Failed to create features. Exiting.")
        return
    
    # Train model
    model = pipeline.train_model(features_df)
    
    # Save model
    model_path = pipeline.save_model()
    
    print(f"\n=== Training Complete ===")
    print(f"✅ QB model trained successfully")
    print(f"📁 Model saved: {model_path}")
    print(f"📊 Ready for predictions")
    print(f"\n💡 Next: Add RB/TE/WR data when available for multi-position training")

if __name__ == "__main__":
    main()