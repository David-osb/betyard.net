#!/usr/bin/env python3
"""
NFL QB Historical Data Processor
Process 26 years of QB data for XGBoost training
Uses ONLY QB data until other position files are populated
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class QBOnlyProcessor:
    def __init__(self):
        self.le_team = LabelEncoder()
        self.le_opponent = LabelEncoder()
        
    def load_and_clean_data(self, filepath='nfl_data/qb_stats_historical.csv'):
        """Load and clean QB historical data"""
        print(f"Loading QB data from {filepath}...")
        
        try:
            # Read the CSV file
            df = pd.read_csv(filepath)
            print(f"Initial data shape: {df.shape}")
            
            # Display basic info
            print(f"Columns: {list(df.columns)}")
            print(f"Date range: {df['Year'].min()} to {df['Year'].max()}")
            
            # Clean and prepare data
            df_clean = self.clean_data(df)
            print(f"After cleaning: {df_clean.shape}")
            
            return df_clean
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def clean_data(self, df):
        """Clean and prepare the data for training"""
        print("Cleaning data...")
        
        # Basic cleaning
        df = df.dropna(subset=['Tm', 'Year'])
        
        # Filter to QBs only (if Pos column exists)
        if 'Pos' in df.columns:
            initial_count = len(df)
            df = df[df['Pos'].str.contains('QB', na=False)]
            print(f"Filtered to QBs: {initial_count} -> {len(df)} records")
        
        # Convert numeric columns
        numeric_cols = ['Year', 'Age', 'G', 'GS', 'QBrec', 'Cmp', 'Att', 'Cmp%', 'Yds', 'TD', 'Int', 'Lng', 'Y/A', 'AY/A', 'Y/C', 'Y/G', 'Rate', 'QBR', 'Sk', 'Yds.1', 'NY/A', 'ANY/A', 'Sk%', '4QC', 'GWD']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Fill missing values
        df = df.fillna(0)
        
        return df
    
    def engineer_features(self, df):
        """Create features for XGBoost training"""
        print("Engineering features...")
        
        # Create target variable (example: TD/game ratio)
        df['TD_per_game'] = df['TD'] / np.maximum(df['G'], 1)
        
        # Create feature set
        feature_data = []
        
        for _, row in df.iterrows():
            # Core performance features
            features = {
                'attempts_per_game': row['Att'] / max(row['G'], 1),
                'completion_pct': row['Cmp%'] / 100.0 if row['Cmp%'] > 0 else 0,
                'yards_per_attempt': row['Y/A'],
                'yards_per_game': row['Y/G'],
                'td_int_ratio': row['TD'] / max(row['Int'], 1),
                'passer_rating': row['Rate'],
                'sack_rate': row['Sk%'] / 100.0 if row['Sk%'] > 0 else 0,
                'games_started_pct': row['GS'] / max(row['G'], 1),
                'experience': 2024 - row['Year'],  # Years since season
                'age': row['Age'],
                'season_games': row['G'],
                'total_tds': row['TD'],
                'total_ints': row['Int']
            }
            
            # Encode team (use 'Tm' column)
            if 'Tm' in df.columns:
                # Create a simple team encoding
                team_code = hash(str(row['Tm'])) % 32  # 32 NFL teams
                features['team_encoded'] = team_code
            else:
                features['team_encoded'] = 0
                
            feature_data.append(features)
        
        features_df = pd.DataFrame(feature_data)
        print(f"Created {len(features_df)} feature vectors with {len(features_df.columns)} features")
        
        return features_df
    
    def prepare_training_data(self, features_df):
        """Prepare final training data"""
        print("Preparing training data...")
        
        # Remove any remaining NaN values
        features_df = features_df.fillna(0)
        
        # Split features and target
        target_col = 'total_tds'  # Use total TDs as target
        if target_col in features_df.columns:
            X = features_df.drop([target_col], axis=1)
            y = features_df[target_col]
        else:
            # If no clear target, use first column as target
            X = features_df.iloc[:, 1:]
            y = features_df.iloc[:, 0]
        
        print(f"Training data shape: X={X.shape}, y={y.shape}")
        print(f"Features: {list(X.columns)}")
        
        # Basic statistics
        print(f"\nTarget statistics:")
        print(f"Mean: {y.mean():.2f}")
        print(f"Std: {y.std():.2f}")
        print(f"Min: {y.min()}")
        print(f"Max: {y.max()}")
        
        return X, y
    
    def save_processed_data(self, X, y, output_path='processed_qb_data.csv'):
        """Save processed data for training"""
        # Combine features and target
        combined_df = X.copy()
        combined_df['target'] = y
        
        combined_df.to_csv(output_path, index=False)
        print(f"Saved processed data to {output_path}")
        
        return output_path

def main():
    """Main processing function"""
    print("=== NFL QB Historical Data Processor ===")
    print("Processing 26-year QB dataset for XGBoost training...")
    
    processor = QBOnlyProcessor()
    
    # Load and clean data
    df = processor.load_and_clean_data()
    if df is None:
        print("Failed to load data. Exiting.")
        return
    
    # Engineer features
    features_df = processor.engineer_features(df)
    
    # Prepare training data
    X, y = processor.prepare_training_data(features_df)
    
    # Save processed data
    output_file = processor.save_processed_data(X, y)
    
    print(f"\n=== Processing Complete ===")
    print(f"QB dataset ready for XGBoost training")
    print(f"Output: {output_file}")
    print(f"Samples: {len(X)}")
    print(f"Features: {len(X.columns)}")

if __name__ == "__main__":
    main()