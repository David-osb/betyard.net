#!/usr/bin/env python3
"""
Robust RB Historical Data Processor
Handles the real 13k-line Pro Football Reference RB dataset
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import logging
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RobustRBProcessor:
    def __init__(self):
        self.le_team = LabelEncoder()
        self.le_opponent = LabelEncoder()
        
    def load_and_clean_data(self, filepath='nfl_data/rb_stats_historical.csv'):
        """Load and clean the massive RB historical dataset"""
        logger.info(f"Loading 26 years of NFL RB historical data from {filepath}...")
        
        try:
            # First, try to read with error handling for inconsistent columns
            df = pd.read_csv(filepath, on_bad_lines='skip', low_memory=False)
            logger.info(f"Successfully loaded data: {df.shape}")
            
            # Skip header rows (Pro Football Reference format)
            # Find where actual data starts (after header rows)
            data_start_idx = 1  # Skip first header row
            df_data = df.iloc[data_start_idx:].copy()
            
            # Set proper column names based on the expected Pro Football Reference format
            expected_columns = [
                'Rank', 'Player', 'Age', 'Team', 'Pos', 'G', 'GS', 'Att', 'Yds', 
                'FirstDown', 'YBC', 'YBC_Att', 'YAC', 'YAC_Att', 'BrkTkl', 'Att_Br', 
                'Awards', 'PlayerID'
            ]
            
            # Adjust column names to match available columns
            if len(df_data.columns) >= len(expected_columns):
                df_data.columns = expected_columns[:len(df_data.columns)]
            
            logger.info(f"Data columns: {list(df_data.columns)}")
            
            # Clean the data
            df_clean = self.clean_rb_data(df_data)
            logger.info(f"After cleaning: {df_clean.shape}")
            
            return df_clean
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            
            # Fallback: try reading with different parameters
            try:
                logger.info("Trying fallback reading method...")
                df = pd.read_csv(filepath, on_bad_lines='skip', sep=',', quotechar='"', 
                               skipinitialspace=True, low_memory=False)
                
                # Remove completely empty rows
                df = df.dropna(how='all')
                
                # Skip header and get data
                df_data = df.iloc[1:].copy()
                
                logger.info(f"Fallback successful: {df_data.shape}")
                return self.clean_rb_data(df_data)
                
            except Exception as e2:
                logger.error(f"Fallback also failed: {e2}")
                return None
    
    def clean_rb_data(self, df):
        """Clean and prepare RB data"""
        logger.info("Cleaning RB data...")
        
        # Remove rows that are clearly headers or invalid
        if len(df.columns) > 1:
            # Remove rows where Player column contains header text
            player_col = df.columns[1]  # Usually 'Player' column
            df = df[~df[player_col].str.contains('Player|Rk|Team', na=False, case=False)]
        
        # Remove rows with missing essential data
        essential_cols = []
        if 'Player' in df.columns:
            essential_cols.append('Player')
        if 'Team' in df.columns:
            essential_cols.append('Team')
        if len(df.columns) >= 8:  # Assuming attempts column exists
            essential_cols.append(df.columns[7])  # Usually 'Att'
            
        if essential_cols:
            df = df.dropna(subset=essential_cols)
        
        # Convert numeric columns
        numeric_columns = []
        for i, col in enumerate(df.columns):
            if i >= 2:  # Skip first few text columns
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    numeric_columns.append(col)
                except:
                    pass
        
        logger.info(f"Converted {len(numeric_columns)} numeric columns")
        
        # Remove rows with all-zero or invalid numeric data
        if numeric_columns:
            # Keep rows where at least one numeric column has valid data > 0
            numeric_df = df[numeric_columns]
            valid_mask = (numeric_df > 0).any(axis=1)
            df = df[valid_mask]
        
        # Fill remaining NaN values
        df = df.fillna(0)
        
        logger.info(f"Final cleaned data: {df.shape}")
        if len(df) > 0:
            logger.info(f"Sample players: {df.iloc[:5, 1].tolist() if len(df.columns) > 1 else 'N/A'}")
        
        return df
    
    def engineer_features(self, df):
        """Create training features from the RB data"""
        logger.info("Engineering features for RB data...")
        
        if df.empty:
            logger.error("No data to process")
            return pd.DataFrame()
        
        features_list = []
        
        for idx, row in df.iterrows():
            try:
                # Extract basic stats (adjust column indices based on actual data structure)
                if len(df.columns) >= 10:
                    player = str(row.iloc[1]) if len(row) > 1 else 'Unknown'
                    team = str(row.iloc[3]) if len(row) > 3 else 'Unknown'
                    age = pd.to_numeric(row.iloc[2], errors='coerce') if len(row) > 2 else 25
                    games = pd.to_numeric(row.iloc[5], errors='coerce') if len(row) > 5 else 1
                    attempts = pd.to_numeric(row.iloc[7], errors='coerce') if len(row) > 7 else 0
                    yards = pd.to_numeric(row.iloc[8], errors='coerce') if len(row) > 8 else 0
                    
                    # Skip invalid records
                    if attempts <= 0 or games <= 0:
                        continue
                    
                    # Create features
                    features = {
                        'attempts_per_game': attempts / games,
                        'yards_per_attempt': yards / attempts if attempts > 0 else 0,
                        'yards_per_game': yards / games,
                        'total_attempts': attempts,
                        'total_yards': yards,
                        'age': age,
                        'games_played': games,
                        'efficiency_score': (yards / attempts * games) if attempts > 0 else 0,
                        'volume_score': attempts * games,
                        'team_encoded': hash(team) % 32,
                        'target': yards / games  # Yards per game as target
                    }
                    
                    # Add advanced stats if available
                    if len(row) > 10:
                        ybc = pd.to_numeric(row.iloc[10], errors='coerce') if len(row) > 10 else 0
                        yac = pd.to_numeric(row.iloc[12], errors='coerce') if len(row) > 12 else 0
                        features['yards_before_contact'] = ybc / attempts if attempts > 0 else 0
                        features['yards_after_contact'] = yac / attempts if attempts > 0 else 0
                    
                    features_list.append(features)
                    
            except Exception as e:
                logger.debug(f"Error processing row {idx}: {e}")
                continue
        
        if not features_list:
            logger.error("No valid features created")
            return pd.DataFrame()
        
        features_df = pd.DataFrame(features_list)
        features_df = features_df.fillna(0)
        
        # Remove outliers
        for col in ['yards_per_attempt', 'yards_per_game']:
            if col in features_df.columns:
                q99 = features_df[col].quantile(0.99)
                features_df = features_df[features_df[col] <= q99]
        
        logger.info(f"Created {len(features_df)} feature samples with {len(features_df.columns)} features")
        logger.info(f"Features: {list(features_df.columns)}")
        
        return features_df
    
    def prepare_training_data(self, features_df):
        """Prepare final training data"""
        logger.info("Preparing training data...")
        
        if features_df.empty:
            return None, None
        
        # Separate features and target
        X = features_df.drop(['target'], axis=1, errors='ignore')
        y = features_df['target'] if 'target' in features_df.columns else features_df.iloc[:, 0]
        
        logger.info(f"Training data: X={X.shape}, y={y.shape}")
        logger.info(f"Target stats: mean={y.mean():.2f}, std={y.std():.2f}, min={y.min():.2f}, max={y.max():.2f}")
        
        return X, y
    
    def save_processed_data(self, X, y, output_path='processed_rb_data.csv'):
        """Save processed data"""
        if X is not None and y is not None:
            combined_df = X.copy()
            combined_df['target'] = y
            combined_df.to_csv(output_path, index=False)
            logger.info(f"Saved processed RB data to {output_path}")
        return output_path

def main():
    """Main processing function"""
    print("=== Robust RB Historical Data Processor ===")
    print("Processing 13,245-line RB dataset for XGBoost training...")
    
    processor = RobustRBProcessor()
    
    # Load and clean data
    df = processor.load_and_clean_data()
    if df is None or df.empty:
        print("❌ Failed to load data. Exiting.")
        return None, None, None
    
    # Engineer features  
    features_df = processor.engineer_features(df)
    if features_df.empty:
        print("❌ Failed to create features. Exiting.")
        return None, None, None
    
    # Prepare training data
    X, y = processor.prepare_training_data(features_df)
    if X is None:
        print("❌ Failed to prepare training data. Exiting.")
        return None, None, None
    
    # Save processed data
    output_file = processor.save_processed_data(X, y)
    
    print(f"\n✅ === Processing Complete ===")
    print(f"📊 RB dataset processed successfully")
    print(f"📁 Output: {output_file}")
    print(f"🔢 Samples: {len(X)}")
    print(f"🎯 Features: {len(X.columns)}")
    print(f"📈 Ready for XGBoost training")
    
    return X, y, processor

if __name__ == "__main__":
    main()