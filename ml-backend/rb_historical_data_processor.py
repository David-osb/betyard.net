"""
NFL RB Historical Data Processor (2000-2025)
Processes 26 years of real NFL rushing data for XGBoost training
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NFLRBHistoricalProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.team_encoder = LabelEncoder()
        self.position_encoder = LabelEncoder()
        
    def load_and_clean_data(self, filepath='nfl_data/rb_stats_historical.csv'):
        """Load and clean the 26-year RB historical dataset"""
        logger.info("Loading 26 years of NFL RB historical data...")
        
        try:
            # Read the CSV - this appears to be game-by-game data, not season totals
            df = pd.read_csv(filepath)
            
            # Print column info for debugging
            logger.info(f"Columns found: {list(df.columns)}")
            logger.info(f"Number of columns: {len(df.columns)}")
            
            logger.info(f"Loaded {len(df)} game records from 2000-2025")
            
            # Clean and rename key columns for XGBoost
            df = self._clean_column_names(df)
            df = self._clean_data_types(df)
            df = self._filter_players(df)
            df = self._create_derived_features(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def _clean_column_names(self, df):
        """Clean and standardize column names for game-by-game data"""
        # The RB data appears to be game-by-game with different structure
        # Keep original column names but clean them
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Standardize key columns
        column_mapping = {
            'player_name': 'player_name',
            'team': 'team', 
            'season': 'season',
            'week': 'week',
            'opponent': 'opponent',
            'home_game': 'home_game',
            'rushing_yards': 'rushing_yards',
            'rushing_tds': 'rushing_tds', 
            'rushing_attempts': 'attempts',
            'receiving_yards': 'receiving_yards',
            'receiving_tds': 'receiving_tds',
            'receptions': 'receptions',
            'targets': 'targets',
            'fantasy_points': 'fantasy_points',
            'game_result': 'game_result',
            'team_score': 'team_score',
            'opponent_score': 'opponent_score',
            'temperature': 'temperature',
            'wind_speed': 'wind_speed',
            'dome_game': 'dome_game'
        }
        
        # Rename columns that exist
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                df = df.rename(columns={old_name: new_name})
        
        return df
    
    def _clean_data_types(self, df):
        """Convert data types and handle missing values for game-by-game data"""
        # Numeric columns to convert
        numeric_cols = [
            'season', 'week', 'home_game', 'rushing_yards', 'rushing_tds',
            'attempts', 'receiving_yards', 'receiving_tds', 'receptions',
            'targets', 'fantasy_points', 'team_score', 'opponent_score',
            'temperature', 'wind_speed', 'dome_game'
        ]
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Fill missing values with 0 for counting stats
        counting_stats = ['rushing_yards', 'rushing_tds', 'attempts', 
                         'receiving_yards', 'receiving_tds', 'receptions', 'targets']
        
        for col in counting_stats:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        # Fill environmental stats with reasonable defaults
        if 'temperature' in df.columns:
            df['temperature'] = df['temperature'].fillna(72)  # Default temp
        if 'wind_speed' in df.columns:
            df['wind_speed'] = df['wind_speed'].fillna(0)  # Default no wind
        if 'dome_game' in df.columns:
            df['dome_game'] = df['dome_game'].fillna(0)  # Default outdoor
        
        return df
    
    def _filter_players(self, df):
        """Include ALL game records with any meaningful data"""
        # Remove only completely invalid rows (no player name)
        if 'player_name' in df.columns:
            df = df[df['player_name'].notna()].copy()
        
        # Include ALL games - no filtering by volume (this is game-by-game data)
        logger.info(f"Included {len(df)} total game records (all players)")
        return df
    
    def _create_derived_features(self, df):
        """Create advanced features for XGBoost from game-by-game data"""
        # Yards per attempt
        if 'rushing_yards' in df.columns and 'attempts' in df.columns:
            df['yards_per_attempt'] = np.where(
                df['attempts'] > 0, 
                df['rushing_yards'] / df['attempts'], 
                0
            ).round(2)
        
        # Total touchdowns
        df['total_tds'] = 0
        if 'rushing_tds' in df.columns:
            df['total_tds'] += df['rushing_tds'].fillna(0)
        if 'receiving_tds' in df.columns:
            df['total_tds'] += df['receiving_tds'].fillna(0)
        
        # Total yards (rushing + receiving)
        df['total_yards'] = 0
        if 'rushing_yards' in df.columns:
            df['total_yards'] += df['rushing_yards'].fillna(0)
        if 'receiving_yards' in df.columns:
            df['total_yards'] += df['receiving_yards'].fillna(0)
        
        # Reception rate for receiving backs
        if 'receptions' in df.columns and 'targets' in df.columns:
            df['reception_rate'] = np.where(
                df['targets'] > 0,
                df['receptions'] / df['targets'] * 100,
                0
            ).round(1)
        
        # Game script (win/loss/close game)
        if 'team_score' in df.columns and 'opponent_score' in df.columns:
            df['score_differential'] = df['team_score'] - df['opponent_score']
            df['close_game'] = (abs(df['score_differential']) <= 7).astype(int)
        
        # Weather impact
        if 'temperature' in df.columns and 'wind_speed' in df.columns:
            df['bad_weather'] = ((df['temperature'] < 32) | (df['wind_speed'] > 15)).astype(int)
        
        # Use actual fantasy points if available, otherwise estimate
        if 'fantasy_points' not in df.columns:
            # Standard fantasy scoring: 0.1 per rush yard, 6 per TD, 0.1 per rec yard, 1 per reception
            df['estimated_fantasy_points'] = (
                (df['rushing_yards'] * 0.1) + 
                (df['rushing_tds'] * 6) +
                (df['receiving_yards'] * 0.1) +
                (df['receptions'] * 1) +
                (df['receiving_tds'] * 6)
            ).round(1)
        
        # Usage rate (attempts + targets as measure of involvement)
        df['usage'] = 0
        if 'attempts' in df.columns:
            df['usage'] += df['attempts'].fillna(0)
        if 'targets' in df.columns:
            df['usage'] += df['targets'].fillna(0)
        
        return df
    
    def prepare_for_xgboost(self, df):
        """Prepare data specifically for XGBoost training - GAME-BY-GAME DATA"""
        logger.info("Preparing game-by-game RB data for XGBoost training...")
        
        # Select features for XGBoost (available in game-by-game data)
        feature_columns = [
            'week', 'home_game', 'attempts', 'yards_per_attempt', 
            'total_yards', 'total_tds', 'reception_rate', 'usage',
            'score_differential', 'close_game', 'bad_weather',
            'temperature', 'wind_speed', 'dome_game'
        ]
        
        # Only include columns that exist and have data
        available_features = []
        for col in feature_columns:
            if col in df.columns and not df[col].isna().all():
                available_features.append(col)
        
        if not available_features:
            raise ValueError("No suitable features found for XGBoost training")
        
        # Create feature matrix
        X = df[available_features].copy()
        
        # Handle any remaining NaN values and infinite values
        X = X.fillna(0)  # Fill with 0 for missing stats
        
        # Clean infinite and extreme values
        X = X.replace([np.inf, -np.inf], 0)  # Replace infinity with 0
        
        # Cap extreme values to reasonable ranges
        for col in X.columns:
            if X[col].dtype in ['float64', 'int64']:
                # Cap at 99th percentile to remove outliers
                upper_cap = X[col].quantile(0.99)
                lower_cap = X[col].quantile(0.01)
                X[col] = X[col].clip(lower=lower_cap, upper=upper_cap)
        
        # Create target variable (fantasy points)
        if 'fantasy_points' in df.columns:
            y = df['fantasy_points'].fillna(0)
        elif 'estimated_fantasy_points' in df.columns:
            y = df['estimated_fantasy_points'].fillna(0)
        else:
            # Emergency fallback - use total yards
            y = df['total_yards'].fillna(0) / 10
        
        # Add team encoding if available
        if 'team' in df.columns:
            # Simple team encoding
            unique_teams = df['team'].unique()
            team_map = {team: i+1 for i, team in enumerate(unique_teams) if pd.notna(team)}
            X['team_encoded'] = df['team'].map(team_map).fillna(0)
            available_features.append('team_encoded')
        
        # Add opponent encoding if available
        if 'opponent' in df.columns:
            # Simple opponent encoding
            unique_opponents = df['opponent'].unique()
            opp_map = {opp: i+1 for i, opp in enumerate(unique_opponents) if pd.notna(opp)}
            X['opponent_encoded'] = df['opponent'].map(opp_map).fillna(0)
            available_features.append('opponent_encoded')
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        logger.info(f"Prepared {len(X_scaled)} game samples with {len(X_scaled.columns)} features")
        
        return X_scaled, y, available_features
    
    def save_preprocessor(self, filepath='models/rb_historical_preprocessor.pkl'):
        """Save the fitted preprocessor"""
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        preprocessor_data = {
            'scaler': self.scaler,
            'team_encoder': self.team_encoder,
            'position_encoder': self.position_encoder
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(preprocessor_data, f)
        
        logger.info(f"RB Preprocessor saved to {filepath}")


def main():
    """Test the RB historical data processor"""
    processor = NFLRBHistoricalProcessor()
    
    try:
        # Load and process the historical data
        df = processor.load_and_clean_data()
        
        print(f"\n📊 RB HISTORICAL DATA SUMMARY:")
        print(f"Total player records: {len(df)}")
        print(f"Years covered: 2000-2025 (26 years)")
        print(f"Columns available: {len(df.columns)}")
        
        # Show sample of processed data
        print(f"\n🏈 SAMPLE PROCESSED RB DATA:")
        if 'player_name' in df.columns:
            sample_cols = ['player_name', 'age', 'team', 'position', 'attempts', 
                          'yards_per_attempt', 'estimated_fantasy_points']
            available_cols = [col for col in sample_cols if col in df.columns]
            print(df[available_cols].head(10).to_string(index=False))
        
        # Prepare for XGBoost
        X, y, features = processor.prepare_for_xgboost(df)
        
        print(f"\n🤖 XGBOOST READY RB DATA:")
        print(f"Training samples: {len(X)}")
        print(f"Features: {len(features)}")
        print(f"Feature names: {features}")
        print(f"Target range: {y.min():.1f} to {y.max():.1f}")
        
        # Save preprocessor
        processor.save_preprocessor()
        
        print(f"\n✅ SUCCESS: RB Historical data processor ready!")
        print(f"🎯 Ready to train XGBoost with 26 years of real NFL RB data")
        
        return X, y, processor
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        raise


if __name__ == "__main__":
    X, y, processor = main()