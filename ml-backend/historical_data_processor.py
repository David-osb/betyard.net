"""
NFL Historical Data Processor (2000-2025)
Processes 26 years of real NFL passing data for XGBoost training
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NFLHistoricalProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.team_encoder = LabelEncoder()
        self.position_encoder = LabelEncoder()
        
    def load_and_clean_data(self, filepath='nfl_data/qb_stats_historical.csv'):
        """Load and clean the 26-year historical dataset"""
        logger.info("Loading 26 years of NFL historical data...")
        
        try:
            # Read the CSV, handling the complex header structure
            df = pd.read_csv(filepath, skiprows=1)  # Skip the first header row
            
            # Remove the last summary row if it exists
            df = df[df['Player'] != 'League Average'].copy()
            
            logger.info(f"Loaded {len(df)} player records from 2000-2025")
            
            # Clean and rename key columns for XGBoost
            df = self._clean_column_names(df)
            df = self._clean_data_types(df)
            df = self._filter_quarterbacks(df)
            df = self._create_derived_features(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def _clean_column_names(self, df):
        """Clean and standardize column names"""
        # Key column mappings for XGBoost features
        column_mapping = {
            'Player': 'player_name',
            'Age': 'age',
            'Team': 'team',
            'Pos': 'position',
            'G': 'games',
            'GS': 'games_started',
            'Cmp': 'completions',
            'Att': 'attempts',
            'IAY': 'intended_air_yards',
            'IAY/PA': 'air_yards_per_attempt',
            'CAY': 'completed_air_yards', 
            'CAY/Cmp': 'air_yards_per_completion',
            'CAY/PA': 'completed_air_yards_per_attempt',
            'YAC': 'yards_after_catch',
            'YAC/Cmp': 'yac_per_completion',
            'Drops': 'drops',
            'Drop%': 'drop_percentage',
            'BadTh': 'bad_throws',
            'Bad%': 'bad_throw_percentage',
            'OnTgt': 'on_target_throws',
            'OnTgt%': 'on_target_percentage',
            'PktTime': 'pocket_time',
            'Bltz': 'blitzes_faced',
            'Hrry': 'hurries',
            'Hits': 'hits_taken',
            'Prss': 'pressures',
            'Prss%': 'pressure_percentage',
            'Scrm': 'scrambles'
        }
        
        # Rename columns that exist
        existing_columns = {k: v for k, v in column_mapping.items() if k in df.columns}
        df = df.rename(columns=existing_columns)
        
        return df
    
    def _clean_data_types(self, df):
        """Convert data types and handle missing values"""
        # Numeric columns to convert
        numeric_cols = [
            'age', 'games', 'games_started', 'completions', 'attempts',
            'intended_air_yards', 'air_yards_per_attempt', 'completed_air_yards',
            'air_yards_per_completion', 'completed_air_yards_per_attempt',
            'yards_after_catch', 'yac_per_completion', 'drops', 'drop_percentage',
            'bad_throws', 'bad_throw_percentage', 'on_target_throws', 
            'on_target_percentage', 'pocket_time', 'blitzes_faced', 'hurries',
            'hits_taken', 'pressures', 'pressure_percentage', 'scrambles'
        ]
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Fill missing values with 0 for counting stats, median for rates
        counting_stats = ['drops', 'bad_throws', 'on_target_throws', 'blitzes_faced', 
                         'hurries', 'hits_taken', 'pressures', 'scrambles']
        
        for col in counting_stats:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        # Fill rate stats with median
        rate_stats = ['drop_percentage', 'bad_throw_percentage', 'on_target_percentage', 
                     'pressure_percentage', 'pocket_time']
        
        for col in rate_stats:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        
        return df
    
    def _filter_quarterbacks(self, df):
        """Include ALL players with any meaningful data"""
        # Remove only completely invalid rows (no position data)
        if 'position' in df.columns:
            df = df[df['position'].notna()].copy()
        
        # Include ALL attempt levels - no filtering by volume
        if 'attempts' in df.columns:
            df = df[df['attempts'].notna()].copy()
        
        logger.info(f"Included {len(df)} total player records (all positions, all volumes)")
        return df
    
    def _create_derived_features(self, df):
        """Create advanced features for XGBoost"""
        # Completion percentage
        if 'completions' in df.columns and 'attempts' in df.columns:
            df['completion_percentage'] = (df['completions'] / df['attempts'] * 100).round(1)
        
        # Yards per attempt (estimated from air yards + YAC)
        if 'completed_air_yards' in df.columns and 'yards_after_catch' in df.columns:
            df['total_yards'] = df['completed_air_yards'] + df['yards_after_catch']
            if 'attempts' in df.columns:
                df['yards_per_attempt'] = (df['total_yards'] / df['attempts']).round(1)
        
        # Efficiency under pressure
        if 'pressures' in df.columns and 'attempts' in df.columns:
            df['pressure_rate'] = (df['pressures'] / df['attempts'] * 100).round(1)
        
        # Accuracy rating (combination of on-target % and drop %)
        if 'on_target_percentage' in df.columns and 'drop_percentage' in df.columns:
            df['accuracy_rating'] = (df['on_target_percentage'] - df['drop_percentage']).round(1)
        
        # Fantasy points estimation (basic formula)
        if 'total_yards' in df.columns:
            # Passing yards / 25 = 1 point, assume 1 TD per 40 attempts
            df['estimated_fantasy_points'] = (
                (df['total_yards'] / 25) + 
                (df['attempts'] / 40 * 4)  # 4 points per TD
            ).round(1)
        
        # Era classification for model features
        df['era'] = 'modern'  # Default to modern era
        if 'age' in df.columns and 'player_name' in df.columns:
            # Estimate season year from current data (rough approximation)
            current_year = 2025
            df['estimated_season'] = current_year - (df['age'] - 25)  # Rough estimate
            
            df.loc[df['estimated_season'] <= 2010, 'era'] = 'legacy'
            df.loc[(df['estimated_season'] > 2010) & (df['estimated_season'] <= 2018), 'era'] = 'transitional'
        
        return df
    
    def prepare_for_xgboost(self, df):
        """Prepare data specifically for XGBoost training - ALL POSITIONS"""
        logger.info("Preparing ALL position data for XGBoost training...")
        
        # Select features for XGBoost (available across all positions)
        feature_columns = [
            'age', 'games', 'attempts', 
            'air_yards_per_attempt', 'yac_per_completion', 'yards_per_attempt',
            'on_target_percentage', 'pressure_rate', 'accuracy_rating',
            'pocket_time', 'estimated_fantasy_points'
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
        
        # Create target variable (estimated fantasy points or fallback)
        if 'estimated_fantasy_points' in df.columns:
            y = df['estimated_fantasy_points'].fillna(0)
        elif 'yards_per_attempt' in df.columns:
            y = df['yards_per_attempt'].fillna(0) * 2  # Simple target
        else:
            # Emergency fallback - use attempts as basic performance indicator
            y = df['attempts'].fillna(0) / 10
        
        # Add position encoding as feature
        if 'position' in df.columns:
            # Create position encoding
            position_map = {
                'QB': 1, 'RB': 2, 'WR': 3, 'TE': 4, 
                'P': 5, 'K': 6, 'FB': 7, 'DB': 8
            }
            X['position_encoded'] = df['position'].map(position_map).fillna(0)
            available_features.append('position_encoded')
        
        # Add team encoding if available
        if 'team' in df.columns:
            # Simple team encoding (could be enhanced)
            unique_teams = df['team'].unique()
            team_map = {team: i+1 for i, team in enumerate(unique_teams) if pd.notna(team)}
            X['team_encoded'] = df['team'].map(team_map).fillna(0)
            available_features.append('team_encoded')
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        logger.info(f"Prepared {len(X_scaled)} samples with {len(X_scaled.columns)} features")
        logger.info(f"Position breakdown: {df['position'].value_counts().to_dict()}")
        
        return X_scaled, y, available_features
    
    def save_preprocessor(self, filepath='models/historical_preprocessor.pkl'):
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
        
        logger.info(f"Preprocessor saved to {filepath}")


def main():
    """Test the historical data processor"""
    processor = NFLHistoricalProcessor()
    
    try:
        # Load and process the historical data
        df = processor.load_and_clean_data()
        
        print(f"\n📊 HISTORICAL DATA SUMMARY:")
        print(f"Total QB seasons: {len(df)}")
        print(f"Years covered: 2000-2025 (26 years)")
        print(f"Columns available: {len(df.columns)}")
        
        # Show sample of processed data
        print(f"\n🏈 SAMPLE PROCESSED DATA:")
        if 'player_name' in df.columns:
            sample_cols = ['player_name', 'age', 'team', 'attempts', 'completion_percentage', 
                          'yards_per_attempt', 'estimated_fantasy_points']
            available_cols = [col for col in sample_cols if col in df.columns]
            print(df[available_cols].head(10).to_string(index=False))
        
        # Prepare for XGBoost
        X, y, features = processor.prepare_for_xgboost(df)
        
        print(f"\n🤖 XGBOOST READY DATA:")
        print(f"Training samples: {len(X)}")
        print(f"Features: {len(features)}")
        print(f"Feature names: {features}")
        print(f"Target range: {y.min():.1f} to {y.max():.1f}")
        
        # Save preprocessor
        processor.save_preprocessor()
        
        print(f"\n✅ SUCCESS: Historical data processor ready!")
        print(f"🎯 Ready to train XGBoost with 26 years of real NFL data")
        
        return X, y, processor
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        raise


if __name__ == "__main__":
    X, y, processor = main()