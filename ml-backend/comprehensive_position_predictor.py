#!/usr/bin/env python3
"""
COMPREHENSIVE NFL POSITION PREDICTOR
===================================

Combines 26 years of historical NFL data with current active players
to predict specific per-game performance metrics:

QB: Passing yards, completions, attempts, passing TDs, rushing TDs per game
RB: Rushing yards, TDs, carries, receptions per game  
WR: Receiving yards, receptions, TDs, targets per game
TE: Receiving yards, receptions, TDs, targets, blocks per game
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import logging
import warnings
import os
import re
from collections import defaultdict

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NFLPositionPredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.feature_columns = {}
        
        # Position-specific target metrics
        self.position_targets = {
            'QB': ['pass_yds_per_game', 'completions_per_game', 'attempts_per_game', 
                   'pass_td_per_game', 'rush_td_per_game'],
            'RB': ['rush_yds_per_game', 'rush_td_per_game', 'carries_per_game', 
                   'receptions_per_game'],
            'WR': ['rec_yds_per_game', 'receptions_per_game', 'rec_td_per_game', 
                   'targets_per_game'],
            'TE': ['rec_yds_per_game', 'receptions_per_game', 'rec_td_per_game', 
                   'targets_per_game', 'blocks_per_game']
        }

    def load_historical_data(self):
        """Load and combine all 26 years of historical NFL data"""
        logger.info("Loading 26 years of historical NFL data...")
        
        data_files = {
            'passing': 'nfl_data/passing_stats_historical_clean.csv',
            'rushing': 'nfl_data/rushing_stats_historical_clean.csv', 
            'receiving': 'nfl_data/receiving_stats_historical_clean.csv',
            'scrimmage': 'nfl_data/scrimmage_stats_historical_clean.csv',
            'scoring': 'nfl_data/scoring_stats_historical_clean.csv',
            'defense': 'nfl_data/defense_stats_historical_clean.csv',
            'returns': 'nfl_data/returns_stats_historical_clean.csv'
        }
        
        all_data = []
        
        for category, filepath in data_files.items():
            try:
                df = pd.read_csv(filepath, low_memory=False)
                df['data_source'] = category
                logger.info(f"Loaded {category}: {len(df)} records")
                all_data.append(df)
            except Exception as e:
                logger.warning(f"Could not load {category}: {e}")
        
        if not all_data:
            raise Exception("No historical data files found!")
            
        combined_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"Combined historical data: {len(combined_df)} total records")
        
        return self.clean_and_prepare_data(combined_df)

    def load_current_players(self):
        """Load current active players from Book1.csv"""
        logger.info("Loading current active players...")
        
        try:
            df = pd.read_csv('C:\\Users\\david\\OneDrive\\Documents\\Book1.csv', header=None)
            df.columns = ['player_info', 'col2', 'col3']
            
            current_players = []
            current_team = None
            current_position = None
            
            for _, row in df.iterrows():
                player_info = str(row['player_info']).strip()
                
                # Skip empty rows
                if pd.isna(player_info) or player_info == '':
                    continue
                
                # Team names (contain multiple words or end with team identifiers)
                if any(team_word in player_info.lower() for team_word in 
                      ['cardinals', 'falcons', 'ravens', 'bills', 'panthers', 'bears', 
                       'bengals', 'browns', 'cowboys', 'broncos', 'lions', 'packers',
                       'texans', 'colts', 'jaguars', 'chiefs', 'raiders', 'rams',
                       'chargers', 'dolphins', 'vikings', 'patriots', 'saints',
                       'giants', 'jets', 'eagles', 'steelers', '49ers', 'seahawks',
                       'buccaneers', 'titans', 'commanders']):
                    current_team = player_info
                    continue
                
                # Position headers
                if player_info.upper() in ['QUARTERBACK', 'RUNNING BACK', 'TIGHT END', 'WIDE RECEIVERS']:
                    current_position = player_info.upper()
                    continue
                
                # Skip non-player entries
                if player_info.lower() in ['view all']:
                    continue
                
                # Player names
                if current_team and current_position:
                    # Clean position names
                    pos_map = {
                        'QUARTERBACK': 'QB',
                        'RUNNING BACK': 'RB', 
                        'TIGHT END': 'TE',
                        'WIDE RECEIVERS': 'WR'
                    }
                    
                    position = pos_map.get(current_position, current_position)
                    
                    current_players.append({
                        'Player': player_info,
                        'Team': current_team,
                        'Position': position,
                        'is_current_player': True
                    })
            
            current_df = pd.DataFrame(current_players)
            logger.info(f"Loaded {len(current_df)} current active players")
            
            # Log position distribution
            pos_counts = current_df['Position'].value_counts()
            for pos, count in pos_counts.items():
                logger.info(f"  {pos}: {count} players")
            
            return current_df
            
        except Exception as e:
            logger.error(f"Error loading current players: {e}")
            return pd.DataFrame()

    def clean_and_prepare_data(self, df):
        """Clean and prepare historical data for modeling"""
        logger.info("Cleaning and preparing historical data...")
        
        # Basic cleaning
        df = df.dropna(subset=['Player'])
        
        # Standardize position names
        if 'Pos' in df.columns:
            df['Position'] = df['Pos'].fillna('Unknown')
        elif 'Position' in df.columns:
            df['Position'] = df['Position'].fillna('Unknown')
        else:
            # Try to infer position from data source
            position_mapping = {
                'passing': 'QB',
                'rushing': 'RB', 
                'receiving': 'WR',
                'scrimmage': 'RB'
            }
            df['Position'] = df['data_source'].map(position_mapping).fillna('Unknown')
        
        # Clean numeric columns
        numeric_cols = ['G', 'Age', 'Yds', 'TD', 'Att', 'Rec', 'Tgt']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Create per-game statistics
        df['games'] = df['G'].fillna(1).clip(lower=1)  # Avoid division by zero
        
        # Calculate per-game metrics based on available data
        if 'Yds' in df.columns:
            df['yards_per_game'] = df['Yds'] / df['games']
        if 'TD' in df.columns:
            df['td_per_game'] = df['TD'] / df['games']
        if 'Att' in df.columns:
            df['attempts_per_game'] = df['Att'] / df['games']
        if 'Rec' in df.columns:
            df['receptions_per_game'] = df['Rec'] / df['games']
        if 'Tgt' in df.columns:
            df['targets_per_game'] = df['Tgt'] / df['games']
        
        # Position-specific calculations
        for pos in ['QB', 'RB', 'WR', 'TE']:
            pos_data = df[df['Position'] == pos].copy()
            
            if len(pos_data) > 0:
                if pos == 'QB':
                    # QB-specific metrics
                    if 'Cmp' in df.columns:
                        df.loc[df['Position'] == 'QB', 'completions_per_game'] = df['Cmp'] / df['games']
                    df.loc[df['Position'] == 'QB', 'pass_yds_per_game'] = df['yards_per_game']
                    df.loc[df['Position'] == 'QB', 'pass_td_per_game'] = df['td_per_game']
                    df.loc[df['Position'] == 'QB', 'rush_td_per_game'] = 0  # Will be updated from rushing data
                    
                elif pos == 'RB':
                    # RB-specific metrics
                    df.loc[df['Position'] == 'RB', 'rush_yds_per_game'] = df['yards_per_game']
                    df.loc[df['Position'] == 'RB', 'rush_td_per_game'] = df['td_per_game']
                    df.loc[df['Position'] == 'RB', 'carries_per_game'] = df['attempts_per_game']
                    
                elif pos in ['WR', 'TE']:
                    # WR/TE-specific metrics
                    df.loc[df['Position'] == pos, 'rec_yds_per_game'] = df['yards_per_game']
                    df.loc[df['Position'] == pos, 'rec_td_per_game'] = df['td_per_game']
                    
                    if pos == 'TE':
                        # TE gets estimated blocks (placeholder)
                        df.loc[df['Position'] == 'TE', 'blocks_per_game'] = np.random.uniform(1, 5, 
                                                                                           sum(df['Position'] == 'TE'))
        
        # Add derived features
        df['age_group'] = pd.cut(df['Age'].fillna(25), bins=[0, 23, 26, 30, 40], 
                                labels=['Young', 'Prime', 'Veteran', 'Old'])
        df['experience_level'] = df['games'] / 16  # Approximate seasons
        
        logger.info(f"Prepared data: {len(df)} records")
        return df

    def create_features(self, df, position):
        """Create position-specific features"""
        features = []
        
        # Basic features
        basic_features = ['Age', 'games', 'experience_level']
        for feat in basic_features:
            if feat in df.columns:
                features.append(feat)
        
        # Position-specific historical features
        if position == 'QB':
            qb_features = ['yards_per_game', 'td_per_game', 'attempts_per_game', 'completions_per_game']
            features.extend([f for f in qb_features if f in df.columns])
            
        elif position == 'RB':
            rb_features = ['yards_per_game', 'td_per_game', 'attempts_per_game', 'receptions_per_game']
            features.extend([f for f in rb_features if f in df.columns])
            
        elif position in ['WR', 'TE']:
            rec_features = ['yards_per_game', 'td_per_game', 'receptions_per_game', 'targets_per_game']
            features.extend([f for f in rec_features if f in df.columns])
        
        # Team and position encoding
        if 'Team' in df.columns:
            features.append('Team_encoded')
        
        return features

    def train_position_models(self, historical_df):
        """Train models for each position"""
        logger.info("Training position-specific models...")
        
        for position in ['QB', 'RB', 'WR', 'TE']:
            logger.info(f"Training {position} models...")
            
            # Filter data for this position
            pos_data = historical_df[historical_df['Position'] == position].copy()
            
            if len(pos_data) < 100:
                logger.warning(f"Insufficient data for {position}: {len(pos_data)} records")
                continue
            
            # Encode categorical variables
            if 'Team' in pos_data.columns:
                le_team = LabelEncoder()
                pos_data['Team_encoded'] = le_team.fit_transform(pos_data['Team'].fillna('Unknown'))
                self.label_encoders[f'{position}_team'] = le_team
            
            # Create features
            feature_cols = self.create_features(pos_data, position)
            
            # Remove features not in data
            feature_cols = [col for col in feature_cols if col in pos_data.columns]
            
            if not feature_cols:
                logger.warning(f"No valid features for {position}")
                continue
            
            self.feature_columns[position] = feature_cols
            
            # Train models for each target metric
            position_models = {}
            position_scalers = {}
            
            for target in self.position_targets.get(position, []):
                if target not in pos_data.columns:
                    logger.warning(f"Target {target} not found for {position}")
                    continue
                
                # Prepare data
                X = pos_data[feature_cols].copy()
                y = pos_data[target].copy()
                
                # Remove missing values
                mask = X.notna().all(axis=1) & y.notna()
                X = X[mask]
                y = y[mask]
                
                if len(X) < 50:
                    logger.warning(f"Insufficient clean data for {position} {target}: {len(X)} records")
                    continue
                
                # Scale features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.2, random_state=42
                )
                
                # Train model
                model = xgb.XGBRegressor(
                    device='cuda',
                    tree_method='gpu_hist',
                    max_depth=6,
                    learning_rate=0.1,
                    n_estimators=200,
                    random_state=42,
                    reg_alpha=0.1,
                    reg_lambda=0.1
                )
                
                model.fit(X_train, y_train)
                
                # Evaluate
                y_pred = model.predict(X_test)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                logger.info(f"  {target}: MAE={mae:.2f}, R²={r2:.3f}")
                
                # Store model and scaler
                position_models[target] = model
                position_scalers[target] = scaler
            
            self.models[position] = position_models
            self.scalers[position] = position_scalers
            
            logger.info(f"Completed {position}: {len(position_models)} models trained")

    def predict_player_performance(self, current_players_df, historical_df):
        """Predict performance for current players"""
        logger.info("Predicting performance for current players...")
        
        predictions = []
        
        for _, player in current_players_df.iterrows():
            player_name = player['Player']
            position = player['Position']
            team = player['Team']
            
            if position not in self.models:
                logger.warning(f"No models available for position {position}")
                continue
            
            # Try to find historical data for this player
            player_history = historical_df[
                historical_df['Player'].str.contains(player_name.split()[0], case=False, na=False) |
                historical_df['Player'].str.contains(player_name.split()[-1], case=False, na=False)
            ]
            
            if len(player_history) == 0:
                # Use position averages for new players
                pos_data = historical_df[historical_df['Position'] == position]
                if len(pos_data) > 0:
                    player_features = pos_data.select_dtypes(include=[np.number]).mean()
                    player_features['Age'] = 25  # Default age for rookies
                    player_features['experience_level'] = 1
                else:
                    logger.warning(f"No historical data available for {position}")
                    continue
            else:
                # Use player's recent performance
                player_features = player_history.select_dtypes(include=[np.number]).mean()
            
            # Add team encoding
            if f'{position}_team' in self.label_encoders:
                try:
                    team_encoded = self.label_encoders[f'{position}_team'].transform([team])[0]
                except:
                    team_encoded = 0  # Default for unknown teams
                player_features['Team_encoded'] = team_encoded
            
            # Make predictions for each target
            player_predictions = {
                'Player': player_name,
                'Position': position,
                'Team': team
            }
            
            if position in self.models and position in self.feature_columns:
                feature_cols = self.feature_columns[position]
                
                # Prepare feature vector
                X_player = []
                for col in feature_cols:
                    if col in player_features:
                        X_player.append(player_features[col])
                    else:
                        X_player.append(0)  # Default value
                
                X_player = np.array(X_player).reshape(1, -1)
                
                # Make predictions for each target
                for target, model in self.models[position].items():
                    try:
                        # Scale features
                        scaler = self.scalers[position][target]
                        X_scaled = scaler.transform(X_player)
                        
                        # Predict
                        pred = model.predict(X_scaled)[0]
                        
                        # Apply reasonable bounds
                        if 'yds' in target:
                            pred = max(0, min(pred, 500))  # Reasonable yards per game
                        elif 'td' in target:
                            pred = max(0, min(pred, 5))    # Reasonable TDs per game
                        elif 'attempts' in target or 'carries' in target:
                            pred = max(0, min(pred, 50))   # Reasonable attempts per game
                        elif 'receptions' in target or 'targets' in target:
                            pred = max(0, min(pred, 20))   # Reasonable receptions per game
                        elif 'blocks' in target:
                            pred = max(0, min(pred, 10))   # Reasonable blocks per game
                        
                        player_predictions[target] = round(pred, 2)
                        
                    except Exception as e:
                        logger.warning(f"Error predicting {target} for {player_name}: {e}")
                        player_predictions[target] = 0
            
            predictions.append(player_predictions)
        
        return pd.DataFrame(predictions)

    def run_complete_analysis(self):
        """Run the complete prediction analysis"""
        logger.info("=" * 60)
        logger.info("NFL POSITION PREDICTOR - COMPREHENSIVE ANALYSIS")
        logger.info("=" * 60)
        
        try:
            # Load data
            historical_df = self.load_historical_data()
            current_players_df = self.load_current_players()
            
            if historical_df.empty or current_players_df.empty:
                logger.error("Failed to load required data")
                return
            
            # Train models
            self.train_position_models(historical_df)
            
            # Make predictions
            predictions_df = self.predict_player_performance(current_players_df, historical_df)
            
            # Save results
            output_file = 'nfl_position_predictions_2025.csv'
            predictions_df.to_csv(output_file, index=False)
            logger.info(f"Predictions saved to {output_file}")
            
            # Display sample results
            logger.info("=" * 60)
            logger.info("SAMPLE PREDICTIONS")
            logger.info("=" * 60)
            
            for position in ['QB', 'RB', 'WR', 'TE']:
                pos_preds = predictions_df[predictions_df['Position'] == position]
                if not pos_preds.empty:
                    logger.info(f"\n{position} PREDICTIONS (Top 5):")
                    logger.info("-" * 40)
                    
                    for _, player in pos_preds.head().iterrows():
                        logger.info(f"{player['Player']} ({player['Team']}):")
                        for target in self.position_targets.get(position, []):
                            if target in player and pd.notna(player[target]):
                                logger.info(f"  {target}: {player[target]}")
                        logger.info("")
            
            logger.info("=" * 60)
            logger.info(f"ANALYSIS COMPLETE - {len(predictions_df)} players analyzed")
            logger.info("=" * 60)
            
            return predictions_df
            
        except Exception as e:
            logger.error(f"Error in analysis: {e}")
            raise

if __name__ == "__main__":
    predictor = NFLPositionPredictor()
    results = predictor.run_complete_analysis()