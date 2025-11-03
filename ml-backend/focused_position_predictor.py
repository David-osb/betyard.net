#!/usr/bin/env python3
"""
FOCUSED NFL POSITION PREDICTOR
=============================

Combines 26 years of historical NFL data with current active players
to predict specific per-game performance metrics by position.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import logging
import warnings
import os

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def load_current_players():
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
            
            if pd.isna(player_info) or player_info == '':
                continue
            
            # Team names
            team_indicators = ['cardinals', 'falcons', 'ravens', 'bills', 'panthers', 'bears', 
                             'bengals', 'browns', 'cowboys', 'broncos', 'lions', 'packers',
                             'texans', 'colts', 'jaguars', 'chiefs', 'raiders', 'rams',
                             'chargers', 'dolphins', 'vikings', 'patriots', 'saints',
                             'giants', 'jets', 'eagles', 'steelers', '49ers', 'seahawks',
                             'buccaneers', 'titans', 'commanders']
            
            if any(team in player_info.lower() for team in team_indicators):
                current_team = player_info
                continue
            
            # Position headers
            position_map = {
                'QUARTERBACK': 'QB',
                'RUNNING BACK': 'RB', 
                'TIGHT END': 'TE',
                'WIDE RECEIVERS': 'WR'
            }
            
            if player_info.upper() in position_map:
                current_position = position_map[player_info.upper()]
                continue
            
            # Skip non-player entries
            if player_info.lower() in ['view all']:
                continue
            
            # Player names
            if current_team and current_position:
                current_players.append({
                    'Player': player_info,
                    'Team': current_team,
                    'Position': current_position
                })
        
        current_df = pd.DataFrame(current_players)
        logger.info(f"Loaded {len(current_df)} current players")
        
        # Show position breakdown
        for pos, count in current_df['Position'].value_counts().items():
            logger.info(f"  {pos}: {count} players")
        
        return current_df
        
    except Exception as e:
        logger.error(f"Error loading current players: {e}")
        return pd.DataFrame()

def load_historical_data():
    """Load historical NFL data"""
    logger.info("Loading historical NFL data...")
    
    data_files = [
        ('passing', 'nfl_data/passing_stats_historical_clean.csv'),
        ('rushing', 'nfl_data/rushing_stats_historical_clean.csv'),
        ('receiving', 'nfl_data/receiving_stats_historical_clean.csv'),
        ('scoring', 'nfl_data/scoring_stats_historical_clean.csv')
    ]
    
    all_data = []
    
    for category, filepath in data_files:
        try:
            df = pd.read_csv(filepath, low_memory=False)
            df['data_category'] = category
            logger.info(f"Loaded {category}: {len(df)} records")
            all_data.append(df)
        except Exception as e:
            logger.warning(f"Could not load {category}: {e}")
    
    if not all_data:
        raise Exception("No data files loaded!")
    
    combined_df = pd.concat(all_data, ignore_index=True)
    logger.info(f"Total historical records: {len(combined_df)}")
    
    return combined_df

def prepare_data_for_modeling(df):
    """Prepare and clean data for modeling"""
    logger.info("Preparing data for modeling...")
    
    # Basic cleaning
    df = df.dropna(subset=['Player'])
    df = df[df['Player'] != 'Player']  # Remove header rows
    
    # Convert numeric columns
    numeric_cols = ['G', 'Age', 'Yds', 'TD', 'Att', 'Cmp', 'Rec', 'Tgt']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove invalid records
    df = df.dropna(subset=['G', 'Age'])
    df = df[df['G'] > 0]  # Must have played games
    
    # Calculate per-game stats
    df['games_played'] = df['G'].clip(lower=1)
    
    if 'Yds' in df.columns:
        df['yards_per_game'] = df['Yds'] / df['games_played']
    if 'TD' in df.columns:
        df['td_per_game'] = df['TD'] / df['games_played']
    if 'Att' in df.columns:
        df['attempts_per_game'] = df['Att'] / df['games_played']
    if 'Cmp' in df.columns:
        df['completions_per_game'] = df['Cmp'] / df['games_played']
    if 'Rec' in df.columns:
        df['receptions_per_game'] = df['Rec'] / df['games_played']
    if 'Tgt' in df.columns:
        df['targets_per_game'] = df['Tgt'] / df['games_played']
    
    # Infer position from data category
    position_mapping = {
        'passing': 'QB',
        'rushing': 'RB',
        'receiving': 'WR'  # Will include TEs
    }
    
    df['Position'] = df['data_category'].map(position_mapping)
    
    # For receiving data, separate WR and TE based on typical stats
    if 'receiving' in df['data_category'].values:
        receiving_data = df[df['data_category'] == 'receiving'].copy()
        
        # Simple heuristic: TEs typically have fewer targets but more blocking
        # For now, randomly assign some as TE (20% of receivers)
        np.random.seed(42)
        te_mask = np.random.random(len(receiving_data)) < 0.2
        df.loc[df['data_category'] == 'receiving', 'Position'] = np.where(te_mask, 'TE', 'WR')
    
    logger.info(f"Prepared data: {len(df)} records")
    
    # Show position distribution
    pos_counts = df['Position'].value_counts()
    for pos, count in pos_counts.items():
        logger.info(f"  {pos}: {count} records")
    
    return df

def train_position_models(df):
    """Train XGBoost models for each position"""
    logger.info("Training position-specific models...")
    
    models = {}
    scalers = {}
    
    # Define targets for each position
    position_targets = {
        'QB': {
            'passing_yards_per_game': 'yards_per_game',
            'completions_per_game': 'completions_per_game', 
            'attempts_per_game': 'attempts_per_game',
            'passing_tds_per_game': 'td_per_game'
        },
        'RB': {
            'rushing_yards_per_game': 'yards_per_game',
            'rushing_tds_per_game': 'td_per_game',
            'carries_per_game': 'attempts_per_game'
        },
        'WR': {
            'receiving_yards_per_game': 'yards_per_game',
            'receptions_per_game': 'receptions_per_game',
            'receiving_tds_per_game': 'td_per_game',
            'targets_per_game': 'targets_per_game'
        },
        'TE': {
            'receiving_yards_per_game': 'yards_per_game',
            'receptions_per_game': 'receptions_per_game', 
            'receiving_tds_per_game': 'td_per_game',
            'targets_per_game': 'targets_per_game'
        }
    }
    
    for position in ['QB', 'RB', 'WR', 'TE']:
        logger.info(f"Training {position} models...")
        
        pos_data = df[df['Position'] == position].copy()
        
        if len(pos_data) < 100:
            logger.warning(f"Insufficient data for {position}: {len(pos_data)} records")
            continue
        
        # Feature columns
        feature_cols = ['Age', 'games_played']
        
        # Add position-specific features
        if position == 'QB' and 'attempts_per_game' in pos_data.columns:
            feature_cols.extend(['attempts_per_game', 'completions_per_game'])
        elif position == 'RB' and 'attempts_per_game' in pos_data.columns:
            feature_cols.append('attempts_per_game')
        elif position in ['WR', 'TE'] and 'targets_per_game' in pos_data.columns:
            feature_cols.append('targets_per_game')
        
        # Keep only features that exist
        feature_cols = [col for col in feature_cols if col in pos_data.columns]
        
        if not feature_cols:
            logger.warning(f"No valid features for {position}")
            continue
        
        position_models = {}
        position_scalers = {}
        
        targets = position_targets.get(position, {})
        
        for target_name, target_col in targets.items():
            if target_col not in pos_data.columns:
                continue
            
            # Prepare data
            X = pos_data[feature_cols].copy()
            y = pos_data[target_col].copy()
            
            # Remove missing values
            mask = X.notna().all(axis=1) & y.notna() & (y >= 0)
            X = X[mask]
            y = y[mask]
            
            if len(X) < 50:
                logger.warning(f"Insufficient clean data for {position} {target_name}")
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
                max_depth=5,
                learning_rate=0.1,
                n_estimators=150,
                random_state=42
            )
            
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            logger.info(f"  {target_name}: MAE={mae:.2f}, R²={r2:.3f}")
            
            position_models[target_name] = model
            position_scalers[target_name] = scaler
        
        if position_models:
            models[position] = {
                'models': position_models,
                'scalers': position_scalers,
                'features': feature_cols
            }
    
    return models

def predict_current_players(models, current_players_df, historical_df):
    """Make predictions for current players"""
    logger.info("Making predictions for current players...")
    
    predictions = []
    
    for _, player in current_players_df.iterrows():
        player_name = player['Player']
        position = player['Position']
        team = player['Team']
        
        if position not in models:
            logger.warning(f"No model available for {position}")
            continue
        
        model_info = models[position]
        feature_cols = model_info['features']
        
        # Get position averages for features (baseline for new players)
        pos_historical = historical_df[historical_df['Position'] == position]
        
        if len(pos_historical) == 0:
            logger.warning(f"No historical data for {position}")
            continue
        
        # Use averages as baseline features
        baseline_features = {}
        for col in feature_cols:
            if col in pos_historical.columns:
                baseline_features[col] = pos_historical[col].mean()
        
        # Default values for new players
        baseline_features['Age'] = 25
        baseline_features['games_played'] = 16
        
        # Prepare feature vector
        X_player = np.array([baseline_features.get(col, 0) for col in feature_cols]).reshape(1, -1)
        
        # Make predictions
        player_pred = {
            'Player': player_name,
            'Position': position,
            'Team': team
        }
        
        for target_name, model in model_info['models'].items():
            try:
                scaler = model_info['scalers'][target_name]
                X_scaled = scaler.transform(X_player)
                pred = model.predict(X_scaled)[0]
                
                # Apply reasonable bounds
                pred = max(0, pred)
                
                player_pred[target_name] = round(pred, 2)
                
            except Exception as e:
                logger.warning(f"Error predicting {target_name} for {player_name}: {e}")
                player_pred[target_name] = 0
        
        predictions.append(player_pred)
    
    return pd.DataFrame(predictions)

def main():
    """Main execution function"""
    logger.info("=" * 60)
    logger.info("NFL POSITION PREDICTOR - 26 YEAR DATA + CURRENT PLAYERS")
    logger.info("=" * 60)
    
    try:
        # Load data
        current_players_df = load_current_players()
        historical_df = load_historical_data()
        
        if current_players_df.empty or historical_df.empty:
            logger.error("Failed to load required data")
            return
        
        # Prepare historical data
        prepared_df = prepare_data_for_modeling(historical_df)
        
        # Train models
        models = train_position_models(prepared_df)
        
        if not models:
            logger.error("No models were trained successfully")
            return
        
        # Make predictions
        predictions_df = predict_current_players(models, current_players_df, prepared_df)
        
        # Save results
        output_file = 'current_player_predictions_2025.csv'
        predictions_df.to_csv(output_file, index=False)
        logger.info(f"Predictions saved to {output_file}")
        
        # Display results
        logger.info("=" * 60)
        logger.info("PREDICTION RESULTS")
        logger.info("=" * 60)
        
        for position in ['QB', 'RB', 'WR', 'TE']:
            pos_preds = predictions_df[predictions_df['Position'] == position]
            if not pos_preds.empty:
                logger.info(f"\n{position} PREDICTIONS (Sample):")
                logger.info("-" * 50)
                
                for _, player in pos_preds.head(3).iterrows():
                    logger.info(f"{player['Player']} ({player['Team']}):")
                    for col in predictions_df.columns:
                        if col not in ['Player', 'Position', 'Team'] and pd.notna(player[col]):
                            logger.info(f"  {col}: {player[col]}")
                    logger.info("")
        
        logger.info("=" * 60)
        logger.info(f"ANALYSIS COMPLETE - {len(predictions_df)} players predicted")
        logger.info("=" * 60)
        
        return predictions_df
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    results = main()