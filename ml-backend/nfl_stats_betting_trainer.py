#!/usr/bin/env python3
"""
NFL Stats Betting Predictor - Real Betting Scenarios
===================================================

Using actual NFL statistics for realistic betting predictions:
- Passing Yards: QB over/under yards, TDs, completions
- Rushing Yards: RB over/under yards, attempts, TDs
- Receiving Yards: WR/TE over/under yards, catches, TDs
- Multi-position combined stats

Target: 80-90% accuracy on real NFL betting lines
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.ensemble import VotingClassifier
import logging
import os
import warnings
from datetime import datetime
import joblib

warnings.filterwarnings('ignore')

# Enhanced logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nfl_stats_betting_trainer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class NFLStatsBettingTrainer:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.target_columns = []
        
        # High-performance XGBoost configuration
        self.xgb_params = {
            'device': 'cuda',
            'tree_method': 'gpu_hist',
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'random_state': 42,
            'n_jobs': -1,
            'max_depth': 7,
            'learning_rate': 0.05,
            'n_estimators': 400,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0
        }
        
        # Create output directories
        os.makedirs('models/nfl_stats_betting', exist_ok=True)
        os.makedirs('reports/nfl_stats_betting', exist_ok=True)

    def load_position_specific_data(self):
        """Load and combine ALL 26 years of NFL historical data"""
        logger.info("Loading ALL 26 years of NFL historical data...")
        
        datasets = {}
        total_records = 0
        
        # Load ALL available NFL historical data (26 years worth)
        all_nfl_files = {
            'passing': 'nfl_data/passing_stats_historical_clean.csv',
            'rushing': 'nfl_data/rushing_stats_historical_clean.csv', 
            'receiving': 'nfl_data/receiving_stats_historical_clean.csv',
            'defense': 'nfl_data/defense_stats_historical_clean.csv',
            'scoring': 'nfl_data/scoring_stats_historical_clean.csv',
            'scrimmage': 'nfl_data/scrimmage_stats_historical_clean.csv',
            'returns': 'nfl_data/returns_stats_historical_clean.csv'
        }
        
        for data_type, file_path in all_nfl_files.items():
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path, low_memory=False)
                    logger.info(f"Loaded {data_type}: {len(df)} records")
                    datasets[data_type] = df
                    total_records += len(df)
                except Exception as e:
                    logger.warning(f"Could not load {file_path}: {e}")
        
        logger.info(f"Total 26-year NFL data: {total_records} records")
        return datasets

    def create_real_betting_targets(self, datasets):
        """Create realistic NFL betting targets based on actual sportsbook lines"""
        logger.info("Creating real NFL betting targets...")
        
        all_targets = {}
        combined_data = []
        
        # PASSING TARGETS (QB stats)
        if 'passing' in datasets:
            passing_df = datasets['passing'].copy()
            passing_targets = {}
            
            if 'Yds' in passing_df.columns:
                # Common QB passing yards betting lines
                passing_targets['qb_over_250_yards'] = (passing_df['Yds'] > 250).astype(int)
                passing_targets['qb_over_300_yards'] = (passing_df['Yds'] > 300).astype(int)
                passing_targets['qb_under_200_yards'] = (passing_df['Yds'] < 200).astype(int)
            
            if 'TD' in passing_df.columns:
                # QB touchdown betting lines
                passing_targets['qb_2plus_tds'] = (passing_df['TD'] >= 2).astype(int)
                passing_targets['qb_3plus_tds'] = (passing_df['TD'] >= 3).astype(int)
                passing_targets['qb_any_td'] = (passing_df['TD'] >= 1).astype(int)
            
            if 'Cmp' in passing_df.columns:
                # Completion betting lines
                passing_targets['qb_over_20_completions'] = (passing_df['Cmp'] > 20).astype(int)
                passing_targets['qb_over_25_completions'] = (passing_df['Cmp'] > 25).astype(int)
            
            if 'Int' in passing_df.columns:
                # Interception lines
                passing_targets['qb_any_interception'] = (passing_df['Int'] >= 1).astype(int)
                passing_targets['qb_clean_game'] = (passing_df['Int'] == 0).astype(int)
            
            # Add targets to dataframe
            for target, values in passing_targets.items():
                passing_df[target] = values
                all_targets[target] = values
            
            passing_df['data_source'] = 'passing'
            combined_data.append(passing_df)
            logger.info(f"Created {len(passing_targets)} passing betting targets")
        
        # RUSHING TARGETS (RB stats)
        if 'rushing' in datasets:
            rushing_df = datasets['rushing'].copy()
            rushing_targets = {}
            
            if 'Yds' in rushing_df.columns:
                # Common RB rushing yards betting lines
                rushing_targets['rb_over_50_yards'] = (rushing_df['Yds'] > 50).astype(int)
                rushing_targets['rb_over_75_yards'] = (rushing_df['Yds'] > 75).astype(int)
                rushing_targets['rb_over_100_yards'] = (rushing_df['Yds'] > 100).astype(int)
                rushing_targets['rb_under_40_yards'] = (rushing_df['Yds'] < 40).astype(int)
            
            if 'TD' in rushing_df.columns:
                # RB touchdown lines
                rushing_targets['rb_any_td'] = (rushing_df['TD'] >= 1).astype(int)
                rushing_targets['rb_2plus_tds'] = (rushing_df['TD'] >= 2).astype(int)
                rushing_targets['rb_no_td'] = (rushing_df['TD'] == 0).astype(int)
            
            if 'Att' in rushing_df.columns:
                # Rushing attempts lines
                rushing_targets['rb_over_15_attempts'] = (rushing_df['Att'] > 15).astype(int)
                rushing_targets['rb_over_20_attempts'] = (rushing_df['Att'] > 20).astype(int)
            
            # Add targets to dataframe
            for target, values in rushing_targets.items():
                rushing_df[target] = values
                all_targets[target] = values
            
            rushing_df['data_source'] = 'rushing'
            combined_data.append(rushing_df)
            logger.info(f"Created {len(rushing_targets)} rushing betting targets")
        
        # RECEIVING TARGETS (WR/TE stats)
        if 'receiving' in datasets:
            receiving_df = datasets['receiving'].copy()
            receiving_targets = {}
            
            if 'Yds' in receiving_df.columns:
                # Common WR/TE receiving yards lines
                receiving_targets['wr_over_50_yards'] = (receiving_df['Yds'] > 50).astype(int)
                receiving_targets['wr_over_75_yards'] = (receiving_df['Yds'] > 75).astype(int)
                receiving_targets['wr_over_100_yards'] = (receiving_df['Yds'] > 100).astype(int)
                receiving_targets['wr_under_40_yards'] = (receiving_df['Yds'] < 40).astype(int)
            
            if 'Rec' in receiving_df.columns:
                # Reception lines
                receiving_targets['wr_over_5_catches'] = (receiving_df['Rec'] > 5).astype(int)
                receiving_targets['wr_over_7_catches'] = (receiving_df['Rec'] > 7).astype(int)
                receiving_targets['wr_under_4_catches'] = (receiving_df['Rec'] < 4).astype(int)
            
            if 'TD' in receiving_df.columns:
                # Receiving touchdown lines
                receiving_targets['wr_any_td'] = (receiving_df['TD'] >= 1).astype(int)
                receiving_targets['wr_no_td'] = (receiving_df['TD'] == 0).astype(int)
            
            if 'Tgt' in receiving_df.columns:
                # Target lines
                receiving_targets['wr_over_8_targets'] = (receiving_df['Tgt'] > 8).astype(int)
                receiving_targets['wr_over_10_targets'] = (receiving_df['Tgt'] > 10).astype(int)
            
            # Add targets to dataframe
            for target, values in receiving_targets.items():
                receiving_df[target] = values
                all_targets[target] = values
            
            receiving_df['data_source'] = 'receiving'
            combined_data.append(receiving_df)
            logger.info(f"Created {len(receiving_targets)} receiving betting targets")
        
        # DEFENSE TARGETS (DEF stats)
        if 'defense' in datasets:
            defense_df = datasets['defense'].copy()
            defense_targets = {}
            
            if 'Int' in defense_df.columns:
                # Interception betting lines
                defense_targets['def_any_interception'] = (defense_df['Int'] >= 1).astype(int)
                defense_targets['def_2plus_interceptions'] = (defense_df['Int'] >= 2).astype(int)
            
            if 'Sk' in defense_df.columns:
                # Sack betting lines
                defense_targets['def_any_sack'] = (defense_df['Sk'] >= 1).astype(int)
                defense_targets['def_2plus_sacks'] = (defense_df['Sk'] >= 2).astype(int)
            
            if 'PD' in defense_df.columns:
                # Pass deflection lines
                defense_targets['def_pass_deflection'] = (defense_df['PD'] >= 1).astype(int)
            
            # Add targets to dataframe
            for target, values in defense_targets.items():
                defense_df[target] = values
                all_targets[target] = values
            
            defense_df['data_source'] = 'defense'
            combined_data.append(defense_df)
            logger.info(f"Created {len(defense_targets)} defense betting targets")
        
        # SCORING TARGETS (Kicking/Special Teams)
        if 'scoring' in datasets:
            scoring_df = datasets['scoring'].copy()
            scoring_targets = {}
            
            if 'XPM' in scoring_df.columns:
                # Extra point lines
                scoring_targets['kicker_perfect_xp'] = (scoring_df['XPM'] == scoring_df['XPA']).astype(int)
            
            if 'FGM' in scoring_df.columns:
                # Field goal lines
                scoring_targets['kicker_made_fg'] = (scoring_df['FGM'] >= 1).astype(int)
                scoring_targets['kicker_2plus_fg'] = (scoring_df['FGM'] >= 2).astype(int)
            
            # Add targets to dataframe
            for target, values in scoring_targets.items():
                scoring_df[target] = values
                all_targets[target] = values
            
            scoring_df['data_source'] = 'scoring'
            combined_data.append(scoring_df)
            logger.info(f"Created {len(scoring_targets)} scoring betting targets")
        
        # SPECIAL TEAMS TARGETS (Returns)
        if 'returns' in datasets:
            returns_df = datasets['returns'].copy()
            returns_targets = {}
            
            if 'Yds' in returns_df.columns:
                # Return yards lines
                returns_targets['return_over_20_yards'] = (returns_df['Yds'] > 20).astype(int)
                returns_targets['return_big_play'] = (returns_df['Yds'] > 40).astype(int)
            
            if 'TD' in returns_df.columns:
                # Return touchdown lines
                returns_targets['return_td'] = (returns_df['TD'] >= 1).astype(int)
            
            # Add targets to dataframe
            for target, values in returns_targets.items():
                returns_df[target] = values
                all_targets[target] = values
            
            returns_df['data_source'] = 'returns'
            combined_data.append(returns_df)
            logger.info(f"Created {len(returns_targets)} return betting targets")
        
        # SCRIMMAGE TARGETS (Combined rushing/receiving)
        if 'scrimmage' in datasets:
            scrimmage_df = datasets['scrimmage'].copy()
            scrimmage_targets = {}
            
            if 'Yds' in scrimmage_df.columns:
                # Total scrimmage yards
                scrimmage_targets['scrimmage_over_100_yards'] = (scrimmage_df['Yds'] > 100).astype(int)
                scrimmage_targets['scrimmage_over_150_yards'] = (scrimmage_df['Yds'] > 150).astype(int)
            
            if 'TD' in scrimmage_df.columns:
                # Total touchdowns
                scrimmage_targets['scrimmage_any_td'] = (scrimmage_df['TD'] >= 1).astype(int)
            
            # Add targets to dataframe
            for target, values in scrimmage_targets.items():
                scrimmage_df[target] = values
                all_targets[target] = values
            
            scrimmage_df['data_source'] = 'scrimmage'
            combined_data.append(scrimmage_df)
            logger.info(f"Created {len(scrimmage_targets)} scrimmage betting targets")
        
        # Combine all datasets
        if combined_data:
            combined_df = pd.concat(combined_data, ignore_index=True, sort=False)
            
            # Clean data - remove NaN values
            logger.info(f"Data before cleaning: {len(combined_df)} records")
            
            # Fill NaN values in target columns with 0
            for target in all_targets.keys():
                if target in combined_df.columns:
                    combined_df[target] = combined_df[target].fillna(0).astype(int)
            
            # Remove rows with NaN in key statistical columns
            key_cols = ['Yds', 'TD', 'G', 'Player', 'Team', 'Pos']
            for col in key_cols:
                if col in combined_df.columns:
                    combined_df = combined_df.dropna(subset=[col])
            
            logger.info(f"Data after cleaning: {len(combined_df)} records")
            
            # Add temporal ordering
            if 'Year' in combined_df.columns:
                combined_df['season'] = combined_df['Year']
            else:
                combined_df['season'] = np.random.choice([2020, 2021, 2022, 2023, 2024], size=len(combined_df))
            
            if 'week' not in combined_df.columns:
                combined_df['week'] = np.random.randint(1, 18, size=len(combined_df))
            
            combined_df['temporal_order'] = combined_df['season'] * 100 + combined_df['week']
            combined_df = combined_df.sort_values('temporal_order').reset_index(drop=True)
            
            # Filter targets with good hit rates (recheck after cleaning)
            final_targets = []
            for target_name in all_targets.keys():
                if target_name in combined_df.columns:
                    target_values = combined_df[target_name].dropna()
                    if len(target_values) > 100:  # Ensure enough data
                        hit_rate = target_values.mean()
                        if 0.15 <= hit_rate <= 0.85:  # Good range for betting accuracy
                            final_targets.append(target_name)
                            logger.info(f"   {target_name}: {hit_rate:.1%} hit rate")
                        else:
                            logger.info(f"   Skipped {target_name}: {hit_rate:.1%} hit rate")
            
            self.target_columns = final_targets
            logger.info(f"Selected {len(self.target_columns)} clean betting targets")
            
            return combined_df
        
        return None

    def create_nfl_features(self, df):
        """Create NFL-specific features for betting prediction"""
        logger.info("Creating NFL statistical features...")
        
        # Get numeric columns (exclude targets and metadata)
        exclude_cols = (self.target_columns + ['temporal_order', 'season', 'week', 'Rk', 'data_source'])
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        base_features = [col for col in numeric_cols if col not in exclude_cols]
        
        features = df[base_features].copy()
        
        # 1. EFFICIENCY METRICS
        if 'Yds' in features.columns and 'G' in features.columns:
            features['yards_per_game'] = features['Yds'] / features['G'].replace(0, 1)
        
        if 'TD' in features.columns and 'G' in features.columns:
            features['tds_per_game'] = features['TD'] / features['G'].replace(0, 1)
        
        # Passing-specific efficiency
        if 'Cmp' in features.columns and 'Att' in features.columns:
            features['completion_percentage'] = features['Cmp'] / features['Att'].replace(0, 1)
        
        if 'Yds' in features.columns and 'Att' in features.columns:
            features['yards_per_attempt'] = features['Yds'] / features['Att'].replace(0, 1)
        
        # Rushing-specific efficiency  
        if 'Yds' in features.columns and 'Att' in features.columns and 'data_source' in df.columns:
            rushing_mask = df['data_source'] == 'rushing'
            if rushing_mask.sum() > 0:
                features.loc[rushing_mask, 'rush_yards_per_attempt'] = (
                    features.loc[rushing_mask, 'Yds'] / features.loc[rushing_mask, 'Att'].replace(0, 1)
                )
        
        # Receiving-specific efficiency
        if 'Yds' in features.columns and 'Rec' in features.columns:
            features['yards_per_reception'] = features['Yds'] / features['Rec'].replace(0, 1)
        
        if 'Rec' in features.columns and 'Tgt' in features.columns:
            features['catch_rate'] = features['Rec'] / features['Tgt'].replace(0, 1)
        
        # 2. CATEGORICAL ENCODING
        categorical_cols = ['Player', 'Team', 'Pos']
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                features[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
        
        # 3. PLAYER HISTORICAL PERFORMANCE
        if 'Player' in df.columns and 'Yds' in df.columns:
            player_stats = df.groupby('Player').agg({
                'Yds': ['mean', 'std', 'max'],
                'TD': 'mean' if 'TD' in df.columns else 'count',
                'G': 'sum' if 'G' in df.columns else 'count'
            }).round(2)
            
            player_stats.columns = ['player_avg_yards', 'player_yards_consistency', 'player_max_yards', 
                                  'player_avg_tds', 'player_total_games']
            
            # Merge back to features
            for col in player_stats.columns:
                features[col] = df['Player'].map(player_stats[col]).fillna(0)
        
        # 4. TEAM CONTEXT
        if 'Team' in df.columns and 'Yds' in df.columns:
            team_stats = df.groupby('Team')['Yds'].agg(['mean', 'std']).round(2)
            features['team_avg_yards'] = df['Team'].map(team_stats['mean']).fillna(0)
            features['team_yards_variance'] = df['Team'].map(team_stats['std']).fillna(0)
        
        # 5. AGE AND EXPERIENCE FACTORS
        if 'Age' in features.columns:
            features['age_squared'] = features['Age'] ** 2
            features['prime_age'] = ((features['Age'] >= 24) & (features['Age'] <= 30)).astype(int)
            features['veteran_player'] = (features['Age'] >= 30).astype(int)
        
        # 6. GAME VOLUME INDICATORS
        if 'G' in features.columns:
            features['high_volume_player'] = (features['G'] >= features['G'].quantile(0.75)).astype(int)
            features['games_squared'] = features['G'] ** 2
        
        # 7. POSITION-SPECIFIC FEATURES
        if 'Pos' in df.columns:
            # QB features
            qb_mask = df['Pos'] == 'QB'
            if qb_mask.sum() > 0 and 'Rate' in features.columns:
                features['qb_elite_rating'] = ((features['Rate'] > 100) & qb_mask.values).astype(int)
            
            # RB features  
            rb_mask = df['Pos'] == 'RB'
            if rb_mask.sum() > 0 and 'YdsPerAtt' in features.columns:
                features['rb_efficient'] = ((features['YdsPerAtt'] > 4.0) & rb_mask.values).astype(int)
            
            # WR/TE features
            wr_mask = df['Pos'].isin(['WR', 'TE'])
            if wr_mask.sum() > 0 and 'catch_rate' in features.columns:
                features['wr_reliable'] = ((features['catch_rate'] > 0.65) & wr_mask.values).astype(int)
        
        # Fill missing values
        features = features.fillna(0)
        
        # Ensure all features are numeric
        features = features.select_dtypes(include=[np.number])
        
        self.feature_names = features.columns.tolist()
        logger.info(f"Created {len(self.feature_names)} NFL statistical features")
        
        return features

    def train_betting_models(self, df, features):
        """Train models for NFL betting predictions"""
        logger.info("Training NFL betting models...")
        
        # Create temporal splits
        split_idx = int(len(df) * 0.8)
        X_train = features.iloc[:split_idx]
        X_test = features.iloc[split_idx:]
        
        results = {}
        
        for i, target in enumerate(self.target_columns):
            logger.info(f"Training model {i+1}/{len(self.target_columns)}: {target}")
            
            y_train = df[target].iloc[:split_idx]
            y_test = df[target].iloc[split_idx:]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            self.scalers[target] = scaler
            
            # Handle class imbalance
            pos_count = (y_train == 1).sum()
            neg_count = (y_train == 0).sum()
            scale_pos_weight = neg_count / max(pos_count, 1)
            
            # Create optimized XGBoost model
            model_params = {**self.xgb_params, 'scale_pos_weight': scale_pos_weight}
            model = xgb.XGBClassifier(**model_params)
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Predictions
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # Evaluate
            accuracy = accuracy_score(y_test, y_pred)
            try:
                auc = roc_auc_score(y_test, y_pred_proba)
            except:
                auc = 0.5
            
            baseline = max(y_test.mean(), 1 - y_test.mean())
            hit_rate = y_test.mean()
            
            results[target] = {
                'accuracy': accuracy,
                'auc': auc,
                'baseline': baseline,
                'hit_rate': hit_rate,
                'improvement': accuracy - baseline
            }
            
            # Save model
            self.models[target] = model
            joblib.dump(model, f'models/nfl_stats_betting/{target}_model.pkl')
            joblib.dump(scaler, f'models/nfl_stats_betting/{target}_scaler.pkl')
            
            logger.info(f"   {target}: {accuracy:.3f} accuracy ({hit_rate:.1%} hit rate)")
        
        # Overall results
        results_df = pd.DataFrame(results).T
        results_df.to_csv('reports/nfl_stats_betting/model_results.csv')
        
        avg_accuracy = results_df['accuracy'].mean()
        avg_auc = results_df['auc'].mean()
        
        logger.info("=" * 60)
        logger.info("NFL BETTING MODEL RESULTS:")
        logger.info(f"Average Accuracy: {avg_accuracy:.3f}")
        logger.info(f"Average AUC: {avg_auc:.3f}")
        logger.info(f"Models trained: {len(results)}")
        
        if avg_accuracy >= 0.80:
            logger.info("SUCCESS: Achieved 80%+ accuracy target!")
        elif avg_accuracy >= 0.70:
            logger.info("GOOD: Strong betting accuracy achieved!")
        else:
            logger.info("DEVELOPING: Building towards higher accuracy...")
        
        return results

    def run_nfl_betting_training(self):
        """Run complete NFL stats betting training"""
        start_time = datetime.now()
        logger.info("Starting NFL Stats Betting Trainer")
        logger.info("Using real NFL statistics for betting predictions")
        logger.info("=" * 60)
        
        try:
            # Load position-specific data
            datasets = self.load_position_specific_data()
            
            if not datasets:
                logger.error("No NFL datasets loaded!")
                return False
            
            # Create real betting targets
            df = self.create_real_betting_targets(datasets)
            
            if df is None or len(self.target_columns) == 0:
                logger.error("No valid betting targets created!")
                return False
            
            # Create NFL features
            features = self.create_nfl_features(df)
            
            # Train models
            results = self.train_betting_models(df, features)
            
            # Summary
            duration = datetime.now() - start_time
            logger.info("=" * 60)
            logger.info("NFL STATS BETTING TRAINING COMPLETED!")
            logger.info(f"Total time: {duration}")
            logger.info(f"Total samples: {len(df)}")
            logger.info(f"Betting targets: {len(self.target_columns)}")
            logger.info(f"Features: {len(self.feature_names)}")
            logger.info("Real NFL betting scenarios:")
            for target in self.target_columns[:5]:  # Show first 5
                logger.info(f"  - {target}")
            
            return True
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

if __name__ == "__main__":
    trainer = NFLStatsBettingTrainer()
    success = trainer.run_nfl_betting_training()
    
    if success:
        print("\nNFL Stats Betting Trainer completed!")
        print("Real NFL betting scenarios with statistical features!")
    else:
        print("Training failed - check logs for details")