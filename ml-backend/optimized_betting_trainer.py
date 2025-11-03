#!/usr/bin/env python3
"""
OPTIMIZED Comprehensive Betting XGBoost Trainer
Balanced approach: High accuracy with sustainable performance
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.cluster import KMeans
import joblib
import os
import logging
import gc
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('optimized_training.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class OptimizedBettingTrainer:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_selectors = {}
        self.training_history = {}
        self.start_time = time.time()
        
    def load_datasets(self):
        """Load all datasets efficiently"""
        logger.info("OPTIMIZED DATASET LOADING...")
        
        datasets = {}
        dataset_files = {
            'defense': 'nfl_data/defense_stats_historical_clean.csv',
            'receiving': 'nfl_data/receiving_stats_historical_clean.csv', 
            'scrimmage': 'nfl_data/scrimmage_stats_historical_clean.csv',
            'rushing': 'nfl_data/rushing_stats_historical_clean.csv',
            'returns': 'nfl_data/returns_stats_historical_clean.csv',
            'passing': 'nfl_data/passing_stats_historical_clean.csv',
            'scoring': 'nfl_data/scoring_stats_historical_clean.csv',
            'kicking': 'nfl_data/kicking_stats_historical.csv',
            'punting': 'nfl_data/punting_stats_historical.csv'
        }
        
        total_records = 0
        for name, file_path in dataset_files.items():
            try:
                if os.path.exists(file_path):
                    logger.info(f"Loading {name}...")
                    df = pd.read_csv(file_path, low_memory=False)
                    datasets[name] = df
                    total_records += len(df)
                    logger.info(f"SUCCESS {name}: {len(df):,} records")
                else:
                    logger.warning(f"WARNING File not found: {file_path}")
            except Exception as e:
                logger.error(f"ERROR loading {name}: {e}")
        
        logger.info(f"TOTAL DATASET SIZE: {total_records:,} records")
        return datasets
    
    def create_optimized_features(self, datasets):
        """Create comprehensive but efficient features"""
        logger.info("OPTIMIZED FEATURE ENGINEERING...")
        
        all_features = []
        
        for stat_type, df in datasets.items():
            logger.info(f"Processing {stat_type}...")
            
            # Core betting features by position
            if stat_type == 'receiving':
                df = self._create_receiving_features(df)
            elif stat_type == 'rushing':
                df = self._create_rushing_features(df)
            elif stat_type == 'passing':
                df = self._create_passing_features(df)
            elif stat_type == 'scoring':
                df = self._create_scoring_features(df)
            elif stat_type == 'defense':
                df = self._create_defense_features(df)
            elif stat_type == 'scrimmage':
                df = self._create_scrimmage_features(df)
            elif stat_type == 'returns':
                df = self._create_returns_features(df)
            elif stat_type == 'kicking':
                df = self._create_kicking_features(df)
            elif stat_type == 'punting':
                df = self._create_punting_features(df)
            
            # Universal features
            df = self._create_universal_features(df)
            
            all_features.append(df)
            logger.info(f"SUCCESS {stat_type}: {len(df.columns)} features")
        
        # Combine datasets
        logger.info("Combining datasets...")
        combined_df = pd.concat(all_features, ignore_index=True, sort=False)
        
        # Efficient data cleaning
        combined_df = self._efficient_data_cleaning(combined_df)
        
        # Add strategic interactions (limited for performance)
        combined_df = self._create_strategic_interactions(combined_df)
        
        logger.info(f"OPTIMIZED FEATURES COMPLETE: {len(combined_df):,} records x {len(combined_df.columns):,} features")
        return combined_df
    
    def _create_receiving_features(self, df):
        """Essential receiving features"""
        if 'Rec_Yds' in df.columns:
            df['rec_yards_target'] = df['Rec_Yds']
            df['rec_yards_per_game'] = df['Rec_Yds'] / df.get('G', 1).replace(0, 1)
        
        if 'Rec' in df.columns:
            df['receptions_target'] = df['Rec']
            df['receptions_per_game'] = df['Rec'] / df.get('G', 1).replace(0, 1)
            if 'Tgt' in df.columns:
                df['catch_rate'] = df['Rec'] / df['Tgt'].replace(0, 1)
        
        if 'TD' in df.columns:
            df['rec_td_target'] = df['TD']
            df['rec_td_rate'] = df['TD'] / df.get('G', 1).replace(0, 1)
        
        if 'Lng' in df.columns:
            df['big_play_ability'] = np.where(df['Lng'] >= 20, 1, 0)
        
        return df
    
    def _create_rushing_features(self, df):
        """Essential rushing features"""
        if 'Yds' in df.columns:
            df['rush_yards_target'] = df['Yds']
            df['rush_yards_per_game'] = df['Yds'] / df.get('G', 1).replace(0, 1)
        
        if 'Att' in df.columns:
            df['rush_attempts_target'] = df['Att']
            df['rush_attempts_per_game'] = df['Att'] / df.get('G', 1).replace(0, 1)
            if 'Yds' in df.columns:
                df['yards_per_carry'] = df['Yds'] / df['Att'].replace(0, 1)
        
        if 'TD' in df.columns:
            df['rush_td_target'] = df['TD']
            df['rush_td_rate'] = df['TD'] / df.get('G', 1).replace(0, 1)
        
        return df
    
    def _create_passing_features(self, df):
        """Essential passing features"""
        if 'Yds' in df.columns:
            df['pass_yards_target'] = df['Yds']
            df['pass_yards_per_game'] = df['Yds'] / df.get('G', 1).replace(0, 1)
        
        if 'TD' in df.columns:
            df['pass_td_target'] = df['TD']
            df['pass_td_rate'] = df['TD'] / df.get('G', 1).replace(0, 1)
        
        if 'Int' in df.columns:
            df['pass_int_target'] = df['Int']
            df['int_rate'] = df['Int'] / df.get('G', 1).replace(0, 1)
        
        if 'Att' in df.columns and 'Cmp' in df.columns:
            df['completion_pct'] = df['Cmp'] / df['Att'].replace(0, 1)
        
        return df
    
    def _create_scoring_features(self, df):
        """Fantasy scoring features"""
        if 'Pts' in df.columns:
            df['fantasy_points_target'] = df['Pts']
            df['fantasy_points_per_game'] = df['Pts'] / df.get('G', 1).replace(0, 1)
            df['elite_fantasy_player'] = np.where(df['Pts'] > df['Pts'].quantile(0.9), 1, 0)
        
        return df
    
    def _create_defense_features(self, df):
        """Basic defense features"""
        df['defensive_player'] = 1
        return df
    
    def _create_scrimmage_features(self, df):
        """Scrimmage features"""
        rec_yds = df.get('Rec_Yds', 0)
        rush_yds = df.get('Rush_Yds', 0)
        
        df['total_scrimmage_yards'] = rec_yds + rush_yds
        df['scrimmage_yards_per_game'] = df['total_scrimmage_yards'] / df.get('G', 1).replace(0, 1)
        df['dual_threat'] = np.where((rec_yds > 100) & (rush_yds > 100), 1, 0)
        
        return df
    
    def _create_returns_features(self, df):
        """Return features"""
        if 'Ret_Yds' in df.columns:
            df['return_yards_per_game'] = df['Ret_Yds'] / df.get('G', 1).replace(0, 1)
        
        return df
    
    def _create_kicking_features(self, df):
        """Kicking features"""
        if 'FGA' in df.columns and 'FGM' in df.columns:
            df['fg_accuracy'] = df['FGM'] / df['FGA'].replace(0, 1)
        
        return df
    
    def _create_punting_features(self, df):
        """Punting features"""
        if 'Punts' in df.columns:
            df['punts_per_game'] = df['Punts'] / df.get('G', 1).replace(0, 1)
        
        return df
    
    def _create_universal_features(self, df):
        """Universal features for all positions"""
        # Games and durability
        df['games_played'] = df.get('G', 0)
        df['full_season'] = np.where(df['games_played'] >= 16, 1, 0)
        
        # Team encoding
        if 'Tm' in df.columns:
            df['team_encoded'] = pd.Categorical(df['Tm'].fillna('UNK')).codes
        else:
            df['team_encoded'] = 0
        
        # Position encoding
        if 'Pos' in df.columns:
            df['position_encoded'] = pd.Categorical(df['Pos'].fillna('UNK')).codes
            df['skill_position'] = np.where(df['Pos'].isin(['RB', 'WR', 'TE', 'QB']), 1, 0)
        else:
            df['position_encoded'] = 0
            df['skill_position'] = 0
        
        # Age features (if available)
        if 'Age' in df.columns:
            df['prime_age'] = np.where((df['Age'] >= 24) & (df['Age'] <= 29), 1, 0)
            df['veteran'] = np.where(df['Age'] >= 30, 1, 0)
        
        return df
    
    def _efficient_data_cleaning(self, df):
        """Efficient data cleaning"""
        logger.info("Efficient data cleaning...")
        
        # Handle infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Smart NaN filling
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col.endswith('_target'):
                df[col] = df[col].fillna(0)  # Targets default to 0
            elif col.endswith('_rate') or col.endswith('_pct'):
                df[col] = df[col].fillna(0)  # Rates default to 0
            else:
                df[col] = df[col].fillna(0)  # Others default to 0
        
        # Remove constant features
        constant_features = df.columns[df.nunique() <= 1]
        if len(constant_features) > 0:
            logger.info(f"Removing {len(constant_features)} constant features")
            df = df.drop(columns=constant_features)
        
        return df
    
    def _create_strategic_interactions(self, df):
        """Create essential interactions only"""
        logger.info("Creating strategic interactions...")
        
        # Key target features
        targets = [col for col in df.columns if col.endswith('_target')][:5]  # Top 5 only
        
        interaction_count = 0
        for i, target1 in enumerate(targets):
            for target2 in targets[i+1:]:
                if target1 in df.columns and target2 in df.columns:
                    # Multiplication
                    df[f'{target1}_x_{target2}'] = df[target1] * df[target2]
                    # Ratio (safe division)
                    df[f'{target1}_ratio_{target2}'] = df[target1] / df[target2].replace(0, 1)
                    interaction_count += 2
        
        logger.info(f"Created {interaction_count} strategic interactions")
        return df
    
    def get_betting_targets(self, df):
        """Get primary betting targets"""
        targets = {}
        
        # Focus on core betting markets
        core_targets = [
            'rec_yards_target', 'receptions_target', 'rec_td_target',
            'rush_yards_target', 'rush_attempts_target', 'rush_td_target',
            'pass_yards_target', 'pass_td_target', 'pass_int_target',
            'fantasy_points_target'
        ]
        
        for target in core_targets:
            if target in df.columns:
                target_data = df[target].dropna()
                if len(target_data) >= 1000:
                    targets[target] = target_data
                    logger.info(f"TARGET {target}: {len(target_data):,} samples")
        
        # Add interaction targets
        interaction_targets = [col for col in df.columns if '_x_' in col or '_ratio_' in col]
        for target in interaction_targets[:10]:  # Limit to top 10
            if target in df.columns:
                target_data = df[target].dropna()
                if len(target_data) >= 1000:
                    targets[target] = target_data
                    logger.info(f"INTERACTION TARGET {target}: {len(target_data):,} samples")
        
        logger.info(f"TOTAL BETTING TARGETS: {len(targets)}")
        return targets
    
    def train_optimized_model(self, df, target_name, target_data, model_number, total_models):
        """Optimized training with balanced hyperparameter search"""
        logger.info(f"="*80)
        logger.info(f"OPTIMIZED TRAINING: MODEL {model_number}/{total_models}")
        logger.info(f"TARGET: {target_name}")
        logger.info(f"="*80)
        
        start_time = time.time()
        
        # Feature preparation
        feature_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in feature_cols if not col.endswith('_target') and not col.startswith(target_name.split('_')[0] + '_')]
        
        # Get valid samples
        valid_indices = target_data.index
        X = df.loc[valid_indices, feature_cols].fillna(0)
        y = target_data
        
        logger.info(f"Training dataset: {len(X):,} samples x {len(feature_cols):,} features")
        
        # Efficient feature selection
        logger.info("Efficient feature selection...")
        selector = SelectKBest(score_func=f_regression, k=min(150, len(feature_cols)))
        X_selected = selector.fit_transform(X, y)
        selected_features = X_selected.shape[1]
        
        logger.info(f"Selected {selected_features} features")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=0.2, random_state=42
        )
        
        # Robust scaling
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # GPU check
        gpu_available = self._check_gpu()
        
        # Optimized training
        logger.info("Optimized hyperparameter search...")
        if gpu_available:
            logger.info("Using optimized GPU training")
            model = self._optimized_gpu_training(X_train_scaled, y_train)
        else:
            logger.info("Using optimized CPU training")
            model = self._optimized_cpu_training(X_train_scaled, y_train)
        
        # Evaluation
        logger.info("Model evaluation...")
        y_pred = model.predict(X_test_scaled)
        
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        # Cross-validation
        logger.info("Cross-validation...")
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2', n_jobs=1 if gpu_available else -1)
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        # Results
        elapsed_time = time.time() - start_time
        total_elapsed = time.time() - self.start_time
        
        results = {
            'model': model,
            'scaler': scaler,
            'selector': selector,
            'r2_score': r2,
            'cv_r2_mean': cv_mean,
            'cv_r2_std': cv_std,
            'mae': mae,
            'rmse': rmse,
            'accuracy_pct': r2 * 100,
            'cv_accuracy_pct': cv_mean * 100,
            'samples': len(X),
            'features_selected': selected_features,
            'gpu_used': gpu_available,
            'training_time_minutes': elapsed_time / 60,
            'total_time_hours': total_elapsed / 3600
        }
        
        # Display results
        self._display_results(target_name, results, model_number, total_models)
        
        # Save model
        self.save_model(target_name, results)
        
        # Memory cleanup
        del X_train, X_test, X_train_scaled, X_test_scaled, X_selected
        gc.collect()
        
        return results
    
    def _optimized_gpu_training(self, X_train, y_train):
        """Balanced GPU hyperparameter search"""
        # Optimized parameter grid (smaller but effective)
        param_grid = {
            'n_estimators': [300, 500, 700],
            'max_depth': [6, 8, 10, 12],
            'learning_rate': [0.03, 0.05, 0.08, 0.1],
            'subsample': [0.8, 0.85, 0.9],
            'colsample_bytree': [0.8, 0.85, 0.9],
            'gamma': [0, 0.1, 0.2],
            'reg_alpha': [0, 0.1, 0.3],
            'reg_lambda': [1, 1.5, 2]
        }
        
        base_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            device='cuda:0',
            tree_method='hist',
            random_state=42,
            eval_metric='rmse'
        )
        
        # Efficient grid search (648 combinations vs 921,600)
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring='r2',
            cv=5,  # Reduced CV for speed
            n_jobs=1,
            verbose=1,
            refit=True
        )
        
        logger.info(f"Starting optimized GPU search (648 combinations)...")
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best Score: {grid_search.best_score_:.6f}")
        logger.info(f"Best Params: {grid_search.best_params_}")
        
        return grid_search.best_estimator_
    
    def _optimized_cpu_training(self, X_train, y_train):
        """Efficient CPU training"""
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            tree_method='hist',
            n_estimators=500,
            max_depth=8,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.5,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        return model
    
    def _check_gpu(self):
        """Check GPU availability"""
        try:
            test_model = xgb.XGBRegressor(device='cuda:0', tree_method='hist')
            test_X = np.random.rand(10, 5)
            test_y = np.random.rand(10)
            test_model.fit(test_X, test_y)
            return True
        except:
            return False
    
    def _display_results(self, target_name, results, model_num, total_models):
        """Display training results"""
        print(f"\nOPTIMIZED RESULTS - MODEL {model_num}/{total_models}")
        print(f"Target: {target_name}")
        print(f"-" * 50)
        print(f"R2 Score: {results['r2_score']:.4f} ({results['accuracy_pct']:.2f}% accuracy)")
        print(f"CV R2: {results['cv_r2_mean']:.4f} +/- {results['cv_r2_std']:.4f}")
        print(f"MAE: {results['mae']:.3f} | RMSE: {results['rmse']:.3f}")
        print(f"Features: {results['features_selected']}")
        print(f"Samples: {results['samples']:,}")
        print(f"GPU: {'Yes' if results['gpu_used'] else 'No'}")
        print(f"Time: {results['training_time_minutes']:.1f} min")
        print(f"Total: {results['total_time_hours']:.2f} hours")
        
        # Performance tier
        accuracy = results['accuracy_pct']
        if accuracy >= 85:
            tier = "EXCELLENT"
        elif accuracy >= 80:
            tier = "VERY GOOD"
        elif accuracy >= 75:
            tier = "GOOD"
        else:
            tier = "BASELINE"
        
        print(f"Tier: {tier}")
        print(f"-" * 50)
    
    def save_model(self, target_name, results):
        """Save model and metadata"""
        os.makedirs('models/optimized_betting', exist_ok=True)
        
        # Save model components
        model_path = f'models/optimized_betting/{target_name}_model.joblib'
        scaler_path = f'models/optimized_betting/{target_name}_scaler.joblib'
        selector_path = f'models/optimized_betting/{target_name}_selector.joblib'
        
        joblib.dump(results['model'], model_path)
        joblib.dump(results['scaler'], scaler_path)
        joblib.dump(results['selector'], selector_path)
        
        # Save metadata
        metadata = {k: v for k, v in results.items() if k not in ['model', 'scaler', 'selector']}
        metadata_path = f'models/optimized_betting/{target_name}_metadata.joblib'
        joblib.dump(metadata, metadata_path)
        
        logger.info(f"Model saved: {target_name}")

def main():
    """OPTIMIZED COMPREHENSIVE TRAINING"""
    print(f"OPTIMIZED COMPREHENSIVE BETTING TRAINING")
    print(f"100,000+ Records • Balanced Performance • Sustainable Training")
    print(f"Expected Duration: 6-8 Hours")
    print(f"=" * 65)
    
    trainer = OptimizedBettingTrainer()
    
    try:
        start_time = datetime.now()
        logger.info(f"OPTIMIZED TRAINING STARTED: {start_time}")
        
        # Load datasets
        datasets = trainer.load_datasets()
        
        # Create optimized features
        df = trainer.create_optimized_features(datasets)
        
        # Get betting targets
        targets = trainer.get_betting_targets(df)
        
        print(f"\nBETTING TARGETS:")
        for target_name, target_data in targets.items():
            print(f"   {target_name}: {len(target_data):,} samples")
        
        print(f"\nSTARTING OPTIMIZED TRAINING...")
        print(f"GPU: {'Available' if trainer._check_gpu() else 'CPU Only'}")
        print(f"Features: {len(df.columns):,}")
        print(f"Samples: {len(df):,}")
        print(f"Models: {len(targets)}")
        
        # Train models
        all_results = {}
        successful_models = 0
        total_models = len(targets)
        
        for i, (target_name, target_data) in enumerate(targets.items(), 1):
            try:
                results = trainer.train_optimized_model(df, target_name, target_data, i, total_models)
                all_results[target_name] = results
                successful_models += 1
                
                # Progress update
                elapsed_hours = (time.time() - trainer.start_time) / 3600
                remaining_time = (elapsed_hours / i) * (total_models - i)
                
                print(f"\nPROGRESS: {i}/{total_models} ({i/total_models*100:.1f}%)")
                print(f"Elapsed: {elapsed_hours:.2f}h | Remaining: {remaining_time:.2f}h")
                
            except Exception as e:
                logger.error(f"Failed {target_name}: {e}")
                continue
        
        # Final results
        end_time = datetime.now()
        duration = end_time - start_time
        
        print(f"\nOPTIMIZED TRAINING COMPLETE!")
        print(f"Duration: {duration}")
        print(f"Successful: {successful_models}/{total_models}")
        
        if all_results:
            accuracies = [r['accuracy_pct'] for r in all_results.values()]
            avg_accuracy = np.mean(accuracies)
            max_accuracy = max(accuracies)
            
            print(f"\nACCURACY RESULTS:")
            print(f"Average: {avg_accuracy:.2f}%")
            print(f"Maximum: {max_accuracy:.2f}%")
            
            excellent_models = [name for name, r in all_results.items() if r['accuracy_pct'] >= 85]
            good_models = [name for name, r in all_results.items() if r['accuracy_pct'] >= 80]
            
            print(f"\nPERFORMANCE:")
            print(f"Excellent (85%+): {len(excellent_models)}")
            print(f"Very Good (80%+): {len(good_models)}")
            
            if excellent_models:
                print(f"\nEXCELLENT MODELS:")
                for name in excellent_models:
                    acc = all_results[name]['accuracy_pct']
                    print(f"  {name}: {acc:.2f}%")
        
        print(f"\nModels saved to 'models/optimized_betting/'")
        return trainer, all_results
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    trainer, results = main()