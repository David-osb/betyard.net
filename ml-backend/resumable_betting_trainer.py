"""
RESUMABLE Comprehensive Betting XGBoost Trainer
- Resume from interruptions
- Save progress after each model
- Handle crashes gracefully
- 100k+ records, GPU acceleration
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression
import joblib
import json
import logging
import os
from datetime import datetime
import pickle
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ResumableBettingTrainer:
    def __init__(self):
        self.models = {}
        self.results = {}
        self.checkpoint_dir = 'checkpoints'
        self.models_dir = 'models/betting'
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        
    def save_checkpoint(self, data, targets, completed_models, current_model_idx):
        """Save training checkpoint"""
        checkpoint = {
            'data': data,
            'targets': targets,
            'completed_models': completed_models,
            'current_model_idx': current_model_idx,
            'timestamp': datetime.now().isoformat(),
            'total_models': len(targets)
        }
        
        checkpoint_path = os.path.join(self.checkpoint_dir, 'training_checkpoint.pkl')
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        logger.info(f"💾 Checkpoint saved: Model {current_model_idx + 1}/{len(targets)}")
        
    def load_checkpoint(self):
        """Load training checkpoint if exists"""
        checkpoint_path = os.path.join(self.checkpoint_dir, 'training_checkpoint.pkl')
        if os.path.exists(checkpoint_path):
            try:
                with open(checkpoint_path, 'rb') as f:
                    checkpoint = pickle.load(f)
                logger.info(f"📂 Checkpoint loaded: Model {checkpoint['current_model_idx'] + 1}/{checkpoint['total_models']}")
                return checkpoint
            except Exception as e:
                logger.warning(f"⚠️ Could not load checkpoint: {e}")
        return None
        
    def save_model_result(self, target_name, model, scaler, selector, metrics):
        """Save individual model and results"""
        # Save model
        model_path = os.path.join(self.models_dir, f'{target_name}_model.joblib')
        joblib.dump(model, model_path)
        
        # Save scaler
        scaler_path = os.path.join(self.models_dir, f'{target_name}_scaler.joblib')
        joblib.dump(scaler, scaler_path)
        
        # Save selector
        selector_path = os.path.join(self.models_dir, f'{target_name}_selector.joblib')
        joblib.dump(selector, selector_path)
        
        # Save metrics
        metadata = {
            'target': target_name,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
            'gpu_used': True
        }
        
        metadata_path = os.path.join(self.models_dir, f'{target_name}_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"💾 Saved model: {target_name} ({metrics['accuracy']:.1f}%)")
        
    def load_all_datasets(self):
        """Load the complete 100k+ dataset"""
        logger.info("📊 Loading COMPLETE dataset (100k+ records)...")
        
        datasets = {}
        data_files = {
            'defense': 'nfl_data/defense_stats_historical_clean.csv',
            'receiving': 'nfl_data/receiving_stats_historical_clean.csv', 
            'scrimmage': 'nfl_data/scrimmage_stats_historical_clean.csv',
            'rushing': 'nfl_data/rushing_stats_historical_clean.csv',
            'returns': 'nfl_data/returns_stats_historical_clean.csv',
            'passing': 'nfl_data/passing_stats_historical_clean.csv',
            'scoring': 'nfl_data/scoring_stats_historical_clean.csv',
            'kicking': 'nfl_data/kicking_stats_historical.csv',
            'punting': 'nfl_data/punting_stats_historical_cleaned.csv'
        }
        
        total_records = 0
        for stat_type, filepath in data_files.items():
            if os.path.exists(filepath):
                try:
                    df = pd.read_csv(filepath, low_memory=False, on_bad_lines='skip')
                    df = self._clean_dataset(df, stat_type)
                    
                    if len(df) > 0:
                        datasets[stat_type] = df
                        total_records += len(df)
                        logger.info(f"✅ {stat_type}: {len(df):,} records")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Could not load {stat_type}: {e}")
        
        logger.info(f"🎯 TOTAL RECORDS: {total_records:,}")
        return datasets
    
    def _clean_dataset(self, df, stat_type):
        """Clean individual dataset"""
        # Remove header rows
        if 'Player' in df.columns:
            df['Player'] = df['Player'].astype(str)
            df = df[~df['Player'].str.contains('Player|Rk|League Average', na=False)]
            df = df[df['Player'].notna()]
            df = df[df['Player'].str.len() > 0]
            df = df[df['Player'] != 'nan']
        
        # Convert numeric columns
        for col in df.columns:
            if col not in ['Player', 'Team', 'Pos', 'Awards', 'stat_type']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Fill NaN with 0
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        # Add identifier
        df['stat_type'] = stat_type
        
        return df
    
    def create_comprehensive_features(self, datasets):
        """Create comprehensive betting features"""
        logger.info("🔧 Creating comprehensive features...")
        
        all_features = []
        for stat_type, df in datasets.items():
            df = self._create_position_features(df, stat_type)
            df = self._create_betting_targets(df, stat_type)
            df = self._create_efficiency_features(df)
            all_features.append(df)
        
        # Combine datasets
        combined_df = pd.concat(all_features, ignore_index=True, sort=False)
        combined_df = combined_df.fillna(0)
        
        # Remove constant features
        numeric_cols = combined_df.select_dtypes(include=[np.number]).columns
        constant_features = [col for col in numeric_cols if combined_df[col].nunique() <= 1]
        if constant_features:
            combined_df = combined_df.drop(columns=constant_features)
            logger.info(f"🧹 Removed {len(constant_features)} constant features")
        
        # Create strategic interactions
        combined_df = self._create_strategic_interactions(combined_df)
        
        logger.info(f"✅ COMPREHENSIVE FEATURES: {len(combined_df):,} records × {len(combined_df.columns):,} features")
        return combined_df
    
    def _create_position_features(self, df, stat_type):
        """Create position-specific features"""
        if stat_type == 'receiving' and 'Rec' in df.columns and 'Rec_Yds' in df.columns:
            df['receptions_target'] = df['Rec']
            df['rec_yards_target'] = df['Rec_Yds']
            df['rec_td_target'] = df.get('TD', 0)
            df['rec_per_game'] = df['Rec'] / df.get('G', 1)
            df['rec_yards_per_game'] = df['Rec_Yds'] / df.get('G', 1)
            
        elif stat_type == 'rushing' and 'Att' in df.columns and 'Yds' in df.columns:
            df['rush_attempts_target'] = df['Att']
            df['rush_yards_target'] = df['Yds']
            df['rush_td_target'] = df.get('TD', 0)
            df['rush_per_game'] = df['Att'] / df.get('G', 1)
            df['rush_yards_per_game'] = df['Yds'] / df.get('G', 1)
            
        elif stat_type == 'passing' and 'Att' in df.columns and 'Yds' in df.columns:
            df['pass_attempts_target'] = df['Att']
            df['pass_yards_target'] = df['Yds']
            df['pass_td_target'] = df.get('TD', 0)
            df['pass_int_target'] = df.get('Int', 0)
            df['pass_per_game'] = df['Att'] / df.get('G', 1)
            df['pass_yards_per_game'] = df['Yds'] / df.get('G', 1)
        
        return df
    
    def _create_betting_targets(self, df, stat_type):
        """Create betting-specific targets"""
        # Fantasy points calculation
        fantasy_points = (
            df.get('receptions_target', 0) * 0.5 +
            df.get('rec_yards_target', 0) * 0.1 +
            df.get('rec_td_target', 0) * 6 +
            df.get('rush_yards_target', 0) * 0.1 +
            df.get('rush_td_target', 0) * 6 +
            df.get('pass_yards_target', 0) * 0.04 +
            df.get('pass_td_target', 0) * 4 -
            df.get('pass_int_target', 0) * 2
        )
        df['fantasy_points_target'] = fantasy_points
        
        return df
    
    def _create_efficiency_features(self, df):
        """Create efficiency and ratio features"""
        # Age features
        if 'Age' in df.columns:
            df['age_prime'] = ((df['Age'] >= 25) & (df['Age'] <= 29)).astype(int)
            df['age_veteran'] = (df['Age'] >= 32).astype(int)
            df['age_squared'] = df['Age'] ** 2
        
        # Games played features
        if 'G' in df.columns:
            df['games_pct'] = df['G'] / 17.0
            df['durability'] = (df['G'] >= 14).astype(int)
        
        # Position encoding
        if 'Pos' in df.columns:
            position_map = {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 4, 'K': 5, 'P': 6, 'DEF': 7}
            df['pos_encoded'] = df['Pos'].map(position_map).fillna(0)
        
        return df
    
    def _create_strategic_interactions(self, df):
        """Create strategic interaction features"""
        logger.info("🧠 Creating strategic interactions...")
        
        # Key interaction pairs for betting
        interaction_pairs = [
            ('receptions_target', 'rec_yards_target'),
            ('rush_attempts_target', 'rush_yards_target'),
            ('pass_attempts_target', 'pass_yards_target'),
            ('Age', 'G'),
            ('games_pct', 'fantasy_points_target')
        ]
        
        interactions_created = 0
        for col1, col2 in interaction_pairs:
            if col1 in df.columns and col2 in df.columns:
                # Multiplication interaction
                df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
                
                # Ratio interaction
                df[f'{col1}_ratio_{col2}'] = df[col1] / (df[col2] + 0.001)
                
                interactions_created += 2
        
        logger.info(f"✅ Created {interactions_created} strategic interactions")
        return df
    
    def create_betting_targets(self, df):
        """Extract all betting targets"""
        targets = {}
        
        target_columns = [col for col in df.columns if col.endswith('_target')]
        
        for col in target_columns:
            targets[col] = df[col]
            
        # Also include interaction targets
        interaction_targets = [col for col in df.columns if '_x_' in col or '_ratio_' in col]
        for col in interaction_targets:
            targets[col] = df[col]
        
        logger.info(f"🎯 BETTING TARGETS: {len(targets)}")
        for target_name in targets.keys():
            logger.info(f"   {target_name}: {len(targets[target_name].dropna()):,} samples")
            
        return targets
    
    def train_single_model(self, X, y, target_name):
        """Train a single model with optimized parameters"""
        logger.info(f"🎯 Training: {target_name}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Feature selection
        selector = SelectKBest(score_func=f_regression, k=min(150, X_train_scaled.shape[1]))
        X_train_selected = selector.fit_transform(X_train_scaled, y_train)
        X_test_selected = selector.transform(X_test_scaled)
        
        # Balanced parameter grid - comprehensive but reasonable
        param_grid = {
            'n_estimators': [300, 500, 700],      # 3 options
            'max_depth': [6, 8, 10, 12],          # 4 options  
            'learning_rate': [0.05, 0.1, 0.15],   # 3 options
            'subsample': [0.8, 0.9],              # 2 options
            'colsample_bytree': [0.8, 0.9],       # 2 options
            'gamma': [0, 0.1, 0.2],               # 3 options
            'reg_alpha': [0, 0.1, 0.3],           # 3 options
            'reg_lambda': [1, 1.5, 2]             # 3 options
        }
        # Total: 3×4×3×2×2×3×3×3 = 1,944 combinations = 9,720 fits (~9 hours)
        
        # GPU model
        base_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            device='cuda:0',
            tree_method='hist',
            random_state=42
        )
        
        # Grid search with detailed verbose output
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring='r2',
            cv=5,  # Reduced from 10 for stability
            n_jobs=1,
            verbose=3  # Maximum verbosity to see all training details
        )
        
        # Train model
        logger.info(f"🚀 Starting training for {target_name}...")
        logger.info(f"📊 Parameter grid: {len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['learning_rate']) * len(param_grid['subsample']) * len(param_grid['colsample_bytree']) * len(param_grid['gamma']) * len(param_grid['reg_alpha']) * len(param_grid['reg_lambda'])} combinations")
        logger.info(f"🔥 Total fits: {len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['learning_rate']) * len(param_grid['subsample']) * len(param_grid['colsample_bytree']) * len(param_grid['gamma']) * len(param_grid['reg_alpha']) * len(param_grid['reg_lambda']) * 5}")
        
        print(f"\n{'='*80}")
        print(f"🎯 TRAINING MODEL: {target_name}")
        print(f"📊 Features: {X_train_selected.shape[1]}")
        print(f"🎲 Parameter Combinations: {len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['learning_rate']) * len(param_grid['subsample']) * len(param_grid['colsample_bytree']) * len(param_grid['gamma']) * len(param_grid['reg_alpha']) * len(param_grid['reg_lambda'])}")
        print(f"🔄 CV Folds: 5")
        print(f"⚡ GPU: cuda:0")
        print(f"{'='*80}")
        
        grid_search.fit(X_train_selected, y_train)
        
        # Evaluate
        y_pred = grid_search.predict(X_test_selected)
        
        print(f"\n🏆 TRAINING COMPLETED FOR {target_name}!")
        print(f"📊 Best Parameters: {grid_search.best_params_}")
        print(f"🎯 Best CV Score: {grid_search.best_score_:.6f}")
        print(f"⭐ Best Index: {grid_search.best_index_}")
        
        metrics = {
            'r2_score': r2_score(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'accuracy': r2_score(y_test, y_pred) * 100,
            'features_used': X_train_selected.shape[1],
            'best_params': grid_search.best_params_,
            'best_cv_score': grid_search.best_score_
        }
        
        print(f"✅ Final Accuracy: {metrics['accuracy']:.2f}%")
        print(f"📏 MAE: {metrics['mae']:.4f}")
        print(f"📏 RMSE: {metrics['rmse']:.4f}")
        print(f"🎯 R² Score: {metrics['r2_score']:.6f}")
        
        logger.info(f"✅ {target_name}: {metrics['accuracy']:.1f}% accuracy")
        
        return grid_search.best_estimator_, scaler, selector, metrics
    
    def run_resumable_training(self):
        """Main training loop with resume capability"""
        print("🚀 RESUMABLE COMPREHENSIVE BETTING TRAINING")
        print("🎯 100k+ Records • GPU Acceleration • Resume from Interruptions")
        print("=" * 65)
        
        # Check for checkpoint
        checkpoint = self.load_checkpoint()
        
        if checkpoint:
            logger.info("📂 Resuming from checkpoint...")
            df = checkpoint['data']
            targets = checkpoint['targets']
            completed_models = checkpoint['completed_models']
            start_idx = checkpoint['current_model_idx']
        else:
            logger.info("🆕 Starting fresh training...")
            # Load data
            datasets = self.load_all_datasets()
            df = self.create_comprehensive_features(datasets)
            targets = self.create_betting_targets(df)
            completed_models = []
            start_idx = 0
        
        # Prepare features
        feature_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in feature_cols if not col.endswith('_target') and '_x_' not in col and '_ratio_' not in col]
        X = df[feature_cols].fillna(0)
        
        logger.info(f"🎯 Training {len(targets)} models starting from model {start_idx + 1}")
        
        # Train models one by one
        target_list = list(targets.items())
        
        for i in range(start_idx, len(target_list)):
            target_name, y = target_list[i]
            
            if len(y.dropna()) < 100:
                logger.warning(f"⚠️ Skipping {target_name}: insufficient data")
                continue
            
            try:
                logger.info(f"🎯 MODEL {i + 1}/{len(target_list)}: {target_name}")
                
                # Train model
                model, scaler, selector, metrics = self.train_single_model(X, y, target_name)
                
                # Save model immediately
                self.save_model_result(target_name, model, scaler, selector, metrics)
                
                # Update completed models
                completed_models.append({
                    'target_name': target_name,
                    'metrics': metrics,
                    'model_idx': i
                })
                
                # Save checkpoint after each model
                self.save_checkpoint(df, targets, completed_models, i)
                
                logger.info(f"✅ Completed {i + 1}/{len(target_list)} models")
                
            except KeyboardInterrupt:
                logger.info("⚠️ Training interrupted by user")
                self.save_checkpoint(df, targets, completed_models, i)
                break
            except Exception as e:
                logger.error(f"❌ Error training {target_name}: {e}")
                continue
        
        # Final summary
        self.print_final_summary(completed_models)
        
        return completed_models
    
    def print_final_summary(self, completed_models):
        """Print comprehensive training summary"""
        if not completed_models:
            logger.warning("No models completed")
            return
            
        print("\n🏆 TRAINING COMPLETE SUMMARY")
        print("=" * 50)
        
        for model_info in completed_models:
            target = model_info['target_name']
            metrics = model_info['metrics']
            accuracy = metrics['accuracy']
            
            if accuracy >= 85:
                status = "🔥 ELITE"
            elif accuracy >= 75:
                status = "⭐ EXCELLENT"
            elif accuracy >= 65:
                status = "✅ GOOD"
            else:
                status = "📊 BASELINE"
                
            print(f"{target:<30} {accuracy:>6.1f}% {status}")
        
        avg_accuracy = np.mean([m['metrics']['accuracy'] for m in completed_models])
        max_accuracy = max([m['metrics']['accuracy'] for m in completed_models])
        
        print(f"\n📊 FINAL RESULTS:")
        print(f"   Models Completed: {len(completed_models)}")
        print(f"   Average Accuracy: {avg_accuracy:.1f}%")
        print(f"   Maximum Accuracy: {max_accuracy:.1f}%")
        print(f"   Models Saved: {len(completed_models)}")


def main():
    """Main execution"""
    trainer = ResumableBettingTrainer()
    
    try:
        results = trainer.run_resumable_training()
        logger.info("🎉 Training completed successfully!")
        return trainer, results
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    trainer, results = main()