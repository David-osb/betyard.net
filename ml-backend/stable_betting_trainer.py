#!/usr/bin/env python3
"""
Stable Betting XGBoost Trainer - One Model at a Time
Designed for 100k+ records without memory crashes
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import os
import logging
import gc
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StableBettingTrainer:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        
    def load_datasets(self):
        """Load all cleaned datasets safely"""
        logger.info("📁 Loading datasets...")
        
        datasets = {}
        
        # Load cleaned datasets with error handling
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
                    df = pd.read_csv(file_path)
                    datasets[name] = df
                    total_records += len(df)
                    logger.info(f"✅ {name}: {len(df):,} records")
                else:
                    logger.warning(f"⚠️ File not found: {file_path}")
            except Exception as e:
                logger.error(f"❌ Error loading {name}: {e}")
        
        logger.info(f"📊 Total records: {total_records:,}")
        return datasets
    
    def create_simple_features(self, datasets):
        """Create essential features without memory-intensive operations"""
        logger.info("🔧 Creating stable features...")
        
        all_features = []
        
        for stat_type, df in datasets.items():
            logger.info(f"Processing {stat_type}...")
            
            # Basic betting features only
            if stat_type == 'receiving':
                if 'Rec_Yds' in df.columns:
                    df['rec_yards_target'] = df['Rec_Yds']
                    df['rec_yards_per_game'] = df['Rec_Yds'] / df.get('G', 1).replace(0, 1)
                
                if 'Rec' in df.columns:
                    df['receptions_target'] = df['Rec']
                    df['receptions_per_game'] = df['Rec'] / df.get('G', 1).replace(0, 1)
                
                if 'TD' in df.columns:
                    df['rec_td_target'] = df['TD']
            
            elif stat_type == 'rushing':
                if 'Yds' in df.columns:
                    df['rush_yards_target'] = df['Yds']
                    df['rush_yards_per_game'] = df['Yds'] / df.get('G', 1).replace(0, 1)
                
                if 'Att' in df.columns:
                    df['rush_attempts_target'] = df['Att']
                    df['yards_per_carry'] = df['Yds'] / df['Att'].replace(0, 1)
                
                if 'TD' in df.columns:
                    df['rush_td_target'] = df['TD']
            
            elif stat_type == 'passing':
                if 'Yds' in df.columns:
                    df['pass_yards_target'] = df['Yds']
                    df['pass_yards_per_game'] = df['Yds'] / df.get('G', 1).replace(0, 1)
                
                if 'TD' in df.columns:
                    df['pass_td_target'] = df['TD']
                
                if 'Int' in df.columns:
                    df['pass_int_target'] = df['Int']
            
            elif stat_type == 'scoring':
                if 'Pts' in df.columns:
                    df['fantasy_points_target'] = df['Pts']
                    df['fantasy_points_per_game'] = df['Pts'] / df.get('G', 1).replace(0, 1)
            
            # Add universal features
            df['games_played'] = df.get('G', 0)
            
            # Safe team encoding
            if 'Tm' in df.columns:
                df['team_encoded'] = pd.Categorical(df['Tm'].fillna('UNK')).codes
            else:
                df['team_encoded'] = 0
            
            all_features.append(df)
        
        # Combine safely
        combined_df = pd.concat(all_features, ignore_index=True, sort=False)
        combined_df = combined_df.fillna(0)
        
        # Remove infinite values
        combined_df = combined_df.replace([np.inf, -np.inf], 0)
        
        logger.info(f"✅ Stable features: {len(combined_df):,} records × {len(combined_df.columns):,} features")
        return combined_df
    
    def get_betting_targets(self, df):
        """Get betting targets one at a time"""
        targets = {}
        
        # Only include targets with sufficient data
        target_columns = [col for col in df.columns if col.endswith('_target')]
        
        for col in target_columns:
            if col in df.columns:
                target_data = df[col].dropna()
                if len(target_data) >= 1000:  # Minimum samples
                    targets[col] = target_data
                    logger.info(f"🎯 {col}: {len(target_data):,} samples")
        
        return targets
    
    def train_single_model(self, df, target_name, target_data):
        """Train one model at a time to prevent crashes"""
        logger.info(f"\n🚀 Training {target_name}...")
        
        # Prepare features
        feature_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in feature_cols if not col.endswith('_target')]
        
        # Get valid samples
        valid_indices = target_data.index
        X = df.loc[valid_indices, feature_cols].fillna(0)
        y = target_data
        
        logger.info(f"📊 Training on {len(X):,} samples with {len(feature_cols)} features")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Check GPU availability
        gpu_available = self._check_gpu()
        
        if gpu_available:
            logger.info("⚡ Using GPU acceleration")
            model = self._train_gpu_model(X_train_scaled, y_train)
        else:
            logger.info("💻 Using CPU training")
            model = self._train_cpu_model(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        # Store results
        self.models[target_name] = model
        self.scalers[target_name] = scaler
        
        # Results
        results = {
            'model': model,
            'scaler': scaler,
            'r2_score': r2,
            'cv_r2_mean': cv_mean,
            'cv_r2_std': cv_std,
            'mae': mae,
            'rmse': rmse,
            'accuracy_pct': r2 * 100,
            'samples': len(X),
            'features': len(feature_cols),
            'gpu_used': gpu_available
        }
        
        # Log results
        print(f"✅ {target_name}:")
        print(f"   📊 R² Score: {r2:.4f} ({r2*100:.2f}% accuracy)")
        print(f"   🔄 CV Score: {cv_mean:.4f} ± {cv_std:.4f}")
        print(f"   📏 MAE: {mae:.3f}, RMSE: {rmse:.3f}")
        print(f"   📈 Samples: {len(X):,}")
        
        # Clean up memory
        del X_train, X_test, X_train_scaled, X_test_scaled
        gc.collect()
        
        return results
    
    def _check_gpu(self):
        """Check GPU availability safely"""
        try:
            test_model = xgb.XGBRegressor(device='cuda:0', tree_method='hist')
            test_X = np.random.rand(10, 5)
            test_y = np.random.rand(10)
            test_model.fit(test_X, test_y)
            return True
        except:
            return False
    
    def _train_gpu_model(self, X_train, y_train):
        """Simplified GPU training"""
        # Conservative GPU parameters to prevent crashes
        param_grid = {
            'n_estimators': [300, 500],
            'max_depth': [6, 8, 10],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.8, 0.9]
        }
        
        base_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            device='cuda:0',
            tree_method='hist',
            random_state=42
        )
        
        # Reduced CV to prevent memory issues
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring='r2',
            cv=3,  # Reduced from 10
            n_jobs=1,
            verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_
    
    def _train_cpu_model(self, X_train, y_train):
        """Simple CPU training"""
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            tree_method='hist',
            n_estimators=300,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        return model
    
    def save_model(self, target_name, results):
        """Save individual model"""
        os.makedirs('models/stable_betting', exist_ok=True)
        
        # Save model
        model_path = f'models/stable_betting/{target_name}_model.joblib'
        joblib.dump(results['model'], model_path)
        
        # Save scaler
        scaler_path = f'models/stable_betting/{target_name}_scaler.joblib'
        joblib.dump(results['scaler'], scaler_path)
        
        logger.info(f"💾 Saved {target_name} model")

def main():
    """Stable training pipeline - one model at a time"""
    print(f"🔧 STABLE BETTING TRAINER")
    print(f"📊 One Model at a Time - No Crashes!")
    print(f"{'='*45}")
    
    trainer = StableBettingTrainer()
    
    try:
        # Load datasets
        datasets = trainer.load_datasets()
        
        # Create stable features
        df = trainer.create_simple_features(datasets)
        
        # Get targets
        targets = trainer.get_betting_targets(df)
        
        print(f"\n🎯 Found {len(targets)} betting targets")
        
        # Train models one by one
        all_results = {}
        successful_models = 0
        
        for i, (target_name, target_data) in enumerate(targets.items(), 1):
            try:
                print(f"\n{'='*50}")
                print(f"🚀 MODEL {i}/{len(targets)}: {target_name}")
                print(f"{'='*50}")
                
                start_time = time.time()
                
                # Train single model
                results = trainer.train_single_model(df, target_name, target_data)
                
                # Save immediately
                trainer.save_model(target_name, results)
                
                all_results[target_name] = results
                successful_models += 1
                
                elapsed = time.time() - start_time
                print(f"⏱️ Completed in {elapsed:.1f} seconds")
                print(f"✅ Progress: {successful_models}/{len(targets)} models")
                
                # Memory cleanup between models
                gc.collect()
                
            except Exception as e:
                logger.error(f"❌ Failed to train {target_name}: {e}")
                continue
        
        # Final summary
        if all_results:
            print(f"\n🏆 TRAINING COMPLETE!")
            print(f"✅ Successful Models: {successful_models}/{len(targets)}")
            
            accuracies = [r['accuracy_pct'] for r in all_results.values()]
            avg_accuracy = np.mean(accuracies)
            max_accuracy = max(accuracies)
            
            print(f"📈 Average Accuracy: {avg_accuracy:.2f}%")
            print(f"🔝 Maximum Accuracy: {max_accuracy:.2f}%")
            
            # High accuracy models
            high_accuracy = [name for name, r in all_results.items() if r['accuracy_pct'] >= 80.0]
            if high_accuracy:
                print(f"🥇 80%+ Accuracy Models: {len(high_accuracy)}")
                for name in high_accuracy:
                    acc = all_results[name]['accuracy_pct']
                    print(f"   🎯 {name}: {acc:.2f}%")
        
        print(f"\n💾 All models saved to 'models/stable_betting/'")
        return trainer, all_results
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    trainer, results = main()