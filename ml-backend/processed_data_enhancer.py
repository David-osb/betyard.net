"""
Enhanced XGBoost Trainer for Existing Processed Data
90%+ Accuracy Target Implementation
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import joblib
import json
import logging
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProcessedDataEnhancer:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_engineers = {}
        
    def load_processed_data(self):
        """Load all processed data files"""
        logger.info("🏈 Loading processed NFL data...")
        
        data_files = {
            'passing': 'processed_data/processed_passing_data.csv',
            'rushing': 'processed_data/processed_rushing_data.csv',
            'receiving': 'processed_data/processed_receiving_data.csv',
            'defense': 'processed_data/processed_defense_data.csv',
            'kicking': 'processed_data/processed_kicking_data.csv',
            'scrimmage': 'processed_data/processed_scrimmage_data.csv'
        }
        
        datasets = {}
        total_records = 0
        
        for position, filepath in data_files.items():
            if os.path.exists(filepath):
                try:
                    df = pd.read_csv(filepath)
                    if len(df) > 0:
                        datasets[position] = df
                        total_records += len(df)
                        logger.info(f"✅ {position.upper()}: {len(df):,} records")
                except Exception as e:
                    logger.warning(f"⚠️ Could not load {position}: {e}")
            else:
                logger.warning(f"⚠️ File not found: {filepath}")
        
        logger.info(f"🎯 Total records loaded: {total_records:,}")
        return datasets
    
    def enhance_features(self, datasets):
        """Create advanced features from processed data"""
        logger.info("🔧 Creating enhanced features...")
        
        enhanced_datasets = {}
        
        for position, df in datasets.items():
            df_enhanced = df.copy()
            
            # Universal enhancements for all positions
            df_enhanced = self._add_universal_features(df_enhanced)
            
            # Position-specific enhancements
            if position == 'passing':
                df_enhanced = self._enhance_passing_features(df_enhanced)
            elif position == 'rushing':
                df_enhanced = self._enhance_rushing_features(df_enhanced)
            elif position == 'receiving':
                df_enhanced = self._enhance_receiving_features(df_enhanced)
            elif position == 'defense':
                df_enhanced = self._enhance_defense_features(df_enhanced)
            elif position == 'kicking':
                df_enhanced = self._enhance_kicking_features(df_enhanced)
            elif position == 'scrimmage':
                df_enhanced = self._enhance_scrimmage_features(df_enhanced)
            
            # Polynomial features for top predictors
            df_enhanced = self._add_polynomial_features(df_enhanced, degree=2)
            
            enhanced_datasets[position] = df_enhanced
            logger.info(f"Enhanced {position}: {len(df_enhanced.columns)} features")
        
        return enhanced_datasets
    
    def _add_universal_features(self, df):
        """Add features that work for all positions"""
        # Age-based features
        if 'age' in df.columns:
            df['age_squared'] = df['age'] ** 2
            df['age_cubed'] = df['age'] ** 3
            df['prime_years'] = ((df['age'] >= 25) & (df['age'] <= 29)).astype(int)
            df['veteran_status'] = (df['age'] >= 30).astype(int)
            df['rookie_era'] = (df['age'] <= 23).astype(int)
            df['age_experience'] = df['age'] - 22  # Estimated years of experience
        
        # Games played features
        if 'games_played' in df.columns:
            df['games_ratio'] = df['games_played'] / 17  # Current season length
            df['durability'] = np.where(df['games_played'] >= 14, 1, df['games_played'] / 14)
            df['injury_prone'] = (df['games_played'] < 12).astype(int)
            df['iron_man'] = (df['games_played'] >= 16).astype(int)
        
        # Team context (assuming team_encoded exists)
        if 'team_encoded' in df.columns:
            df['team_tier'] = np.where(df['team_encoded'] <= 10, 1,
                                     np.where(df['team_encoded'] <= 20, 2, 3))
            df['market_size'] = df['team_encoded'] % 4  # Rough market classification
        
        return df
    
    def _enhance_passing_features(self, df):
        """Enhanced features for passing data"""
        if 'attempts_per_game' in df.columns and 'completion_rate' in df.columns:
            # Efficiency metrics
            df['completion_efficiency'] = df['completion_rate'] * df['attempts_per_game']
            df['volume_accuracy_combo'] = df['attempts_per_game'] * (df['completion_rate'] ** 2)
            
            # Risk assessment
            df['conservative_passer'] = (df['attempts_per_game'] < 30).astype(int)
            df['gunslinger'] = (df['attempts_per_game'] > 40).astype(int)
            
            # Accuracy tiers
            df['elite_accuracy'] = (df['completion_rate'] > 0.68).astype(int)
            df['accuracy_concern'] = (df['completion_rate'] < 0.58).astype(int)
            
            # Combined metrics
            df['passing_impact'] = df['attempts_per_game'] * df['completion_rate'] * df['age'] / 100
        
        return df
    
    def _enhance_rushing_features(self, df):
        """Enhanced features for rushing data"""
        if 'attempts_per_game' in df.columns and 'yards_per_attempt' in df.columns:
            # Efficiency metrics
            df['rushing_efficiency'] = df['yards_per_attempt'] * df['attempts_per_game']
            df['power_runner'] = (df['attempts_per_game'] > 15).astype(int)
            df['explosive_runner'] = (df['yards_per_attempt'] > 5.0).astype(int)
            
            # Workload categories
            df['bellcow_back'] = (df['attempts_per_game'] > 18).astype(int)
            df['change_of_pace'] = (df['attempts_per_game'] < 8).astype(int)
            
            # Efficiency tiers
            df['elite_efficiency'] = (df['yards_per_attempt'] > 4.5).astype(int)
            df['efficiency_concern'] = (df['yards_per_attempt'] < 3.5).astype(int)
        
        if 'yards_per_game' in df.columns:
            df['volume_producer'] = (df['yards_per_game'] > 80).astype(int)
            df['goal_line_back'] = ((df['attempts_per_game'] > 0) & (df['yards_per_game'] < 40)).astype(int)
        
        return df
    
    def _enhance_receiving_features(self, df):
        """Enhanced features for receiving data"""
        if 'receptions_per_game' in df.columns and 'yards_per_reception' in df.columns:
            # Target share and efficiency
            df['reception_efficiency'] = df['receptions_per_game'] * df['yards_per_reception']
            df['volume_receiver'] = (df['receptions_per_game'] > 8).astype(int)
            df['deep_threat'] = (df['yards_per_reception'] > 15).astype(int)
            
            # Role classification
            df['possession_receiver'] = ((df['receptions_per_game'] > 6) & (df['yards_per_reception'] < 12)).astype(int)
            df['red_zone_target'] = ((df['receptions_per_game'] > 4) & (df['yards_per_reception'] < 10)).astype(int)
            df['big_play_specialist'] = ((df['receptions_per_game'] < 5) & (df['yards_per_reception'] > 18)).astype(int)
        
        if 'yards_per_game' in df.columns:
            df['yardage_producer'] = (df['yards_per_game'] > 60).astype(int)
            df['chain_mover'] = ((df['receptions_per_game'] > 5) & (df['yards_per_game'] > 40)).astype(int)
        
        return df
    
    def _enhance_defense_features(self, df):
        """Enhanced features for defense data (if available)"""
        # This would depend on what columns are available in defense data
        # For now, just add basic universal features
        return df
    
    def _enhance_kicking_features(self, df):
        """Enhanced features for kicking data (if available)"""
        # This would depend on what columns are available in kicking data
        return df
    
    def _enhance_scrimmage_features(self, df):
        """Enhanced features for scrimmage data"""
        # This would depend on what columns are available
        return df
    
    def _add_polynomial_features(self, df, degree=2):
        """Add polynomial features for key predictors"""
        # Only for numeric columns, and only a subset to avoid explosion
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Select key columns for polynomial expansion (max 3 to avoid too many features)
        key_cols = []
        priority_cols = ['age', 'games_played', 'completion_rate', 'yards_per_attempt', 
                        'yards_per_reception', 'receptions_per_game', 'attempts_per_game']
        
        for col in priority_cols:
            if col in numeric_cols and len(key_cols) < 3:
                key_cols.append(col)
        
        if key_cols:
            # Create polynomial features
            poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=True)
            poly_features = poly.fit_transform(df[key_cols])
            
            # Get feature names
            poly_names = poly.get_feature_names_out(key_cols)
            
            # Add only interaction terms (not pure polynomials to avoid redundancy)
            for i, name in enumerate(poly_names):
                if '*' in name:  # Only interaction terms
                    df[f'poly_{name}'] = poly_features[:, i]
        
        return df
    
    def prepare_combined_training_data(self, enhanced_datasets):
        """Combine all enhanced datasets for training"""
        logger.info("🤖 Preparing combined training data...")
        
        all_data = []
        
        for position, df in enhanced_datasets.items():
            # Add position encoding
            df_pos = df.copy()
            df_pos['pos_passing'] = int(position == 'passing')
            df_pos['pos_rushing'] = int(position == 'rushing')
            df_pos['pos_receiving'] = int(position == 'receiving')
            df_pos['pos_defense'] = int(position == 'defense')
            df_pos['pos_kicking'] = int(position == 'kicking')
            df_pos['pos_scrimmage'] = int(position == 'scrimmage')
            
            all_data.append(df_pos)
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True, sort=False)
        
        # Fill missing values
        combined_df = combined_df.fillna(0)
        
        # Separate features and target
        if 'target' in combined_df.columns:
            y = combined_df['target']
            X = combined_df.drop('target', axis=1)
        else:
            raise ValueError("No target column found in data")
        
        # Select only numeric features
        X = X.select_dtypes(include=[np.number])
        
        # Remove infinite values
        X = X.replace([np.inf, -np.inf], 0)
        
        # Remove columns with all zeros
        X = X.loc[:, (X != 0).any(axis=0)]
        
        logger.info(f"✅ Combined data: {len(X):,} samples, {len(X.columns)} features")
        
        return X, y
    
    def train_enhanced_model(self, X, y):
        """Train enhanced XGBoost model with aggressive hyperparameter tuning"""
        logger.info("🚀 Training enhanced XGBoost model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # GPU-optimized hyperparameter grid (faster + comprehensive)
        param_grid = {
            'n_estimators': [200, 400, 600],
            'max_depth': [6, 8, 10],
            'learning_rate': [0.05, 0.1, 0.15],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.8, 0.9],
            'min_child_weight': [1, 3],
            'gamma': [0, 0.1],
            'reg_alpha': [0, 0.1],
            'reg_lambda': [1, 1.5]
        }
        
        # Try GPU first, fallback to optimized CPU
        try:
            # GPU-accelerated XGBoost
            xgb_model = xgb.XGBRegressor(
                objective='reg:squarederror',
                random_state=42,
                tree_method='gpu_hist',
                gpu_id=0,
                n_jobs=-1
            )
            logger.info("🚀 Using GPU acceleration (CUDA)")
        except:
            # Optimized CPU version
            xgb_model = xgb.XGBRegressor(
                objective='reg:squarederror',
                random_state=42,
                tree_method='hist',
                n_jobs=-1
            )
            logger.info("💻 Using optimized CPU training")
        
        # GPU-optimized Grid search
        logger.info("🔧 GPU-accelerated hyperparameter tuning...")
        grid_search = GridSearchCV(
            estimator=xgb_model,
            param_grid=param_grid,
            scoring='r2',
            cv=3,  # Reduced from 5 for speed
            n_jobs=4,  # Limited parallel jobs for GPU compatibility
            verbose=2  # More verbose for progress tracking
        )
        
        grid_search.fit(X_train_scaled, y_train)
        best_model = grid_search.best_estimator_
        
        # Train final model with early stopping
        best_model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_val_scaled, y_val)],
            early_stopping_rounds=50,
            verbose=False
        )
        
        # Predictions
        y_pred_train = best_model.predict(X_train_scaled)
        y_pred_test = best_model.predict(X_test_scaled)
        
        # Metrics
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        # Cross-validation
        cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=5, scoring='r2')
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        results = {
            'model': best_model,
            'scaler': scaler,
            'best_params': grid_search.best_params_,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'feature_importance': feature_importance,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'features_count': len(X.columns)
        }
        
        # Results display
        accuracy_pct = test_r2 * 100
        
        print(f"\\n🏆 ENHANCED MODEL RESULTS:")
        print(f"{'='*50}")
        print(f"Training R²: {train_r2:.4f}")
        print(f"Test R²: {test_r2:.4f}")
        print(f"Training MAE: {train_mae:.3f}")
        print(f"Test MAE: {test_mae:.3f}")
        print(f"Cross-validation R²: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        print(f"\\n🎯 ESTIMATED ACCURACY: {accuracy_pct:.1f}%")
        
        if accuracy_pct >= 90:
            print("🔥 ELITE TIER: Professional sportsbook quality!")
        elif accuracy_pct >= 85:
            print("⭐ EXCELLENT: High-quality predictions!")
        elif accuracy_pct >= 80:
            print("✅ VERY GOOD: Strong predictive power!")
        elif accuracy_pct >= 75:
            print("📊 GOOD: Solid performance!")
        else:
            print("📈 BASELINE: Improvement needed")
        
        return results
    
    def save_enhanced_model(self, results):
        """Save the enhanced model and all metadata"""
        os.makedirs('models/enhanced', exist_ok=True)
        
        # Save model
        model_path = 'models/enhanced/enhanced_nfl_model.joblib'
        joblib.dump(results['model'], model_path)
        
        # Save scaler
        scaler_path = 'models/enhanced/enhanced_scaler.joblib'
        joblib.dump(results['scaler'], scaler_path)
        
        # Save comprehensive metadata
        metadata = {
            'model_info': {
                'type': 'Enhanced XGBoost',
                'training_date': datetime.now().isoformat(),
                'version': '2.0_enhanced'
            },
            'performance': {
                'train_r2': float(results['train_r2']),
                'test_r2': float(results['test_r2']),
                'train_mae': float(results['train_mae']),
                'test_mae': float(results['test_mae']),
                'cv_mean': float(results['cv_mean']),
                'cv_std': float(results['cv_std']),
                'accuracy_percentage': float(results['test_r2'] * 100)
            },
            'training_data': {
                'total_samples': int(results['training_samples'] + results['test_samples']),
                'training_samples': int(results['training_samples']),
                'test_samples': int(results['test_samples']),
                'features_count': int(results['features_count'])
            },
            'hyperparameters': results['best_params'],
            'top_features': results['feature_importance'].head(15).to_dict('records')
        }
        
        metadata_path = 'models/enhanced/enhanced_model_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save detailed feature importance
        importance_path = 'models/enhanced/feature_importance.csv'
        results['feature_importance'].to_csv(importance_path, index=False)
        
        logger.info(f"💾 Model saved: {model_path}")
        logger.info(f"⚖️ Scaler saved: {scaler_path}")
        logger.info(f"📊 Metadata saved: {metadata_path}")
        logger.info(f"📈 Feature importance saved: {importance_path}")
        
        return model_path, metadata_path


def main():
    """Main enhanced training pipeline"""
    print(f"🏈 ENHANCED NFL PREDICTION MODEL")
    print(f"Target: 90%+ Accuracy with Processed Data")
    print(f"{'='*55}")
    
    enhancer = ProcessedDataEnhancer()
    
    try:
        # Step 1: Load processed data
        datasets = enhancer.load_processed_data()
        
        if not datasets:
            print("❌ No processed data found. Please run data processing first.")
            return None, None
        
        # Step 2: Enhance features
        enhanced_datasets = enhancer.enhance_features(datasets)
        
        # Step 3: Prepare training data
        X, y = enhancer.prepare_combined_training_data(enhanced_datasets)
        
        # Step 4: Train enhanced model
        results = enhancer.train_enhanced_model(X, y)
        
        # Step 5: Save everything
        model_path, metadata_path = enhancer.save_enhanced_model(results)
        
        # Summary
        print(f"\\n🎉 ENHANCEMENT COMPLETE!")
        print(f"🎯 Final Accuracy: {results['test_r2']*100:.1f}%")
        print(f"📊 Total Samples: {results['training_samples'] + results['test_samples']:,}")
        print(f"🔧 Features Used: {results['features_count']}")
        print(f"💾 Model: {model_path}")
        
        # Show top features
        print(f"\\n🔝 TOP 10 MOST IMPORTANT FEATURES:")
        print(results['feature_importance'].head(10)[['feature', 'importance']].to_string(index=False))
        
        return enhancer, results
        
    except Exception as e:
        logger.error(f"❌ Enhancement failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    enhancer, results = main()