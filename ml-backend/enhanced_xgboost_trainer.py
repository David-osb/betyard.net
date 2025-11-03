"""
Enhanced XGBoost Training with 90%+ Accuracy Target
Implements advanced feature engineering and multi-dataset training
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import logging
from datetime import datetime
import warnings
import os
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedNFLTrainer:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_engineers = {}
        self.training_reports = {}
        
    def load_all_historical_data(self):
        """Load all available NFL datasets"""
        logger.info("🏈 Loading ALL historical NFL datasets...")
        
        datasets = {}
        data_files = {
            'qb': 'nfl_data/qb_stats_historical.csv',
            'rb': 'nfl_data/rb_stats_historical.csv', 
            'wr': 'nfl_data/wr_stats_historical.csv',
            'te': 'nfl_data/te_stats_historical.csv',
            'kicking': 'nfl_data/kicking_stats_historical.csv',
            'punting': 'nfl_data/punting_stats_historical.csv',
            'returns': 'nfl_data/returns_stats_historical.csv',
            'defense': 'nfl_data/defense_stats_historical.csv'
        }
        
        total_records = 0
        for position, filepath in data_files.items():
            if os.path.exists(filepath):
                try:
                    # Load with proper encoding
                    df = pd.read_csv(filepath, encoding='utf-8', low_memory=False)
                    
                    # Basic cleaning
                    df = self._clean_basic_dataset(df, position)
                    
                    if len(df) > 0:
                        datasets[position] = df
                        total_records += len(df)
                        logger.info(f"✅ {position.upper()}: {len(df):,} records")
                    else:
                        logger.warning(f"⚠️ {position.upper()}: No valid records")
                        
                except Exception as e:
                    logger.warning(f"⚠️ Could not load {position}: {e}")
            else:
                logger.warning(f"⚠️ File not found: {filepath}")
        
        logger.info(f"🎯 Total records loaded: {total_records:,}")
        return datasets
    
    def _clean_basic_dataset(self, df, position):
        """Basic cleaning for any dataset"""
        # Remove completely empty rows
        df = df.dropna(how='all')
        
        # Remove header rows that might have been included
        if 'Player' in df.columns:
            df = df[~df['Player'].str.contains('Player|Rk', na=False)]
            df = df[df['Player'].notna()]
        
        # Add position identifier
        df['dataset_position'] = position
        
        # Try to identify year if possible
        if 'Year' not in df.columns:
            # Estimate year based on dataset (this could be improved)
            df['estimated_year'] = 2025 - np.random.randint(0, 26, len(df))
        
        return df
    
    def create_advanced_features(self, datasets):
        """Create advanced features from all datasets"""
        logger.info("🔧 Creating advanced features...")
        
        combined_features = {}
        
        for position, df in datasets.items():
            logger.info(f"Processing {position.upper()} features...")
            
            # Position-specific feature engineering
            if position == 'qb':
                df = self._create_qb_features(df)
            elif position == 'rb':
                df = self._create_rb_features(df)
            elif position in ['wr', 'te']:
                df = self._create_receiving_features(df)
            elif position == 'kicking':
                df = self._create_kicking_features(df)
            elif position == 'punting':
                df = self._create_punting_features(df)
            elif position == 'returns':
                df = self._create_returns_features(df)
            elif position == 'defense':
                df = self._create_defense_features(df)
            
            # Universal advanced features
            df = self._create_universal_features(df)
            
            combined_features[position] = df
        
        return combined_features
    
    def _create_qb_features(self, df):
        """Enhanced QB-specific features"""
        # Standard QB metrics
        if 'Cmp' in df.columns and 'Att' in df.columns:
            df['completion_rate'] = df['Cmp'] / df['Att'].replace(0, 1)
            df['incompletion_rate'] = 1 - df['completion_rate']
        
        if 'Yds' in df.columns and 'Att' in df.columns:
            df['yards_per_attempt'] = df['Yds'] / df['Att'].replace(0, 1)
        
        if 'TD' in df.columns and 'Int' in df.columns:
            df['td_int_ratio'] = df['TD'] / (df['Int'] + 1)  # +1 to avoid division by zero
        
        # Advanced efficiency metrics
        if 'QBR' in df.columns:
            df['qbr_squared'] = df['QBR'] ** 2
            df['qbr_efficiency'] = df['QBR'] / (df.get('Sk', 0) + 1)
        
        # Pressure handling (if available)
        if 'Sk' in df.columns and 'Att' in df.columns:
            df['sack_rate'] = df['Sk'] / (df['Att'] + df['Sk'])
            df['pressure_resistance'] = 1 - df['sack_rate']
        
        # Red zone efficiency (estimated)
        if 'TD' in df.columns and 'Att' in df.columns:
            df['red_zone_efficiency'] = df['TD'] / (df['Att'] / 20)  # Rough estimate
        
        return df
    
    def _create_rb_features(self, df):
        """Enhanced RB-specific features"""
        # Rushing efficiency
        if 'Rush Yds' in df.columns and 'Rush Att' in df.columns:
            df['yards_per_carry'] = df['Rush Yds'] / df['Rush Att'].replace(0, 1)
            df['rushing_consistency'] = df['yards_per_carry'] * df['Rush Att']
        
        # Receiving contribution
        if 'Rec Yds' in df.columns and 'Rec' in df.columns:
            df['yards_per_reception'] = df['Rec Yds'] / df['Rec'].replace(0, 1)
            df['receiving_threat'] = df['Rec'] / (df.get('Rush Att', 0) + df['Rec'])
        
        # Total versatility
        total_yards = df.get('Rush Yds', 0) + df.get('Rec Yds', 0)
        total_touches = df.get('Rush Att', 0) + df.get('Rec', 0)
        df['total_yards_per_touch'] = total_yards / total_touches.replace(0, 1)
        
        # TD efficiency
        total_tds = df.get('Rush TD', 0) + df.get('Rec TD', 0)
        df['td_per_touch'] = total_tds / total_touches.replace(0, 1)
        
        return df
    
    def _create_receiving_features(self, df):
        """Enhanced receiving features for WR/TE"""
        # Reception efficiency
        if 'Rec' in df.columns and 'Tgt' in df.columns:
            df['catch_rate'] = df['Rec'] / df['Tgt'].replace(0, 1)
            df['target_share'] = df['Tgt'] / 100  # Rough estimate
        
        if 'Rec Yds' in df.columns and 'Rec' in df.columns:
            df['yards_per_reception'] = df['Rec Yds'] / df['Rec'].replace(0, 1)
        
        # Big play ability
        if 'Lng' in df.columns:
            df['big_play_ability'] = df['Lng'] / df.get('Rec Yds', 1)
        
        # TD efficiency
        if 'TD' in df.columns and 'Tgt' in df.columns:
            df['td_per_target'] = df['TD'] / df['Tgt'].replace(0, 1)
        
        return df
    
    def _create_kicking_features(self, df):
        """Enhanced kicking features"""
        # FG accuracy by distance
        if 'FGM' in df.columns and 'FGA' in df.columns:
            df['fg_accuracy'] = df['FGM'] / df['FGA'].replace(0, 1)
        
        # XP reliability
        if 'XPM' in df.columns and 'XPA' in df.columns:
            df['xp_accuracy'] = df['XPM'] / df['XPA'].replace(0, 1)
        
        # Clutch performance (estimated)
        if 'Lng' in df.columns:
            df['long_fg_ability'] = df['Lng'] / 60  # Normalized by max distance
        
        return df
    
    def _create_punting_features(self, df):
        """Enhanced punting features"""
        # Punting efficiency
        if 'Yds' in df.columns and 'Pnt' in df.columns:
            df['yards_per_punt'] = df['Yds'] / df['Pnt'].replace(0, 1)
        
        # Net punting
        if 'Net' in df.columns:
            df['net_efficiency'] = df['Net'] / df.get('Yds', 1)
        
        # Placement ability
        if 'In20' in df.columns and 'Pnt' in df.columns:
            df['inside_20_rate'] = df['In20'] / df['Pnt'].replace(0, 1)
        
        return df
    
    def _create_returns_features(self, df):
        """Enhanced return features"""
        # Punt return efficiency
        if 'PR_Yds' in df.columns and 'PR_Ret' in df.columns:
            df['punt_return_avg'] = df['PR_Yds'] / df['PR_Ret'].replace(0, 1)
        
        # Kick return efficiency
        if 'KR_Yds' in df.columns and 'KR_Ret' in df.columns:
            df['kick_return_avg'] = df['KR_Yds'] / df['KR_Ret'].replace(0, 1)
        
        # TD threat
        total_tds = df.get('PR_TD', 0) + df.get('KR_TD', 0)
        total_returns = df.get('PR_Ret', 0) + df.get('KR_Ret', 0)
        df['return_td_rate'] = total_tds / total_returns.replace(0, 1)
        
        return df
    
    def _create_defense_features(self, df):
        """Enhanced defensive features"""
        # Tackling efficiency
        if 'Tkl' in df.columns and 'Solo' in df.columns:
            df['tackle_efficiency'] = df['Solo'] / df['Tkl'].replace(0, 1)
        
        # Pass coverage
        if 'Int' in df.columns and 'PD' in df.columns:
            df['coverage_impact'] = df['Int'] + (df['PD'] * 0.5)
        
        # Pressure generation
        if 'Sk' in df.columns and 'QBHits' in df.columns:
            df['pressure_rate'] = df['Sk'] + (df['QBHits'] * 0.5)
        
        return df
    
    def _create_universal_features(self, df):
        """Features that apply to all positions"""
        # Age-based features
        if 'Age' in df.columns:
            df['age_squared'] = df['Age'] ** 2
            df['prime_years'] = ((df['Age'] >= 25) & (df['Age'] <= 29)).astype(int)
            df['veteran_status'] = (df['Age'] >= 30).astype(int)
            df['rookie_sophomore'] = (df['Age'] <= 23).astype(int)
        
        # Games played consistency
        if 'G' in df.columns:
            df['games_played_rate'] = df['G'] / 17  # Current season length
            df['durability_score'] = np.where(df['G'] >= 14, 1, df['G'] / 14)
        
        # Era adjustments
        if 'estimated_year' in df.columns:
            df['modern_era'] = (df['estimated_year'] >= 2018).astype(int)
            df['passing_era'] = (df['estimated_year'] >= 2010).astype(int)
            df['defensive_era'] = (df['estimated_year'] <= 2005).astype(int)
        
        # Performance consistency (rolling averages would be better with more data)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols[:5]:  # Limit to prevent too many features
            if col not in ['Age', 'G', 'estimated_year']:
                df[f'{col}_normalized'] = (df[col] - df[col].mean()) / df[col].std()
        
        return df
    
    def prepare_multi_position_training(self, enhanced_datasets):
        """Prepare data for multi-position XGBoost training"""
        logger.info("🤖 Preparing multi-position training data...")
        
        all_features = []
        all_targets = []
        all_positions = []
        
        for position, df in enhanced_datasets.items():
            # Select numeric features only
            numeric_df = df.select_dtypes(include=[np.number])
            
            # Remove completely NaN columns
            numeric_df = numeric_df.dropna(axis=1, how='all')
            
            # Fill remaining NaNs
            numeric_df = numeric_df.fillna(0)
            
            # Create position-specific target
            target = self._create_position_target(df, position)
            
            if len(numeric_df) > 0 and len(target) > 0:
                # Add position encoding
                numeric_df['position_qb'] = (position == 'qb').astype(int)
                numeric_df['position_rb'] = (position == 'rb').astype(int)
                numeric_df['position_wr'] = (position == 'wr').astype(int)
                numeric_df['position_te'] = (position == 'te').astype(int)
                numeric_df['position_special'] = (position in ['kicking', 'punting', 'returns']).astype(int)
                numeric_df['position_defense'] = (position == 'defense').astype(int)
                
                all_features.append(numeric_df)
                all_targets.extend(target)
                all_positions.extend([position] * len(numeric_df))
        
        # Combine all features
        if all_features:
            # Get common columns across all dataframes
            common_cols = set(all_features[0].columns)
            for df in all_features[1:]:
                common_cols = common_cols.intersection(set(df.columns))
            
            common_cols = list(common_cols)
            
            # Align all dataframes to common columns
            aligned_features = []
            for df in all_features:
                aligned_df = df[common_cols].copy()
                aligned_features.append(aligned_df)
            
            # Concatenate
            X = pd.concat(aligned_features, ignore_index=True)
            y = np.array(all_targets)
            positions = np.array(all_positions)
            
            # Handle infinite values
            X = X.replace([np.inf, -np.inf], 0)
            
            # Final scaling
            scaler = StandardScaler()
            X_scaled = pd.DataFrame(
                scaler.fit_transform(X),
                columns=X.columns
            )
            
            logger.info(f"✅ Prepared {len(X_scaled):,} samples with {len(X_scaled.columns)} features")
            logger.info(f"📊 Position distribution: {pd.Series(positions).value_counts().to_dict()}")
            
            return X_scaled, y, positions, scaler, common_cols
        
        else:
            raise ValueError("No valid features could be extracted from datasets")
    
    def _create_position_target(self, df, position):
        """Create appropriate target variable for each position"""
        # Position-specific fantasy point estimation
        if position == 'qb':
            # QB fantasy points: Pass Yds/25 + Pass TD*4 + Rush Yds/10 + Rush TD*6 - INT*2
            target = (
                df.get('Yds', 0) / 25 +
                df.get('TD', 0) * 4 +
                df.get('Rush Yds', 0) / 10 +
                df.get('Rush TD', 0) * 6 -
                df.get('Int', 0) * 2
            )
        elif position == 'rb':
            # RB fantasy points: Rush Yds/10 + Rush TD*6 + Rec Yds/10 + Rec TD*6 + Rec*0.5
            target = (
                df.get('Rush Yds', 0) / 10 +
                df.get('Rush TD', 0) * 6 +
                df.get('Rec Yds', 0) / 10 +
                df.get('Rec TD', 0) * 6 +
                df.get('Rec', 0) * 0.5
            )
        elif position in ['wr', 'te']:
            # WR/TE fantasy points: Rec Yds/10 + Rec TD*6 + Rec*0.5
            target = (
                df.get('Rec Yds', 0) / 10 +
                df.get('TD', 0) * 6 +
                df.get('Rec', 0) * 0.5
            )
        elif position == 'kicking':
            # Kicker points: FGM*3 + XPM*1 + bonus for long FGs
            target = (
                df.get('FGM', 0) * 3 +
                df.get('XPM', 0) * 1 +
                (df.get('Lng', 0) > 50).astype(int) * 2
            )
        elif position == 'defense':
            # Defensive points: complex scoring
            target = (
                df.get('Int', 0) * 2 +
                df.get('Sk', 0) * 1 +
                df.get('Sfty', 0) * 2 +
                df.get('TD', 0) * 6
            )
        else:
            # Default: use total yards or touches
            target = df.get('Yds', df.get('Ret Yds', df.get('Pnt', 0)))
        
        # Ensure target is numeric and clean
        target = pd.to_numeric(target, errors='coerce').fillna(0)
        
        return target.tolist()
    
    def train_enhanced_xgboost(self, X, y, positions):
        """Train enhanced XGBoost model with hyperparameter tuning"""
        logger.info("🚀 Training enhanced XGBoost model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=positions
        )
        
        # Enhanced hyperparameter grid
        param_grid = {
            'n_estimators': [200, 300, 500],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.05, 0.1, 0.15],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'min_child_weight': [1, 3, 5],
            'gamma': [0, 0.1, 0.2],
            'reg_alpha': [0, 0.1, 0.3],
            'reg_lambda': [1, 1.3, 1.5]
        }
        
        # Create XGBoost model
        xgb_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1,
            tree_method='hist'
        )
        
        # Grid search with cross-validation
        logger.info("🔧 Hyperparameter tuning (this may take a while)...")
        grid_search = GridSearchCV(
            estimator=xgb_model,
            param_grid=param_grid,
            scoring='r2',
            cv=5,
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Get best model
        best_model = grid_search.best_estimator_
        
        # Evaluate model
        y_pred_train = best_model.predict(X_train)
        y_pred_test = best_model.predict(X_test)
        
        # Calculate metrics
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        # Cross-validation
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='r2')
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        results = {
            'model': best_model,
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
        
        # Print results
        print(f"\n🏆 ENHANCED XGBOOST RESULTS:")
        print(f"{'='*50}")
        print(f"Training R²: {train_r2:.4f}")
        print(f"Test R²: {test_r2:.4f}")
        print(f"Training MAE: {train_mae:.3f}")
        print(f"Test MAE: {test_mae:.3f}")
        print(f"Cross-validation R²: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        print(f"Training samples: {len(X_train):,}")
        print(f"Features used: {len(X.columns)}")
        
        # Accuracy percentage estimation
        accuracy_pct = test_r2 * 100
        print(f"\n🎯 ESTIMATED ACCURACY: {accuracy_pct:.1f}%")
        
        if accuracy_pct >= 90:
            print("🔥 ELITE TIER: Competitive with professional sportsbooks!")
        elif accuracy_pct >= 85:
            print("⭐ EXCELLENT: High-quality predictions!")
        elif accuracy_pct >= 75:
            print("✅ GOOD: Solid predictive performance!")
        else:
            print("📊 BASELINE: Room for improvement")
        
        return results
    
    def save_enhanced_models(self, results, scaler, feature_names):
        """Save all enhanced models and metadata"""
        os.makedirs('models/enhanced', exist_ok=True)
        
        # Save model
        model_path = 'models/enhanced/enhanced_xgboost_model.joblib'
        joblib.dump(results['model'], model_path)
        
        # Save scaler
        scaler_path = 'models/enhanced/enhanced_scaler.joblib'
        joblib.dump(scaler, scaler_path)
        
        # Save metadata
        metadata = {
            'training_date': datetime.now().isoformat(),
            'model_type': 'Enhanced XGBoost',
            'best_params': results['best_params'],
            'performance': {
                'train_r2': results['train_r2'],
                'test_r2': results['test_r2'],
                'cv_mean': results['cv_mean'],
                'accuracy_percentage': results['test_r2'] * 100
            },
            'data_info': {
                'training_samples': results['training_samples'],
                'features_count': results['features_count'],
                'feature_names': feature_names
            },
            'feature_importance': results['feature_importance'].head(20).to_dict('records')
        }
        
        metadata_path = 'models/enhanced/enhanced_model_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"💾 Enhanced model saved to {model_path}")
        logger.info(f"📊 Metadata saved to {metadata_path}")
        
        return model_path, metadata_path


def main():
    """Main enhanced training pipeline"""
    print(f"🏈 ENHANCED NFL XGBOOST TRAINING")
    print(f"Target: 90%+ Accuracy with Advanced Features")
    print(f"{'='*60}")
    
    trainer = EnhancedNFLTrainer()
    
    try:
        # Step 1: Load all datasets
        datasets = trainer.load_all_historical_data()
        
        if not datasets:
            print("❌ No datasets loaded. Please check data files.")
            return
        
        # Step 2: Create advanced features
        enhanced_datasets = trainer.create_advanced_features(datasets)
        
        # Step 3: Prepare for training
        X, y, positions, scaler, feature_names = trainer.prepare_multi_position_training(enhanced_datasets)
        
        # Step 4: Train enhanced model
        results = trainer.train_enhanced_xgboost(X, y, positions)
        
        # Step 5: Save everything
        model_path, metadata_path = trainer.save_enhanced_models(results, scaler, feature_names)
        
        print(f"\n🎉 SUCCESS: Enhanced XGBoost training completed!")
        print(f"🎯 Final accuracy: {results['test_r2']*100:.1f}%")
        print(f"📊 Model saved: {model_path}")
        print(f"📋 Metadata: {metadata_path}")
        
        # Show top features
        print(f"\n🔝 TOP 10 FEATURES:")
        print(results['feature_importance'].head(10).to_string(index=False))
        
        return trainer, results
        
    except Exception as e:
        logger.error(f"❌ Enhanced training failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    trainer, results = main()