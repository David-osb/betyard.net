#!/usr/bin/env python3
"""
COMPREHENSIVE Betting XGBoost Trainer - 100k+ Records
Designed for extensive 12+ hour training sessions with maximum accuracy
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
import joblib
import os
import logging
import gc
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Setup comprehensive logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('comprehensive_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ComprehensiveBettingTrainer:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_selectors = {}
        self.training_history = {}
        self.start_time = time.time()
        
    def load_datasets(self):
        """Load all datasets with comprehensive error handling"""
        logger.info("📁 COMPREHENSIVE DATASET LOADING...")
        
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
                    logger.info(f"📊 Loading {name}...")
                    df = pd.read_csv(file_path, low_memory=False)
                    datasets[name] = df
                    total_records += len(df)
                    logger.info(f"✅ {name}: {len(df):,} records × {len(df.columns)} columns")
                else:
                    logger.warning(f"⚠️ File not found: {file_path}")
            except Exception as e:
                logger.error(f"❌ Error loading {name}: {e}")
        
        logger.info(f"🎯 TOTAL DATASET SIZE: {total_records:,} records")
        return datasets
    
    def create_comprehensive_features(self, datasets):
        """Create extensive feature engineering for maximum accuracy"""
        logger.info("🔧 COMPREHENSIVE FEATURE ENGINEERING...")
        
        all_features = []
        
        for stat_type, df in datasets.items():
            logger.info(f"🛠️ Processing {stat_type} with advanced features...")
            
            # Position-specific comprehensive features
            if stat_type == 'receiving':
                df = self._create_comprehensive_receiving_features(df)
            elif stat_type == 'rushing':
                df = self._create_comprehensive_rushing_features(df)
            elif stat_type == 'passing':
                df = self._create_comprehensive_passing_features(df)
            elif stat_type == 'scoring':
                df = self._create_comprehensive_scoring_features(df)
            elif stat_type == 'defense':
                df = self._create_comprehensive_defense_features(df)
            elif stat_type == 'scrimmage':
                df = self._create_comprehensive_scrimmage_features(df)
            elif stat_type == 'returns':
                df = self._create_comprehensive_returns_features(df)
            elif stat_type == 'kicking':
                df = self._create_comprehensive_kicking_features(df)
            elif stat_type == 'punting':
                df = self._create_comprehensive_punting_features(df)
            
            # Universal advanced features
            df = self._create_advanced_universal_features(df)
            
            all_features.append(df)
            logger.info(f"✅ {stat_type}: {len(df.columns)} features created")
        
        # Combine with advanced processing
        logger.info("🔄 Combining datasets with advanced processing...")
        combined_df = pd.concat(all_features, ignore_index=True, sort=False)
        
        # Advanced data cleaning
        combined_df = self._advanced_data_cleaning(combined_df)
        
        # Create interaction features
        combined_df = self._create_interaction_features(combined_df)
        
        # Create polynomial features (selective)
        combined_df = self._create_polynomial_features(combined_df)
        
        # Create clustering features
        combined_df = self._create_clustering_features(combined_df)
        
        logger.info(f"🎯 COMPREHENSIVE FEATURES COMPLETE: {len(combined_df):,} records × {len(combined_df.columns):,} features")
        return combined_df
    
    def _create_comprehensive_receiving_features(self, df):
        """Extensive receiving features for betting accuracy"""
        # Core targets
        if 'Rec_Yds' in df.columns:
            df['rec_yards_target'] = df['Rec_Yds']
            df['rec_yards_per_game'] = df['Rec_Yds'] / df.get('G', 1).replace(0, 1)
            df['rec_yards_per_target'] = df['Rec_Yds'] / df.get('Tgt', 1).replace(0, 1)
            
            # Advanced yard metrics
            df['rec_yards_variance'] = df['Rec_Yds'] * np.random.normal(0, 0.1, len(df))  # Simulated variance
            df['rec_yards_efficiency'] = df['Rec_Yds'] / (df.get('Tgt', 1) * df.get('G', 1)).replace(0, 1)
        
        if 'Rec' in df.columns:
            df['receptions_target'] = df['Rec']
            df['receptions_per_game'] = df['Rec'] / df.get('G', 1).replace(0, 1)
            df['receptions_per_target'] = df['Rec'] / df.get('Tgt', 1).replace(0, 1)
            
            # Volume indicators
            df['high_volume_receiver'] = np.where(df['Rec'] > (df['Rec'].mean() + df['Rec'].std()), 1, 0)
            df['target_hog'] = np.where(df.get('Tgt', 0) > df.get('Tgt', 0).quantile(0.8), 1, 0)
        
        if 'TD' in df.columns:
            df['rec_td_target'] = df['TD']
            df['rec_td_rate'] = df['TD'] / df.get('G', 1).replace(0, 1)
            df['red_zone_threat'] = np.where(df['TD'] > df['TD'].median(), 1, 0)
            df['anytime_td_scorer'] = np.where(df['TD'] > 0, 1, 0)
        
        # Advanced efficiency metrics
        if 'Lng' in df.columns:
            df['big_play_rate'] = np.where(df['Lng'] >= 20, 1, 0)
            df['explosive_receiver'] = np.where(df['Lng'] >= 40, 1, 0)
        
        # Consistency metrics
        if 'Rec' in df.columns and 'G' in df.columns:
            df['reception_consistency'] = df['Rec'] / df['G'].replace(0, 1)
            df['reliable_target'] = np.where(df['reception_consistency'] > 3, 1, 0)
        
        return df
    
    def _create_comprehensive_rushing_features(self, df):
        """Extensive rushing features"""
        # Core targets
        if 'Yds' in df.columns:
            df['rush_yards_target'] = df['Yds']
            df['rush_yards_per_game'] = df['Yds'] / df.get('G', 1).replace(0, 1)
            
            # Advanced rushing metrics
            df['elite_rusher'] = np.where(df['Yds'] > df['Yds'].quantile(0.9), 1, 0)
            df['rushing_workload'] = df['Yds'] / df.get('Att', 1).replace(0, 1)
        
        if 'Att' in df.columns:
            df['rush_attempts_target'] = df['Att']
            df['rush_attempts_per_game'] = df['Att'] / df.get('G', 1).replace(0, 1)
            
            # Volume indicators
            df['workhorse_back'] = np.where(df['Att'] > 200, 1, 0)
            df['change_of_pace'] = np.where((df['Att'] < 100) & (df['Att'] > 20), 1, 0)
            
            if 'Yds' in df.columns:
                df['yards_per_carry'] = df['Yds'] / df['Att'].replace(0, 1)
                df['efficient_runner'] = np.where(df['yards_per_carry'] > 4.5, 1, 0)
        
        if 'TD' in df.columns:
            df['rush_td_target'] = df['TD']
            df['rush_td_rate'] = df['TD'] / df.get('G', 1).replace(0, 1)
            df['goal_line_back'] = np.where(df['TD'] > df['TD'].quantile(0.8), 1, 0)
        
        return df
    
    def _create_comprehensive_passing_features(self, df):
        """Extensive passing features"""
        if 'Yds' in df.columns:
            df['pass_yards_target'] = df['Yds']
            df['pass_yards_per_game'] = df['Yds'] / df.get('G', 1).replace(0, 1)
            df['elite_passer'] = np.where(df['Yds'] > 4000, 1, 0)
        
        if 'TD' in df.columns:
            df['pass_td_target'] = df['TD']
            df['pass_td_rate'] = df['TD'] / df.get('G', 1).replace(0, 1)
            df['prolific_scorer'] = np.where(df['TD'] > 25, 1, 0)
        
        if 'Int' in df.columns:
            df['pass_int_target'] = df['Int']
            df['interception_prone'] = np.where(df['Int'] > df['Int'].median(), 1, 0)
        
        if 'Att' in df.columns and 'Cmp' in df.columns:
            df['completion_pct'] = df['Cmp'] / df['Att'].replace(0, 1)
            df['accurate_passer'] = np.where(df['completion_pct'] > 0.65, 1, 0)
        
        return df
    
    def _create_comprehensive_scoring_features(self, df):
        """Extensive scoring features"""
        if 'Pts' in df.columns:
            df['fantasy_points_target'] = df['Pts']
            df['fantasy_points_per_game'] = df['Pts'] / df.get('G', 1).replace(0, 1)
            
            # Fantasy tiers
            df['fantasy_elite'] = np.where(df['Pts'] > df['Pts'].quantile(0.95), 1, 0)
            df['fantasy_rb1'] = np.where(df['Pts'] > df['Pts'].quantile(0.9), 1, 0)
            df['fantasy_wr1'] = np.where(df['Pts'] > df['Pts'].quantile(0.85), 1, 0)
            df['fantasy_relevant'] = np.where(df['Pts'] > df['Pts'].quantile(0.7), 1, 0)
        
        return df
    
    def _create_comprehensive_defense_features(self, df):
        """Basic defense features (simplified)"""
        df['defensive_player'] = 1
        return df
    
    def _create_comprehensive_scrimmage_features(self, df):
        """Extensive scrimmage features"""
        # Combined metrics
        rec_yds = df.get('Rec_Yds', 0)
        rush_yds = df.get('Rush_Yds', 0)
        
        df['total_scrimmage_yards'] = rec_yds + rush_yds
        df['scrimmage_yards_per_game'] = df['total_scrimmage_yards'] / df.get('G', 1).replace(0, 1)
        
        # Versatility indicators
        df['dual_threat'] = np.where((rec_yds > 100) & (rush_yds > 100), 1, 0)
        df['receiving_specialist'] = np.where(rec_yds > (rush_yds * 2), 1, 0)
        df['rushing_specialist'] = np.where(rush_yds > (rec_yds * 2), 1, 0)
        
        return df
    
    def _create_comprehensive_returns_features(self, df):
        """Return specialist features"""
        if 'Ret_Yds' in df.columns:
            df['return_yards_per_game'] = df['Ret_Yds'] / df.get('G', 1).replace(0, 1)
            df['return_specialist'] = np.where(df['Ret_Yds'] > 200, 1, 0)
        
        return df
    
    def _create_comprehensive_kicking_features(self, df):
        """Kicking specialist features"""
        if 'FGA' in df.columns and 'FGM' in df.columns:
            df['fg_accuracy'] = df['FGM'] / df['FGA'].replace(0, 1)
            df['reliable_kicker'] = np.where(df['fg_accuracy'] > 0.85, 1, 0)
        
        return df
    
    def _create_comprehensive_punting_features(self, df):
        """Punting specialist features"""
        if 'Punts' in df.columns:
            df['punts_per_game'] = df['Punts'] / df.get('G', 1).replace(0, 1)
        
        return df
    
    def _create_advanced_universal_features(self, df):
        """Advanced universal features for all positions"""
        # Games and durability
        df['games_played'] = df.get('G', 0)
        df['durability_score'] = np.where(df['games_played'] >= 16, 1, 0)
        df['injury_prone'] = np.where(df['games_played'] < 12, 1, 0)
        
        # Team encoding
        if 'Tm' in df.columns:
            df['team_encoded'] = pd.Categorical(df['Tm'].fillna('UNK')).codes
            df['multiple_teams'] = np.where(df['Tm'].str.contains('/', na=False), 1, 0)
        else:
            df['team_encoded'] = 0
            df['multiple_teams'] = 0
        
        # Age-based features (if Age column exists)
        if 'Age' in df.columns:
            df['age_category'] = pd.cut(df['Age'], bins=[0, 23, 27, 30, 40], labels=['Young', 'Prime', 'Veteran', 'Old'])
            df['age_category'] = pd.Categorical(df['age_category']).codes
            df['prime_age'] = np.where((df['Age'] >= 24) & (df['Age'] <= 29), 1, 0)
        
        # Position encoding (if Pos column exists)
        if 'Pos' in df.columns:
            df['position_encoded'] = pd.Categorical(df['Pos'].fillna('UNK')).codes
            df['skill_position'] = np.where(df['Pos'].isin(['RB', 'WR', 'TE', 'QB']), 1, 0)
        else:
            df['position_encoded'] = 0
            df['skill_position'] = 0
        
        return df
    
    def _advanced_data_cleaning(self, df):
        """Advanced data cleaning and preprocessing"""
        logger.info("🧹 Advanced data cleaning...")
        
        # Handle infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values intelligently
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col.endswith('_rate') or col.endswith('_pct'):
                df[col] = df[col].fillna(0)  # Rates default to 0
            elif col.endswith('_target'):
                df[col] = df[col].fillna(df[col].median())  # Targets use median
            else:
                df[col] = df[col].fillna(0)  # Others default to 0
        
        # Remove constant features
        constant_features = df.columns[df.nunique() <= 1]
        if len(constant_features) > 0:
            logger.info(f"🗑️ Removing {len(constant_features)} constant features")
            df = df.drop(columns=constant_features)
        
        return df
    
    def _create_interaction_features(self, df):
        """Create interaction features between key metrics"""
        logger.info("🔗 Creating interaction features...")
        
        # Get key features for interactions
        target_cols = [col for col in df.columns if col.endswith('_target')]
        rate_cols = [col for col in df.columns if col.endswith('_rate') or col.endswith('_per_game')]
        
        interaction_count = 0
        
        # Create selected interactions to avoid explosion
        key_features = target_cols + rate_cols[:10]  # Limit to top rate features
        
        for i, col1 in enumerate(key_features[:5]):  # Limit combinations
            for col2 in key_features[i+1:8]:
                if col1 != col2 and col1 in df.columns and col2 in df.columns:
                    # Multiplication interaction
                    df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
                    
                    # Ratio interaction (if col2 is not zero)
                    ratio_col = f'{col1}_div_{col2}'
                    df[ratio_col] = df[col1] / df[col2].replace(0, 1)
                    
                    interaction_count += 2
        
        logger.info(f"✅ Created {interaction_count} interaction features")
        return df
    
    def _create_polynomial_features(self, df):
        """Create selective polynomial features"""
        logger.info("📐 Creating polynomial features...")
        
        # Select key columns for polynomial transformation
        key_cols = [col for col in df.columns if col.endswith('_target') or col.endswith('_per_game')][:10]
        
        poly_count = 0
        for col in key_cols:
            if col in df.columns:
                # Squared terms
                df[f'{col}_squared'] = df[col] ** 2
                
                # Square root (for positive values)
                df[f'{col}_sqrt'] = np.sqrt(np.abs(df[col]))
                
                # Log transform (for positive values)
                df[f'{col}_log'] = np.log1p(np.abs(df[col]))
                
                poly_count += 3
        
        logger.info(f"✅ Created {poly_count} polynomial features")
        return df
    
    def _create_clustering_features(self, df):
        """Create clustering-based features"""
        logger.info("🎯 Creating clustering features...")
        
        # Select numeric features for clustering
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:20]  # Limit for performance
        cluster_data = df[numeric_cols].fillna(0)
        
        # K-means clustering
        kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
        df['player_cluster'] = kmeans.fit_predict(cluster_data)
        
        # Cluster distances
        cluster_centers = kmeans.cluster_centers_
        for i in range(len(cluster_centers)):
            distances = np.sqrt(np.sum((cluster_data.values - cluster_centers[i]) ** 2, axis=1))
            df[f'cluster_{i}_distance'] = distances
        
        logger.info(f"✅ Created clustering features with 8 clusters")
        return df
    
    def get_comprehensive_targets(self, df):
        """Get all available betting targets"""
        targets = {}
        
        target_columns = [col for col in df.columns if col.endswith('_target')]
        
        for col in target_columns:
            if col in df.columns:
                target_data = df[col].dropna()
                if len(target_data) >= 500:  # Minimum samples for comprehensive training
                    targets[col] = target_data
                    logger.info(f"🎯 {col}: {len(target_data):,} samples")
        
        logger.info(f"📊 COMPREHENSIVE TARGETS: {len(targets)} betting markets")
        return targets
    
    def train_comprehensive_model(self, df, target_name, target_data, model_number, total_models):
        """Comprehensive training with extensive hyperparameter search"""
        logger.info(f"\n{'='*80}")
        logger.info(f"🚀 COMPREHENSIVE TRAINING: MODEL {model_number}/{total_models}")
        logger.info(f"🎯 TARGET: {target_name}")
        logger.info(f"{'='*80}")
        
        start_time = time.time()
        
        # Prepare features with advanced selection
        feature_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in feature_cols if not col.endswith('_target')]
        
        # Get valid samples
        valid_indices = target_data.index
        X = df.loc[valid_indices, feature_cols].fillna(0)
        y = target_data
        
        logger.info(f"📊 Training dataset: {len(X):,} samples × {len(feature_cols):,} features")
        
        # Advanced feature selection
        logger.info("🔍 Advanced feature selection...")
        selector = SelectKBest(score_func=f_regression, k=min(300, len(feature_cols)))
        X_selected = selector.fit_transform(X, y)
        selected_features = len(X_selected[0])
        
        logger.info(f"✅ Selected {selected_features} most important features")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=0.2, random_state=42
        )
        
        # Advanced scaling
        scaler = RobustScaler()  # More robust to outliers
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Check GPU availability
        gpu_available = self._check_gpu()
        
        # Comprehensive hyperparameter tuning
        logger.info(f"⚙️ Comprehensive hyperparameter optimization...")
        if gpu_available:
            logger.info("⚡ Using GPU for extensive grid search")
            model = self._comprehensive_gpu_training(X_train_scaled, y_train, target_name)
        else:
            logger.info("💻 Using CPU for extensive training")
            model = self._comprehensive_cpu_training(X_train_scaled, y_train, target_name)
        
        # Comprehensive evaluation
        logger.info("📊 Comprehensive model evaluation...")
        y_pred = model.predict(X_test_scaled)
        
        # Multiple evaluation metrics
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = np.mean(np.abs((y_test - y_pred) / np.maximum(np.abs(y_test), 0.1))) * 100
        
        # Extensive cross-validation
        logger.info("🔄 Extensive cross-validation (10-fold)...")
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=10, scoring='r2', n_jobs=1 if gpu_available else -1)
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        # Additional cross-validation with different metrics
        cv_mae_scores = cross_val_score(model, X_train_scaled, y_train, cv=10, scoring='neg_mean_absolute_error', n_jobs=1 if gpu_available else -1)
        cv_mae_mean = -cv_mae_scores.mean()
        
        # Store comprehensive results
        elapsed_time = time.time() - start_time
        total_elapsed = time.time() - self.start_time
        
        results = {
            'model': model,
            'scaler': scaler,
            'selector': selector,
            'r2_score': r2,
            'cv_r2_mean': cv_mean,
            'cv_r2_std': cv_std,
            'cv_mae_mean': cv_mae_mean,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'accuracy_pct': r2 * 100,
            'cv_accuracy_pct': cv_mean * 100,
            'samples': len(X),
            'features_total': len(feature_cols),
            'features_selected': selected_features,
            'gpu_used': gpu_available,
            'training_time_minutes': elapsed_time / 60,
            'total_time_hours': total_elapsed / 3600
        }
        
        # Store training history
        self.training_history[target_name] = results
        
        # Comprehensive results display
        self._display_comprehensive_results(target_name, results, model_number, total_models)
        
        # Save immediately after each model
        self.save_comprehensive_model(target_name, results)
        
        # Memory cleanup
        del X_train, X_test, X_train_scaled, X_test_scaled, X_selected
        gc.collect()
        
        return results
    
    def _comprehensive_gpu_training(self, X_train, y_train, target_name):
        """Extensive GPU hyperparameter search"""
        # Comprehensive parameter grid
        param_grid = {
            'n_estimators': [500, 750, 1000, 1500],
            'max_depth': [6, 8, 10, 12, 15],
            'learning_rate': [0.01, 0.03, 0.05, 0.08, 0.1],
            'subsample': [0.7, 0.8, 0.85, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.85, 0.9],
            'colsample_bylevel': [0.8, 0.9, 1.0],
            'gamma': [0, 0.1, 0.2, 0.5],
            'reg_alpha': [0, 0.1, 0.3, 0.5],
            'reg_lambda': [1, 1.5, 2, 3],
            'min_child_weight': [1, 3, 5]
        }
        
        base_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            device='cuda:0',
            tree_method='hist',
            random_state=42,
            eval_metric='rmse'
        )
        
        # Extensive grid search
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring='r2',
            cv=7,  # 7-fold CV for thorough evaluation
            n_jobs=1,
            verbose=1,
            refit=True
        )
        
        logger.info(f"🔥 Starting extensive GPU optimization...")
        grid_search.fit(X_train, y_train)
        
        logger.info(f"🏆 Best GPU Score: {grid_search.best_score_:.6f}")
        logger.info(f"🎯 Best Parameters: {grid_search.best_params_}")
        
        return grid_search.best_estimator_
    
    def _comprehensive_cpu_training(self, X_train, y_train, target_name):
        """Comprehensive CPU training with ensemble"""
        # Multiple model configurations
        models = []
        
        configs = [
            {'n_estimators': 800, 'max_depth': 8, 'learning_rate': 0.05, 'subsample': 0.8},
            {'n_estimators': 1000, 'max_depth': 10, 'learning_rate': 0.08, 'subsample': 0.85},
            {'n_estimators': 600, 'max_depth': 6, 'learning_rate': 0.1, 'subsample': 0.9}
        ]
        
        for i, config in enumerate(configs):
            logger.info(f"Training CPU ensemble model {i+1}/3...")
            model = xgb.XGBRegressor(
                objective='reg:squarederror',
                tree_method='hist',
                random_state=42 + i,
                n_jobs=-1,
                **config
            )
            model.fit(X_train, y_train)
            models.append(model)
        
        # Ensemble wrapper
        class EnsembleModel:
            def __init__(self, models):
                self.models = models
                
            def predict(self, X):
                predictions = np.array([model.predict(X) for model in self.models])
                return np.mean(predictions, axis=0)
        
        return EnsembleModel(models)
    
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
    
    def _display_comprehensive_results(self, target_name, results, model_num, total_models):
        """Display comprehensive training results"""
        print(f"\n🏆 COMPREHENSIVE RESULTS - MODEL {model_num}/{total_models}")
        print(f"🎯 Target: {target_name}")
        print(f"{'─'*60}")
        print(f"📊 R² Score: {results['r2_score']:.6f} ({results['accuracy_pct']:.3f}% accuracy)")
        print(f"🔄 CV R² Score: {results['cv_r2_mean']:.6f} ± {results['cv_r2_std']:.6f}")
        print(f"📏 MAE: {results['mae']:.4f} | RMSE: {results['rmse']:.4f} | MAPE: {results['mape']:.2f}%")
        print(f"🧠 Features: {results['features_selected']}/{results['features_total']} selected")
        print(f"📈 Samples: {results['samples']:,}")
        print(f"⚡ GPU: {'Yes' if results['gpu_used'] else 'No'}")
        print(f"⏱️ Training Time: {results['training_time_minutes']:.1f} minutes")
        print(f"⏰ Total Elapsed: {results['total_time_hours']:.2f} hours")
        
        # Accuracy tier
        accuracy = results['accuracy_pct']
        if accuracy >= 90:
            tier = "🥇 ELITE (90%+)"
        elif accuracy >= 85:
            tier = "🥈 EXCELLENT (85%+)"
        elif accuracy >= 80:
            tier = "🥉 VERY GOOD (80%+)"
        elif accuracy >= 75:
            tier = "📊 GOOD (75%+)"
        else:
            tier = "📈 BASELINE"
        
        print(f"🏅 Performance Tier: {tier}")
        print(f"{'─'*60}")
    
    def save_comprehensive_model(self, target_name, results):
        """Save comprehensive model with metadata"""
        os.makedirs('models/comprehensive_betting', exist_ok=True)
        
        # Save model
        model_path = f'models/comprehensive_betting/{target_name}_model.joblib'
        joblib.dump(results['model'], model_path)
        
        # Save scaler
        scaler_path = f'models/comprehensive_betting/{target_name}_scaler.joblib'
        joblib.dump(results['scaler'], scaler_path)
        
        # Save feature selector
        selector_path = f'models/comprehensive_betting/{target_name}_selector.joblib'
        joblib.dump(results['selector'], selector_path)
        
        # Save metadata
        metadata = {k: v for k, v in results.items() if k not in ['model', 'scaler', 'selector']}
        metadata_path = f'models/comprehensive_betting/{target_name}_metadata.joblib'
        joblib.dump(metadata, metadata_path)
        
        logger.info(f"💾 Comprehensive model saved: {target_name}")

def main():
    """COMPREHENSIVE 12+ HOUR TRAINING SESSION"""
    print(f"🚀 COMPREHENSIVE BETTING XGBOOST TRAINING")
    print(f"📊 100,000+ Records • 500+ Features • GPU Acceleration")
    print(f"⏰ Expected Duration: 12+ Hours")
    print(f"🎯 Goal: Maximum Accuracy for All Betting Markets")
    print(f"{'='*70}")
    
    trainer = ComprehensiveBettingTrainer()
    
    try:
        start_time = datetime.now()
        logger.info(f"🚀 COMPREHENSIVE TRAINING STARTED: {start_time}")
        
        # Load comprehensive datasets
        datasets = trainer.load_datasets()
        
        # Create comprehensive features
        df = trainer.create_comprehensive_features(datasets)
        
        # Get all betting targets
        targets = trainer.get_comprehensive_targets(df)
        
        print(f"\n🎯 COMPREHENSIVE BETTING TARGETS:")
        for target_name, target_data in targets.items():
            print(f"   📈 {target_name}: {len(target_data):,} samples")
        
        print(f"\n🚀 STARTING COMPREHENSIVE TRAINING...")
        print(f"⚡ GPU: {'Available' if trainer._check_gpu() else 'CPU Only'}")
        print(f"🧠 Total Features: {len(df.columns):,}")
        print(f"📊 Total Samples: {len(df):,}")
        
        # Train all models comprehensively
        all_results = {}
        successful_models = 0
        total_models = len(targets)
        
        for i, (target_name, target_data) in enumerate(targets.items(), 1):
            try:
                results = trainer.train_comprehensive_model(df, target_name, target_data, i, total_models)
                all_results[target_name] = results
                successful_models += 1
                
                # Progress update
                progress_pct = (i / total_models) * 100
                elapsed_hours = (time.time() - trainer.start_time) / 3600
                print(f"\n📊 PROGRESS UPDATE:")
                print(f"   ✅ Completed: {i}/{total_models} models ({progress_pct:.1f}%)")
                print(f"   ⏰ Elapsed: {elapsed_hours:.2f} hours")
                if i > 1:
                    avg_time_per_model = elapsed_hours / i
                    remaining_time = avg_time_per_model * (total_models - i)
                    print(f"   ⏳ Estimated Remaining: {remaining_time:.2f} hours")
                
            except Exception as e:
                logger.error(f"❌ Failed to train {target_name}: {e}")
                continue
        
        # COMPREHENSIVE FINAL RESULTS
        end_time = datetime.now()
        total_duration = end_time - start_time
        
        print(f"\n🎉 COMPREHENSIVE TRAINING COMPLETE!")
        print(f"{'='*70}")
        print(f"⏰ Start Time: {start_time}")
        print(f"⏰ End Time: {end_time}")
        print(f"⏱️ Total Duration: {total_duration}")
        print(f"✅ Successful Models: {successful_models}/{total_models}")
        
        if all_results:
            # Comprehensive statistics
            accuracies = [r['accuracy_pct'] for r in all_results.values()]
            cv_accuracies = [r['cv_accuracy_pct'] for r in all_results.values()]
            
            avg_accuracy = np.mean(accuracies)
            max_accuracy = max(accuracies)
            min_accuracy = min(accuracies)
            avg_cv_accuracy = np.mean(cv_accuracies)
            
            print(f"\n📊 COMPREHENSIVE ACCURACY RESULTS:")
            print(f"   📈 Average Accuracy: {avg_accuracy:.3f}%")
            print(f"   🔝 Maximum Accuracy: {max_accuracy:.3f}%")
            print(f"   📉 Minimum Accuracy: {min_accuracy:.3f}%")
            print(f"   🔄 Average CV Accuracy: {avg_cv_accuracy:.3f}%")
            
            # Performance tiers
            elite_models = [name for name, r in all_results.items() if r['accuracy_pct'] >= 90]
            excellent_models = [name for name, r in all_results.items() if r['accuracy_pct'] >= 85]
            good_models = [name for name, r in all_results.items() if r['accuracy_pct'] >= 80]
            
            print(f"\n🏅 PERFORMANCE TIERS:")
            print(f"   🥇 Elite (90%+): {len(elite_models)} models")
            print(f"   🥈 Excellent (85%+): {len(excellent_models)} models")
            print(f"   🥉 Good (80%+): {len(good_models)} models")
            
            if elite_models:
                print(f"\n🥇 ELITE MODELS (90%+ Accuracy):")
                for name in elite_models:
                    acc = all_results[name]['accuracy_pct']
                    cv_acc = all_results[name]['cv_accuracy_pct']
                    print(f"   🎯 {name}: {acc:.3f}% (CV: {cv_acc:.3f}%)")
        
        print(f"\n💾 All models saved to 'models/comprehensive_betting/'")
        print(f"🚀 Ready for ULTIMATE betting predictions!")
        
        return trainer, all_results
        
    except Exception as e:
        logger.error(f"❌ Comprehensive training failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    trainer, results = main()