#!/usr/bin/env python3
"""
Enhanced Model Training and Hyperparameter Tuning for NFL Predictions
Uses GridSearchCV to find optimal XGBoost parameters
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import logging
from typing import Tuple, Dict
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NFLModelTrainer:
    """Train and tune XGBoost models for NFL player predictions"""
    
    def __init__(self):
        self.models = {}
        self.best_params = {}
        self.training_scores = {}
    
    def tune_hyperparameters(self, X_train, y_train, position: str, quick_tune: bool = False):
        """
        Find optimal hyperparameters using GridSearchCV
        
        Args:
            quick_tune: If True, use smaller parameter grid for faster tuning
        """
        logger.info(f"🔍 Tuning hyperparameters for {position} model...")
        
        if quick_tune:
            # Quick tune - takes ~5-10 minutes
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.05, 0.1],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0],
                'min_child_weight': [1, 3],
                'gamma': [0, 0.1]
            }
        else:
            # Comprehensive tune - takes 30-60 minutes
            param_grid = {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [3, 4, 5, 6, 7, 8],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
                'min_child_weight': [1, 2, 3, 5],
                'gamma': [0, 0.1, 0.2, 0.5],
                'reg_alpha': [0, 0.01, 0.1],
                'reg_lambda': [0.1, 1, 10]
            }
        
        # Create base model
        base_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1,
            verbosity=1
        )
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=2
        )
        
        logger.info(f"Testing {len(grid_search.get_params()['param_grid'])} parameter combinations...")
        grid_search.fit(X_train, y_train)
        
        logger.info(f"✅ Best parameters for {position}:")
        logger.info(json.dumps(grid_search.best_params_, indent=2))
        logger.info(f"Best CV score: {-grid_search.best_score_:.4f} MSE")
        
        self.best_params[position] = grid_search.best_params_
        
        return grid_search.best_estimator_
    
    def train_with_best_params(self, X_train, y_train, X_test, y_test, position: str, 
                               custom_params: Dict = None):
        """
        Train model with optimized hyperparameters
        """
        logger.info(f"🎯 Training {position} model with best parameters...")
        
        # Use custom params or recommended params
        if custom_params:
            params = custom_params
        elif position in self.best_params:
            params = self.best_params[position]
        else:
            # Recommended starting params (better than defaults)
            params = {
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 3,
                'gamma': 0.1,
                'reg_alpha': 0.01,
                'reg_lambda': 1
            }
        
        # Add fixed params
        params['objective'] = 'reg:squarederror'
        params['random_state'] = 42
        params['n_jobs'] = -1
        
        # XGBoost 3.0+ removed early_stopping_rounds parameter
        # Just train without early stopping for compatibility
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, verbose=False)
        
        # Evaluate
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        train_metrics = {
            'mse': mean_squared_error(y_train, train_pred),
            'mae': mean_absolute_error(y_train, train_pred),
            'r2': r2_score(y_train, train_pred)
        }
        
        test_metrics = {
            'mse': mean_squared_error(y_test, test_pred),
            'mae': mean_absolute_error(y_test, test_pred),
            'r2': r2_score(y_test, test_pred)
        }
        
        logger.info(f"\n📊 {position} Model Performance:")
        logger.info(f"Train - MSE: {train_metrics['mse']:.2f}, MAE: {train_metrics['mae']:.2f}, R²: {train_metrics['r2']:.4f}")
        logger.info(f"Test  - MSE: {test_metrics['mse']:.2f}, MAE: {test_metrics['mae']:.2f}, R²: {test_metrics['r2']:.4f}")
        
        self.training_scores[position] = {
            'train': train_metrics,
            'test': test_metrics,
            'params': params
        }
        
        return model
    
    def generate_better_training_data(self, position: str, n_samples: int = 10000):
        """
        Generate more realistic training data with position-specific features
        TODO: Replace with real NFL data scraping
        """
        logger.info(f"Generating {n_samples} training samples for {position}...")
        
        np.random.seed(42)
        
        if position == 'QB':
            return self._generate_qb_data(n_samples)
        elif position == 'RB':
            return self._generate_rb_data(n_samples)
        elif position == 'WR':
            return self._generate_wr_data(n_samples)
        elif position == 'TE':
            return self._generate_te_data(n_samples)
    
    def _generate_qb_data(self, n: int):
        """Enhanced QB training data with more realistic correlations"""
        # Base features
        recent_form = np.random.beta(8, 2, n)  # Skewed toward good performance
        home_advantage = np.random.choice([0, 1], n, p=[0.44, 0.56])  # Home teams win ~56%
        opponent_defense_rank = np.random.uniform(1, 32, n)
        temperature = np.random.normal(65, 18, n)
        wind_speed = np.clip(np.random.exponential(7, n), 0, 40)
        precipitation = np.random.exponential(0.05, n)
        injury_factor = np.random.choice([1.0, 0.95, 0.85, 0.7], n, p=[0.75, 0.15, 0.07, 0.03])
        experience_years = np.clip(np.random.gamma(2, 3, n), 0, 20)
        season_completion_pct = np.clip(np.random.normal(65, 7, n), 45, 85)
        season_avg_yards = np.clip(np.random.normal(260, 60, n), 150, 450)
        season_avg_tds = np.clip(np.random.gamma(2, 1, n), 0, 5)
        team_offensive_rank = np.random.uniform(1, 32, n)
        
        # Weather impact
        weather_penalty = np.where(temperature < 35, 0.85, 1.0)
        weather_penalty = np.where(wind_speed > 20, weather_penalty * 0.9, weather_penalty)
        weather_penalty = np.where(precipitation > 0.2, weather_penalty * 0.92, weather_penalty)
        
        X = np.column_stack([
            recent_form, home_advantage, opponent_defense_rank,
            temperature, wind_speed, precipitation, injury_factor,
            experience_years, season_completion_pct, season_avg_yards,
            season_avg_tds, team_offensive_rank, weather_penalty
        ])
        
        # More realistic passing yards calculation
        base_yards = (
            season_avg_yards * recent_form * injury_factor * weather_penalty
            + home_advantage * 18
            - (opponent_defense_rank - 16) * 4
            + (team_offensive_rank - 16) * -3
            + experience_years * 2
            + (season_completion_pct - 65) * 1.5
        )
        
        # Add realistic noise
        noise = np.random.normal(0, 35, n)
        y = np.clip(base_yards + noise, 120, 500)
        
        return X, y
    
    def _generate_rb_data(self, n: int):
        """Enhanced RB training data"""
        recent_form = np.random.beta(8, 2, n)
        home_advantage = np.random.choice([0, 1], n, p=[0.44, 0.56])
        opponent_rush_defense_rank = np.random.uniform(1, 32, n)
        temperature = np.random.normal(65, 18, n)
        injury_factor = np.random.choice([1.0, 0.95, 0.85, 0.7], n, p=[0.70, 0.18, 0.09, 0.03])
        experience_years = np.clip(np.random.gamma(1.5, 2, n), 0, 15)
        season_avg_yards = np.clip(np.random.normal(75, 35, n), 20, 180)
        season_avg_attempts = np.clip(np.random.normal(16, 6, n), 5, 30)
        yards_per_carry = np.clip(np.random.normal(4.3, 0.8, n), 2.5, 7)
        team_offensive_rank = np.random.uniform(1, 32, n)
        game_script = np.random.normal(0, 1, n)  # Positive = winning (more rushes)
        
        X = np.column_stack([
            recent_form, home_advantage, opponent_rush_defense_rank,
            temperature, injury_factor, experience_years,
            season_avg_yards, season_avg_attempts, yards_per_carry,
            team_offensive_rank, game_script
        ])
        
        base_yards = (
            season_avg_yards * recent_form * injury_factor
            + home_advantage * 10
            - (opponent_rush_defense_rank - 16) * 2.5
            + game_script * 8  # More rushes when winning
            + yards_per_carry * 5
        )
        
        noise = np.random.normal(0, 25, n)
        y = np.clip(base_yards + noise, 15, 250)
        
        return X, y
    
    def _generate_wr_data(self, n: int):
        """Enhanced WR training data"""
        recent_form = np.random.beta(7, 2, n)
        home_advantage = np.random.choice([0, 1], n, p=[0.44, 0.56])
        opponent_pass_defense_rank = np.random.uniform(1, 32, n)
        temperature = np.random.normal(65, 18, n)
        wind_speed = np.clip(np.random.exponential(7, n), 0, 40)
        injury_factor = np.random.choice([1.0, 0.95, 0.85, 0.7], n, p=[0.72, 0.17, 0.08, 0.03])
        experience_years = np.clip(np.random.gamma(1.5, 2, n), 0, 15)
        season_avg_yards = np.clip(np.random.normal(65, 30, n), 20, 150)
        season_avg_targets = np.clip(np.random.normal(7, 3, n), 2, 18)
        catch_rate = np.clip(np.random.normal(62, 10, n), 35, 85)
        qb_quality = np.random.uniform(0.4, 1.0, n)  # QB rating normalized
        team_pass_volume = np.random.normal(35, 8, n)  # Team pass attempts per game
        
        weather_penalty = np.where(wind_speed > 20, 0.88, 1.0)
        
        X = np.column_stack([
            recent_form, home_advantage, opponent_pass_defense_rank,
            temperature, wind_speed, injury_factor, experience_years,
            season_avg_yards, season_avg_targets, catch_rate,
            qb_quality, team_pass_volume, weather_penalty
        ])
        
        base_yards = (
            season_avg_yards * recent_form * injury_factor * weather_penalty
            + home_advantage * 8
            - (opponent_pass_defense_rank - 16) * 2
            + qb_quality * 25
            + (season_avg_targets - 7) * 3
            + (team_pass_volume - 35) * 0.8
        )
        
        noise = np.random.normal(0, 22, n)
        y = np.clip(base_yards + noise, 10, 200)
        
        return X, y
    
    def _generate_te_data(self, n: int):
        """Enhanced TE training data"""
        recent_form = np.random.beta(7, 2, n)
        home_advantage = np.random.choice([0, 1], n, p=[0.44, 0.56])
        opponent_pass_defense_rank = np.random.uniform(1, 32, n)
        temperature = np.random.normal(65, 18, n)
        wind_speed = np.clip(np.random.exponential(7, n), 0, 40)
        injury_factor = np.random.choice([1.0, 0.95, 0.85, 0.7], n, p=[0.71, 0.18, 0.08, 0.03])
        experience_years = np.clip(np.random.gamma(1.5, 2, n), 0, 15)
        season_avg_yards = np.clip(np.random.normal(45, 25, n), 15, 120)
        season_avg_targets = np.clip(np.random.normal(5, 2, n), 1, 12)
        catch_rate = np.clip(np.random.normal(66, 9, n), 45, 85)  # TEs typically higher catch rate
        qb_quality = np.random.uniform(0.4, 1.0, n)
        red_zone_targets = np.clip(np.random.poisson(1.5, n), 0, 6)
        
        weather_penalty = np.where(wind_speed > 20, 0.90, 1.0)
        
        X = np.column_stack([
            recent_form, home_advantage, opponent_pass_defense_rank,
            temperature, wind_speed, injury_factor, experience_years,
            season_avg_yards, season_avg_targets, catch_rate,
            qb_quality, red_zone_targets, weather_penalty
        ])
        
        base_yards = (
            season_avg_yards * recent_form * injury_factor * weather_penalty
            + home_advantage * 6
            - (opponent_pass_defense_rank - 16) * 1.8
            + qb_quality * 18
            + red_zone_targets * 4
            + (season_avg_targets - 5) * 3.5
        )
        
        noise = np.random.normal(0, 18, n)
        y = np.clip(base_yards + noise, 8, 150)
        
        return X, y
    
    def save_model(self, model, position: str):
        """Save trained model to file"""
        filename = f'{position.lower()}_model.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"✅ Saved {position} model to {filename}")
    
    def save_training_report(self):
        """Save training metrics to JSON"""
        report = {
            'best_params': self.best_params,
            'training_scores': self.training_scores,
            'timestamp': str(pd.Timestamp.now())
        }
        
        with open('training_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        logger.info("✅ Saved training report to training_report.json")


def main():
    """Main training pipeline"""
    print("\n" + "="*60)
    print("🏈 NFL PREDICTION MODEL TRAINING & TUNING")
    print("="*60 + "\n")
    
    trainer = NFLModelTrainer()
    positions = ['QB', 'RB', 'WR', 'TE']
    
    print("Choose training mode:")
    print("1. Quick Train (5-10 min) - Use recommended params, no tuning")
    print("2. Quick Tune (20-40 min) - Find better params with limited search")
    print("3. Full Tune (2-4 hours) - Comprehensive hyperparameter search")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    for position in positions:
        print(f"\n{'='*60}")
        print(f"Training {position} Model")
        print('='*60)
        
        # Generate training data
        X, y = trainer.generate_better_training_data(position, n_samples=10000)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        if choice == '1':
            # Quick train with recommended params
            model = trainer.train_with_best_params(
                X_train, y_train, X_test, y_test, position
            )
        elif choice == '2':
            # Quick tuning
            model = trainer.tune_hyperparameters(
                X_train, y_train, position, quick_tune=True
            )
            model = trainer.train_with_best_params(
                X_train, y_train, X_test, y_test, position
            )
        else:
            # Full tuning
            model = trainer.tune_hyperparameters(
                X_train, y_train, position, quick_tune=False
            )
            model = trainer.train_with_best_params(
                X_train, y_train, X_test, y_test, position
            )
        
        trainer.save_model(model, position)
        trainer.models[position] = model
    
    # Save report
    trainer.save_training_report()
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print("\nModels saved:")
    for pos in positions:
        print(f"  • {pos.lower()}_model.pkl")
    print("\nTraining report saved: training_report.json")
    print("\n💡 Next steps:")
    print("  1. Review training_report.json for model performance")
    print("  2. Upload new .pkl files to Render (they'll auto-deploy)")
    print("  3. Or restart your local server to test improvements")


if __name__ == '__main__':
    main()
