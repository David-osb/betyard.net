#!/usr/bin/env python3
"""
Train NFL prediction models with REAL NFL data
Uses collected CSV files from nfl_data_py
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import logging
import json
from typing import Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealDataTrainer:
    """Train models on real NFL statistics"""
    
    def __init__(self):
        self.models = {}
        self.training_scores = {}
    
    def load_qb_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load and prepare QB data"""
        logger.info("📊 Loading QB data...")
        df = pd.read_csv('nfl_qb_data.csv')
        logger.info(f"Loaded {len(df)} QB games from 2020-2024")
        
        # Features for QB model
        features = df[[
            'completions', 'attempts', 'passing_tds', 
            'interceptions', 'sacks', 'passing_air_yards',
            'passing_yards_after_catch', 'passing_first_downs'
        ]].copy()
        
        # Fill NaN values with 0
        features = features.fillna(0)
        
        # Target: passing yards
        target = df['passing_yards'].fillna(0)
        
        # Remove extreme outliers
        mask = (target >= 50) & (target <= 500)
        features = features[mask]
        target = target[mask]
        
        logger.info(f"✅ Prepared {len(features)} QB samples")
        return features, target
    
    def load_rb_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load and prepare RB data"""
        logger.info("📊 Loading RB data...")
        df = pd.read_csv('nfl_rb_data.csv')
        logger.info(f"Loaded {len(df)} RB games from 2020-2024")
        
        # Features for RB model
        features = df[[
            'carries', 'rushing_tds', 'rushing_fumbles',
            'rushing_first_downs', 'receptions', 'targets',
            'receiving_yards', 'receiving_tds'
        ]].copy()
        
        features = features.fillna(0)
        
        # Target: rushing yards
        target = df['rushing_yards'].fillna(0)
        
        # Remove extreme outliers
        mask = (target >= 0) & (target <= 300)
        features = features[mask]
        target = target[mask]
        
        logger.info(f"✅ Prepared {len(features)} RB samples")
        return features, target
    
    def load_wr_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load and prepare WR data"""
        logger.info("📊 Loading WR data...")
        df = pd.read_csv('nfl_wr_data.csv')
        logger.info(f"Loaded {len(df)} WR games from 2020-2024")
        
        # Features for WR model
        features = df[[
            'receptions', 'targets', 'receiving_tds',
            'receiving_fumbles', 'receiving_air_yards',
            'receiving_yards_after_catch', 'receiving_first_downs'
        ]].copy()
        
        features = features.fillna(0)
        
        # Target: receiving yards
        target = df['receiving_yards'].fillna(0)
        
        # Remove extreme outliers
        mask = (target >= 0) & (target <= 250)
        features = features[mask]
        target = target[mask]
        
        logger.info(f"✅ Prepared {len(features)} WR samples")
        return features, target
    
    def load_te_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load and prepare TE data"""
        logger.info("📊 Loading TE data...")
        df = pd.read_csv('nfl_te_data.csv')
        logger.info(f"Loaded {len(df)} TE games from 2020-2024")
        
        # Features for TE model
        features = df[[
            'receptions', 'targets', 'receiving_tds',
            'receiving_fumbles', 'receiving_air_yards',
            'receiving_yards_after_catch', 'receiving_first_downs'
        ]].copy()
        
        features = features.fillna(0)
        
        # Target: receiving yards
        target = df['receiving_yards'].fillna(0)
        
        # Remove extreme outliers
        mask = (target >= 0) & (target <= 200)
        features = features[mask]
        target = target[mask]
        
        logger.info(f"✅ Prepared {len(features)} TE samples")
        return features, target
    
    def train_model(self, X, y, position: str):
        """Train XGBoost model on real data"""
        logger.info(f"🎯 Training {position} model on REAL NFL data...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Optimized hyperparameters for real NFL data
        params = {
            'n_estimators': 300,
            'max_depth': 7,
            'learning_rate': 0.05,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'min_child_weight': 3,
            'gamma': 0.1,
            'reg_alpha': 0.01,
            'reg_lambda': 1,
            'objective': 'reg:squarederror',
            'random_state': 42,
            'n_jobs': -1
        }
        
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
        
        logger.info(f"\n📊 {position} Model Performance (REAL DATA):")
        logger.info(f"Train - MSE: {train_metrics['mse']:.2f}, MAE: {train_metrics['mae']:.2f}, R²: {train_metrics['r2']:.4f}")
        logger.info(f"Test  - MSE: {test_metrics['mse']:.2f}, MAE: {test_metrics['mae']:.2f}, R²: {test_metrics['r2']:.4f}")
        
        self.training_scores[position] = {
            'train': train_metrics,
            'test': test_metrics,
            'samples': len(X),
            'params': params,
            'data_source': 'NFL 2020-2024 Real Stats'
        }
        
        return model
    
    def save_model(self, model, position: str):
        """Save trained model"""
        filename = f'{position.lower()}_model.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"✅ Saved {position} model to {filename}")
    
    def save_report(self):
        """Save training report"""
        report = {
            'training_scores': self.training_scores,
            'timestamp': str(pd.Timestamp.now()),
            'data_source': 'NFL Official Stats 2020-2024',
            'total_samples': sum(s['samples'] for s in self.training_scores.values())
        }
        
        with open('training_report_real_data.json', 'w') as f:
            json.dump(report, f, indent=2)
        logger.info("✅ Saved training report to training_report_real_data.json")


def main():
    print("\n" + "="*60)
    print("🏈 NFL PREDICTION MODEL TRAINING - REAL DATA")
    print("="*60)
    print("📊 Training on actual NFL statistics from 2020-2024")
    print("="*60 + "\n")
    
    trainer = RealDataTrainer()
    
    positions = {
        'QB': trainer.load_qb_data,
        'RB': trainer.load_rb_data,
        'WR': trainer.load_wr_data,
        'TE': trainer.load_te_data
    }
    
    for position, loader in positions.items():
        print(f"\n{'='*60}")
        print(f"Training {position} Model on Real NFL Data")
        print('='*60)
        
        try:
            # Load data
            X, y = loader()
            
            # Train model
            model = trainer.train_model(X, y, position)
            
            # Save model
            trainer.save_model(model, position)
            trainer.models[position] = model
            
        except FileNotFoundError:
            logger.error(f"❌ CSV file not found for {position}")
            logger.info(f"   Run: python collect_nfl_data.py --seasons 2020,2021,2022,2023,2024")
        except Exception as e:
            logger.error(f"❌ Error training {position}: {e}")
    
    # Save report
    trainer.save_report()
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print("\n📊 Training Summary:")
    for pos, scores in trainer.training_scores.items():
        print(f"\n{pos}:")
        print(f"  Samples: {scores['samples']:,}")
        print(f"  Test MAE: {scores['test']['mae']:.2f} yards")
        print(f"  Test R²: {scores['test']['r2']:.4f}")
    
    total = sum(s['samples'] for s in trainer.training_scores.values())
    print(f"\n🎯 Total training samples: {total:,} real NFL games")
    
    print("\n💡 Next steps:")
    print("  1. Review training_report_real_data.json")
    print("  2. Deploy to Render: git add *.pkl && git commit && git push")
    print("  3. Test improved predictions!")


if __name__ == '__main__':
    main()
