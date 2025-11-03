"""
XGBoost Training with 26 Years of Real NFL Data (2000-2025)
Advanced model training using comprehensive historical data
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from historical_data_processor import NFLHistoricalProcessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedNFLModel:
    def __init__(self):
        self.model = None
        self.feature_importance = None
        self.training_history = []
        self.best_params = None
        self.processor = NFLHistoricalProcessor()
        
    def load_historical_data(self):
        """Load and process 26 years of NFL data"""
        logger.info("🏈 Loading 26 years of NFL historical data...")
        
        # Load and process the data
        df = self.processor.load_and_clean_data()
        X, y, feature_names = self.processor.prepare_for_xgboost(df)
        
        self.feature_names = feature_names
        
        logger.info(f"✅ Loaded {len(X)} QB seasons from 2000-2025")
        logger.info(f"📊 Features: {len(feature_names)}")
        
        return X, y, df
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into training and testing sets"""
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    def tune_hyperparameters(self, X_train, y_train):
        """Tune XGBoost hyperparameters using GridSearch"""
        logger.info("🔧 Tuning XGBoost hyperparameters...")
        
        # Parameter grid for tuning
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 4, 5, 6],
            'learning_rate': [0.05, 0.1, 0.15],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'min_child_weight': [1, 3, 5]
        }
        
        # Create XGBoost regressor
        xgb_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1
        )
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            estimator=xgb_model,
            param_grid=param_grid,
            scoring='neg_mean_squared_error',
            cv=5,
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        self.best_params = grid_search.best_params_
        logger.info(f"🎯 Best parameters: {self.best_params}")
        
        return grid_search.best_estimator_
    
    def train_model(self, X_train, y_train, X_val, y_val, use_tuned_params=True):
        """Train the XGBoost model"""
        logger.info("🚀 Training XGBoost model...")
        
        if use_tuned_params and self.best_params:
            # Use tuned parameters
            self.model = xgb.XGBRegressor(
                **self.best_params,
                objective='reg:squarederror',
                random_state=42,
                n_jobs=-1
            )
        else:
            # Use default good parameters
            self.model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.9,
                colsample_bytree=0.9,
                min_child_weight=3,
                objective='reg:squarederror',
                random_state=42,
                n_jobs=-1
            )
        
        # Train with early stopping
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=20,
            verbose=False
        )
        
        # Get feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info(f"✅ Model training completed")
        logger.info(f"🎯 Best iteration: {self.model.best_iteration}")
        
    def evaluate_model(self, X_test, y_test, X_train, y_train):
        """Comprehensive model evaluation"""
        logger.info("📊 Evaluating model performance...")
        
        # Predictions
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        # Metrics
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        # Cross-validation score
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='r2')
        
        evaluation_results = {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'features_used': len(self.feature_names)
        }
        
        # Print results
        print(f"\n🏆 MODEL PERFORMANCE RESULTS:")
        print(f"{'='*50}")
        print(f"Training R²: {train_r2:.4f}")
        print(f"Test R²: {test_r2:.4f}")
        print(f"Training MAE: {train_mae:.3f}")
        print(f"Test MAE: {test_mae:.3f}")
        print(f"Cross-validation R²: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        print(f"Training samples: {len(X_train):,}")
        print(f"Test samples: {len(X_test):,}")
        
        return evaluation_results, y_pred_test
    
    def plot_feature_importance(self, top_n=15):
        """Plot feature importance"""
        plt.figure(figsize=(12, 8))
        top_features = self.feature_importance.head(top_n)
        
        sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')
        plt.title('Top Feature Importance (XGBoost)', fontsize=16, fontweight='bold')
        plt.xlabel('Feature Importance', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.tight_layout()
        plt.savefig('models/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n🎯 TOP {top_n} MOST IMPORTANT FEATURES:")
        print(top_features.to_string(index=False))
    
    def plot_predictions(self, y_test, y_pred, title="Model Predictions vs Actual"):
        """Plot predictions vs actual values"""
        plt.figure(figsize=(10, 8))
        
        # Scatter plot
        plt.scatter(y_test, y_pred, alpha=0.6, color='blue', s=50)
        
        # Perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        # Labels and title
        plt.xlabel('Actual Values', fontsize=12)
        plt.ylabel('Predicted Values', fontsize=12)
        plt.title(title, fontsize=16, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # R² annotation
        r2 = r2_score(y_test, y_pred)
        plt.annotate(f'R² = {r2:.4f}', xy=(0.05, 0.95), xycoords='axes fraction',
                    fontsize=14, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('models/predictions_plot.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, filepath='models/nfl_xgboost_historical.joblib'):
        """Save the trained model and metadata"""
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model
        joblib.dump(self.model, filepath)
        
        # Save metadata
        metadata = {
            'training_date': datetime.now().isoformat(),
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance.to_dict('records'),
            'best_params': self.best_params,
            'model_type': 'XGBoost',
            'data_source': '26 years NFL historical data (2000-2025)'
        }
        
        metadata_path = filepath.replace('.joblib', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"💾 Model saved to {filepath}")
        logger.info(f"📋 Metadata saved to {metadata_path}")
    
    def predict_player_performance(self, player_stats):
        """Predict performance for new player stats"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Convert to DataFrame if needed
        if isinstance(player_stats, dict):
            player_stats = pd.DataFrame([player_stats])
        
        # Make prediction
        prediction = self.model.predict(player_stats)
        return prediction[0] if len(prediction) == 1 else prediction


def main():
    """Main training pipeline"""
    print(f"🏈 NFL XGBOOST TRAINING WITH 26 YEARS OF DATA")
    print(f"{'='*60}")
    
    # Initialize model
    nfl_model = AdvancedNFLModel()
    
    try:
        # Step 1: Load historical data
        X, y, df = nfl_model.load_historical_data()
        
        # Step 2: Split data
        X_train, X_test, y_train, y_test = nfl_model.split_data(X, y)
        X_train, X_val, y_train, y_val = nfl_model.split_data(X_train, y_train, test_size=0.2)
        
        print(f"\n📊 DATA SPLIT:")
        print(f"Training: {len(X_train)} samples")
        print(f"Validation: {len(X_val)} samples")  
        print(f"Test: {len(X_test)} samples")
        
        # Step 3: Hyperparameter tuning (optional - can be slow)
        tune_params = input("\n🔧 Tune hyperparameters? (y/n): ").lower().strip() == 'y'
        
        if tune_params:
            best_model = nfl_model.tune_hyperparameters(X_train, y_train)
        
        # Step 4: Train model
        nfl_model.train_model(X_train, y_train, X_val, y_val, use_tuned_params=tune_params)
        
        # Step 5: Evaluate model
        results, y_pred = nfl_model.evaluate_model(X_test, y_test, X_train, y_train)
        
        # Step 6: Visualizations
        nfl_model.plot_feature_importance()
        nfl_model.plot_predictions(y_test, y_pred)
        
        # Step 7: Save model
        nfl_model.save_model()
        
        print(f"\n🎉 SUCCESS: Advanced XGBoost model trained with 26 years of real NFL data!")
        print(f"🎯 Model accuracy (R²): {results['test_r2']:.4f}")
        print(f"📊 Training samples: {results['training_samples']:,}")
        print(f"🔧 Features used: {results['features_used']}")
        
        return nfl_model, results
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise


if __name__ == "__main__":
    model, results = main()