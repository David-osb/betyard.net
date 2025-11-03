#!/usr/bin/env python3
"""
Multi-Position XGBoost Training Pipeline
Combines QB (2,748 samples) + RB (2,717 samples) for comprehensive NFL training
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class MultiPositionTrainer:
    def __init__(self):
        self.qb_model = None
        self.rb_model = None
        self.scaler_qb = StandardScaler()
        self.scaler_rb = StandardScaler()
        
    def load_qb_data(self):
        """Load and process QB historical data"""
        print("📊 Loading QB data...")
        
        try:
            # Use the existing historical processor for QB data
            from historical_data_processor import HistoricalDataProcessor
            
            qb_processor = HistoricalDataProcessor()
            df = qb_processor.load_and_clean_data('nfl_data/qb_stats_historical.csv')
            
            if df is not None and not df.empty:
                features_df = qb_processor.engineer_features(df)
                X, y = qb_processor.prepare_training_data(features_df)
                print(f"✅ QB data loaded: {X.shape[0]} samples, {X.shape[1]} features")
                return X, y
            else:
                print("❌ Failed to load QB data")
                return None, None
                
        except Exception as e:
            print(f"❌ QB data error: {e}")
            return None, None
    
    def load_rb_data(self):
        """Load and process RB historical data"""
        print("🏃 Loading RB data...")
        
        try:
            # Load the processed RB data
            df = pd.read_csv('processed_rb_data.csv')
            
            if 'target' in df.columns:
                X = df.drop(['target'], axis=1)
                y = df['target']
            else:
                X = df.iloc[:, :-1]
                y = df.iloc[:, -1]
            
            print(f"✅ RB data loaded: {X.shape[0]} samples, {X.shape[1]} features")
            return X, y
            
        except Exception as e:
            print(f"❌ RB data error: {e}")
            return None, None
    
    def train_position_model(self, X, y, position_name, model_params=None):
        """Train XGBoost model for a specific position"""
        print(f"\n🎯 Training {position_name} model...")
        
        if X is None or y is None:
            print(f"❌ No data available for {position_name}")
            return None
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=None
        )
        
        # Scale features
        if position_name == 'QB':
            X_train_scaled = self.scaler_qb.fit_transform(X_train)
            X_test_scaled = self.scaler_qb.transform(X_test)
        else:
            X_train_scaled = self.scaler_rb.fit_transform(X_train)
            X_test_scaled = self.scaler_rb.transform(X_test)
        
        # Default model parameters
        if model_params is None:
            model_params = {
                'n_estimators': 200,
                'max_depth': 8,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'n_jobs': -1
            }
        
        # Train model
        model = xgb.XGBRegressor(**model_params)
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        test_mse = mean_squared_error(y_test, y_pred_test)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        print(f"📈 {position_name} Model Performance:")
        print(f"   Training R²: {train_r2:.4f}")
        print(f"   Test R²: {test_r2:.4f}")
        print(f"   Test MSE: {test_mse:.4f}")
        print(f"   Test MAE: {test_mae:.4f}")
        print(f"   Training samples: {len(X_train)}")
        print(f"   Test samples: {len(X_test)}")
        
        # Feature importance
        importance = model.feature_importances_
        feature_names = X.columns.tolist()
        feature_importance = list(zip(feature_names, importance))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        print(f"🔍 Top {position_name} Features:")
        for feature, imp in feature_importance[:5]:
            print(f"   {feature}: {imp:.4f}")
        
        return model
    
    def train_all_models(self):
        """Train models for all available positions"""
        print("=== Multi-Position XGBoost Training ===")
        print("Training comprehensive NFL prediction models...")
        
        # Load and train QB model
        X_qb, y_qb = self.load_qb_data()
        self.qb_model = self.train_position_model(X_qb, y_qb, "QB")
        
        # Load and train RB model  
        X_rb, y_rb = self.load_rb_data()
        self.rb_model = self.train_position_model(X_rb, y_rb, "RB")
        
        # Summary
        print(f"\n🎊 === Training Complete ===")
        
        models_trained = []
        if self.qb_model is not None:
            models_trained.append("QB")
        if self.rb_model is not None:
            models_trained.append("RB")
            
        print(f"✅ Models trained: {', '.join(models_trained)}")
        print(f"📊 Total positions: {len(models_trained)}")
        
        if len(models_trained) > 0:
            print(f"🎯 Ready for multi-position NFL predictions!")
        
        return models_trained
    
    def save_models(self):
        """Save all trained models"""
        saved_models = []
        
        if self.qb_model is not None:
            qb_path = 'qb_xgboost_model.json'
            self.qb_model.save_model(qb_path)
            saved_models.append(qb_path)
            print(f"💾 QB model saved: {qb_path}")
        
        if self.rb_model is not None:
            rb_path = 'rb_xgboost_model.json'
            self.rb_model.save_model(rb_path)
            saved_models.append(rb_path)
            print(f"💾 RB model saved: {rb_path}")
        
        return saved_models
    
    def make_prediction(self, position, features_dict):
        """Make prediction for a specific position"""
        if position.upper() == 'QB' and self.qb_model is not None:
            # Convert features to array and scale
            features_array = np.array(list(features_dict.values())).reshape(1, -1)
            features_scaled = self.scaler_qb.transform(features_array)
            prediction = self.qb_model.predict(features_scaled)[0]
            return prediction
            
        elif position.upper() == 'RB' and self.rb_model is not None:
            # Convert features to array and scale
            features_array = np.array(list(features_dict.values())).reshape(1, -1)
            features_scaled = self.scaler_rb.transform(features_array)
            prediction = self.rb_model.predict(features_scaled)[0]
            return prediction
        
        else:
            print(f"❌ No trained model available for {position}")
            return None

def main():
    """Main training pipeline"""
    print("🚀 Starting Multi-Position NFL Training Pipeline...")
    
    trainer = MultiPositionTrainer()
    
    # Train all available models
    trained_positions = trainer.train_all_models()
    
    # Save models
    if trained_positions:
        saved_models = trainer.save_models()
        
        print(f"\n🎉 === Pipeline Complete ===")
        print(f"✅ Positions trained: {len(trained_positions)}")
        print(f"💾 Models saved: {len(saved_models)}")
        print(f"📈 NFL prediction system ready!")
        
        # Example prediction demo
        if 'RB' in trained_positions:
            print(f"\n🧪 Testing RB prediction...")
            sample_rb_features = {
                'attempts_per_game': 15.0,
                'yards_per_attempt': 4.2,
                'total_attempts': 200,
                'total_yards': 840,
                'age': 25,
                'games_played': 16,
                'efficiency_score': 67.2,
                'volume_score': 3200,
                'team_encoded': 15,
                'yards_before_contact': 2.1,
                'yards_after_contact': 2.1
            }
            
            try:
                prediction = trainer.make_prediction('RB', sample_rb_features)
                if prediction is not None:
                    print(f"🎯 Sample RB prediction: {prediction:.2f} yards/game")
            except Exception as e:
                print(f"❌ Prediction test failed: {e}")
        
    else:
        print("❌ No models were successfully trained")
    
    return trainer

if __name__ == "__main__":
    main()