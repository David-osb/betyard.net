#!/usr/bin/env python3
"""
API Training Demo - Just the Training Part (No Flask Server)
"""

import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
import json
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class APITrainableXGBoost:
    """XGBoost model with API training capabilities"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.training_history = []
    
    def train_via_api(self, training_data: dict, position: str) -> dict:
        """Train XGBoost model with data from API"""
        
        try:
            logger.info(f"🚀 Training {position} model via API...")
            
            # Extract data
            features = np.array(training_data['features'])
            targets = np.array(training_data['targets'])
            
            if len(features) < 10:
                raise ValueError("Need at least 10 samples")
            
            logger.info(f"   Data: {len(features)} samples x {features.shape[1]} features")
            
            # Split data
            split_idx = int(len(features) * 0.8)
            X_train, X_test = features[:split_idx], features[split_idx:]
            y_train, y_test = targets[:split_idx], targets[split_idx:]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train XGBoost
            params = {
                'n_estimators': min(100, len(X_train) * 3),
                'max_depth': 4,
                'learning_rate': 0.1,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'random_state': 42
            }
            
            model = xgb.XGBRegressor(**params)
            model.fit(X_train_scaled, y_train)
            
            # Validate
            y_pred = model.predict(X_test_scaled)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            betting_acc = accuracy_score(y_test > np.median(y_test), y_pred > np.median(y_test))
            
            # Store model
            self.models[position] = model
            self.scalers[position] = scaler
            
            # Log training
            training_record = {
                'timestamp': datetime.now().isoformat(),
                'position': position,
                'samples': len(features),
                'mae': round(mae, 2),
                'r2': round(r2, 3),
                'betting_accuracy': round(betting_acc, 3),
                'params': params
            }
            self.training_history.append(training_record)
            
            logger.info(f"✅ {position} model trained successfully!")
            logger.info(f"   Performance: MAE={mae:.2f}, R²={r2:.3f}, Betting Acc={betting_acc:.1%}")
            
            return {
                'success': True,
                'position': position,
                'performance': {
                    'mae': round(mae, 2),
                    'r2': round(r2, 3),
                    'betting_accuracy': round(betting_acc, 3),
                    'training_samples': len(X_train),
                    'test_samples': len(X_test)
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def predict(self, position: str, features: list) -> dict:
        """Make prediction using trained model"""
        
        try:
            if position not in self.models:
                return {'success': False, 'error': f'No model for {position}'}
            
            # Scale features
            features_array = np.array(features).reshape(1, -1)
            features_scaled = self.scalers[position].transform(features_array)
            
            # Predict
            prediction = self.models[position].predict(features_scaled)[0]
            
            return {
                'success': True,
                'position': position,
                'prediction': round(float(prediction), 1),
                'model_type': 'API-trained XGBoost'
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}

if __name__ == "__main__":
    print("🚀 API Training Demo for XGBoost Models")
    print("=" * 50)
    
    # Create trainer
    nfl_trainer = APITrainableXGBoost()
    
    # Example: Train QB model with sample data
    print("\n1️⃣ Training QB model with sample data...")
    
    sample_qb_data = {
        "features": [
            [28, 5, 220.5, 1.8, 1],  # age, experience, prev_yards, prev_tds, prime_age
            [25, 2, 180.3, 1.2, 1],
            [32, 10, 285.7, 2.1, 0],
            [29, 7, 265.4, 1.9, 1],
            [26, 3, 195.8, 1.5, 1],
            [31, 9, 240.2, 1.7, 0],
            [27, 4, 210.6, 1.6, 1],
            [30, 8, 270.1, 2.0, 1],
            [24, 1, 165.3, 1.1, 0],
            [33, 11, 255.9, 1.8, 0],
            [28, 6, 235.7, 1.9, 1],
            [26, 3, 188.4, 1.3, 1],
            [35, 13, 275.2, 2.2, 0],
            [23, 0, 145.8, 0.9, 0],
            [29, 8, 268.3, 2.1, 1]
        ],
        "targets": [
            235.2, 195.8, 275.3, 255.1, 205.7, 245.8, 
            220.4, 285.6, 175.2, 260.3, 240.9, 198.5,
            270.8, 160.2, 265.7
        ]
    }
    
    # Train the QB model
    result = nfl_trainer.train_via_api(sample_qb_data, 'QB')
    print(f"\n📊 QB Training Result:")
    print(json.dumps(result, indent=2))
    
    # Test prediction with QB model
    if result['success']:
        print("\n2️⃣ Testing QB prediction...")
        test_features = [29, 6, 240.0, 1.8, 1]  # 29yr old, 6yr exp, 240 yards avg, 1.8 TDs, prime age
        pred_result = nfl_trainer.predict('QB', test_features)
        print(f"\n🎯 QB Prediction Result:")
        print(json.dumps(pred_result, indent=2))
    
    # Train RB model with different data
    print("\n3️⃣ Training RB model...")
    
    sample_rb_data = {
        "features": [
            [26, 3, 85.5, 0.8, 1],   # age, experience, prev_yards, prev_tds, prime_age
            [23, 1, 65.2, 0.5, 1],
            [29, 7, 95.8, 1.2, 1],
            [25, 2, 72.4, 0.7, 1],
            [31, 9, 78.6, 0.9, 0],
            [24, 1, 58.3, 0.4, 1],
            [28, 6, 102.7, 1.4, 1],
            [30, 8, 88.9, 1.1, 1],
            [27, 4, 91.2, 1.0, 1],
            [32, 10, 76.8, 0.8, 0],
            [25, 2, 69.5, 0.6, 1],
            [26, 3, 87.1, 0.9, 1]
        ],
        "targets": [
            92.3, 71.8, 108.5, 78.2, 82.1, 64.7,
            115.2, 95.6, 98.8, 81.4, 73.9, 94.3
        ]
    }
    
    rb_result = nfl_trainer.train_via_api(sample_rb_data, 'RB')
    print(f"\n📊 RB Training Result:")
    print(json.dumps(rb_result, indent=2))
    
    # Test RB prediction
    if rb_result['success']:
        print("\n4️⃣ Testing RB prediction...")
        test_rb_features = [27, 4, 85.0, 0.9, 1]  # 27yr old, 4yr exp, 85 yards avg, 0.9 TDs, prime age
        rb_pred_result = nfl_trainer.predict('RB', test_rb_features)
        print(f"\n🎯 RB Prediction Result:")
        print(json.dumps(rb_pred_result, indent=2))
    
    # Show training history
    print("\n5️⃣ Training History:")
    for i, record in enumerate(nfl_trainer.training_history, 1):
        print(f"   {i}. {record['position']}: R²={record['r2']}, Betting Acc={record['betting_accuracy']:.1%}")
    
    print("\n🎯 API Training Benefits:")
    print("✅ Train models with HTTP requests")
    print("✅ Real-time model updates")
    print("✅ Performance validation")
    print("✅ Easy pipeline integration")
    print("✅ No application restarts needed")
    
    print("\n🌐 Your Next Steps:")
    print("1. Add retrain_model_via_api() method to your NFLMLModel class")
    print("2. Add Flask endpoints: /api/retrain/<position>")
    print("3. Test with real NFL data from your CSV files")
    print("4. Set up automated retraining with fresh game data")
    
    print(f"\n✅ Demo complete! Trained {len(nfl_trainer.models)} models via API.")