#!/usr/bin/env python3
"""
Complete API Training Example - Standalone Demo
Shows how to train XGBoost models via API endpoints
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
import pickle
import json
from datetime import datetime
from flask import Flask, request, jsonify
import logging

# Configure logging
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

# Create Flask app
app = Flask(__name__)
nfl_trainer = APITrainableXGBoost()

@app.route('/train/<position>', methods=['POST'])
def train_model(position):
    """API endpoint to train XGBoost model"""
    
    try:
        if position.upper() not in ['QB', 'RB', 'WR', 'TE']:
            return jsonify({'success': False, 'error': 'Invalid position'}), 400
        
        data = request.json
        if not data or 'features' not in data or 'targets' not in data:
            return jsonify({'success': False, 'error': 'Missing features or targets'}), 400
        
        result = nfl_trainer.train_via_api(data, position.upper())
        
        return jsonify(result), 200 if result['success'] else 400
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/predict/<position>', methods=['POST'])
def predict_performance(position):
    """API endpoint to make predictions"""
    
    try:
        data = request.json
        if not data or 'features' not in data:
            return jsonify({'success': False, 'error': 'Missing features'}), 400
        
        result = nfl_trainer.predict(position.upper(), data['features'])
        
        return jsonify(result), 200 if result['success'] else 400
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/models', methods=['GET'])
def get_models():
    """Get information about trained models"""
    
    model_info = {
        'available_models': list(nfl_trainer.models.keys()),
        'training_history': nfl_trainer.training_history,
        'total_models': len(nfl_trainer.models)
    }
    
    return jsonify(model_info)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'models_loaded': len(nfl_trainer.models)})

if __name__ == "__main__":
    print("🚀 API Training Demo for XGBoost Models")
    print("=" * 50)
    
    # Example: Train a QB model programmatically
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
            [26, 3, 188.4, 1.3, 1]
        ],
        "targets": [
            235.2, 195.8, 275.3, 255.1, 205.7, 245.8, 
            220.4, 285.6, 175.2, 260.3, 240.9, 198.5
        ]
    }
    
    # Train the model
    result = nfl_trainer.train_via_api(sample_qb_data, 'QB')
    print(f"Training result: {json.dumps(result, indent=2)}")
    
    # Test prediction
    if result['success']:
        print("\n2️⃣ Testing prediction...")
        test_features = [29, 6, 240.0, 1.8, 1]  # 29yr old, 6yr exp, 240 yards avg, 1.8 TDs, prime age
        pred_result = nfl_trainer.predict('QB', test_features)
        print(f"Prediction result: {json.dumps(pred_result, indent=2)}")
    
    print("\n3️⃣ API Endpoints Available:")
    print("POST /train/<position> - Train model with data")
    print("POST /predict/<position> - Make predictions")
    print("GET /models - Get model information")
    print("GET /health - Health check")
    
    print("\n📊 Sample API Call (cURL):")
    curl_example = f'''
curl -X POST http://localhost:5000/train/QB \\
  -H "Content-Type: application/json" \\
  -d '{json.dumps(sample_qb_data, indent=2)}'
'''
    print(curl_example)
    
    print("\n🎯 BENEFITS of API Training:")
    print("• Train models with fresh data via HTTP requests")
    print("• No need to restart application for model updates")
    print("• Easy integration with data pipelines")
    print("• Real-time performance validation")
    print("• Automatic model versioning")
    
    # Uncomment to start Flask server
    print("\n🌐 Starting Flask API server on http://localhost:5000...")
    print("Press Ctrl+C to stop")
    app.run(debug=True, port=5000)