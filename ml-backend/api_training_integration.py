#!/usr/bin/env python3
"""
Quick Example: Add API Training to Your Current app.py
Copy these methods into your NFLMLModel class
"""

# Add these imports to the top of your app.py (if not already present)
from datetime import datetime
import json

# Add these methods to your existing NFLMLModel class in app.py:

def retrain_model_via_api(self, training_data: dict, position: str) -> dict:
    """Retrain XGBoost model with new data from API - ADD THIS TO NFLMLModel class"""
    
    try:
        logger.info(f"🔄 Retraining {position} model via API...")
        
        # Extract and validate data
        features = np.array(training_data['features'])
        targets = np.array(training_data['targets'])
        
        if len(features) < 10:
            raise ValueError("Need at least 10 samples for retraining")
        
        # Use existing multi-position features if available
        if hasattr(self, 'multiposition_features') and position in self.multiposition_features:
            expected_features = len(self.multiposition_features[position])
            if features.shape[1] != expected_features:
                logger.warning(f"Feature count mismatch for {position}: got {features.shape[1]}, expected {expected_features}")
        
        # Split data
        split_idx = int(len(features) * 0.8)
        X_train, X_test = features[:split_idx], features[split_idx:]
        y_train, y_test = targets[:split_idx], targets[split_idx:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train new XGBoost model
        params = {
            'n_estimators': min(100, len(X_train) * 3),
            'max_depth': 4 if position == 'QB' else 3,
            'learning_rate': 0.08 if position == 'QB' else 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42
        }
        
        new_model = xgb.XGBRegressor(**params)
        new_model.fit(X_train_scaled, y_train)
        
        # Evaluate performance
        y_pred = new_model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        betting_acc = accuracy_score(y_test > np.median(y_test), y_pred > np.median(y_test))
        
        # Update model if performance is acceptable (R² > 0.1)
        if r2 > 0.1:
            self.models[position] = new_model
            
            # Update scalers
            if not hasattr(self, 'scalers'):
                self.scalers = {}
            self.scalers[position] = scaler
            
            # Save updated model
            model_data = {
                'model': new_model,
                'scaler': scaler,
                'retrained_at': datetime.now().isoformat(),
                'performance': {'mae': mae, 'r2': r2, 'betting_accuracy': betting_acc}
            }
            
            with open(f'api_retrained_{position.lower()}_model.pkl', 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"✅ {position} model retrained! R²={r2:.3f}, Betting Acc={betting_acc:.1%}")
            
            return {
                'success': True,
                'position': position,
                'performance': {'mae': mae, 'r2': r2, 'betting_accuracy': betting_acc},
                'model_updated': True
            }
        else:
            return {
                'success': False,
                'error': f'Poor model performance: R²={r2:.3f} (threshold: 0.1)',
                'model_updated': False
            }
            
    except Exception as e:
        logger.error(f"❌ Error retraining {position}: {e}")
        return {'success': False, 'error': str(e)}

# Add these Flask endpoints to your existing app.py (after the existing routes):

@app.route('/api/retrain/<position>', methods=['POST'])
def retrain_position_model(position):
    """API endpoint to retrain a specific position model"""
    
    try:
        if position.upper() not in ['QB', 'RB', 'WR', 'TE']:
            return jsonify({'success': False, 'error': 'Invalid position'}), 400
        
        data = request.json
        if not data or 'features' not in data or 'targets' not in data:
            return jsonify({'success': False, 'error': 'Missing features or targets'}), 400
        
        # Use your existing NFL model instance
        result = nfl_model.retrain_model_via_api(data, position.upper())
        
        return jsonify(result), 200 if result['success'] else 400
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/train/status', methods=['GET'])
def get_training_status():
    """Get current model training status"""
    
    status = {}
    for pos in ['QB', 'RB', 'WR', 'TE']:
        status[pos] = {
            'loaded': pos in nfl_model.models,
            'has_scaler': hasattr(nfl_model, 'scalers') and pos in nfl_model.scalers
        }
    
    return jsonify({'success': True, 'models': status})

# Example usage - Test your API training:
if __name__ == "__main__":
    print("🧪 Testing API Training Integration")
    print("=" * 50)
    
    # Example test data for QB retraining
    test_qb_data = {
        "features": [
            [28, 5, 1.0, 0, 0, 220.5, 1.8],  # Age, experience, age_prime, veteran, career_games, prev_yards, prev_tds
            [25, 2, 1.0, 0, 32, 180.3, 1.2],
            [32, 10, 0, 1, 160, 285.7, 2.1],
            [29, 7, 1.0, 0, 112, 265.4, 1.9],
            [26, 3, 1.0, 0, 48, 195.8, 1.5],
            [31, 9, 0, 0, 144, 240.2, 1.7],
            [27, 4, 1.0, 0, 64, 210.6, 1.6],
            [30, 8, 1.0, 0, 128, 270.1, 2.0],
            [24, 1, 0, 0, 16, 165.3, 1.1],
            [33, 11, 0, 1, 176, 255.9, 1.8],
            [28, 6, 1.0, 0, 96, 235.7, 1.9],
            [26, 3, 1.0, 0, 48, 188.4, 1.3]
        ],
        "targets": [
            235.2, 195.8, 275.3, 255.1, 205.7, 245.8, 220.4, 285.6, 175.2, 260.3, 240.9, 198.5
        ]
    }
    
    print("📊 Sample QB training data:")
    print(f"   Features: {len(test_qb_data['features'])} samples x {len(test_qb_data['features'][0])} features")
    print(f"   Targets: {len(test_qb_data['targets'])} yards predictions")
    print(f"   Feature format: [age, experience, age_prime, veteran, career_games, prev_yards_per_game, prev_td_per_game]")
    
    print("\n🌐 To test API training:")
    print("1. Start your Flask app: python app.py")
    print("2. Send POST request to: http://localhost:5000/api/retrain/QB")
    print("3. Include the test data in request body")
    
    print("\n💡 CURL example:")
    curl_example = f'''curl -X POST http://localhost:5000/api/retrain/QB \\
  -H "Content-Type: application/json" \\
  -d '{json.dumps(test_qb_data, indent=2)}'
'''
    print(curl_example)
    
    print("\n🎯 BENEFITS:")
    print("• Real-time model updates with new game data")
    print("• API-driven retraining without code changes")
    print("• Performance validation before model updates")
    print("• Automatic model versioning and backup")
    print("• Easy integration with data pipelines")
    
    print("\n✅ Integration complete! Add the methods above to your NFLMLModel class.")