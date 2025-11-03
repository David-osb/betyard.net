#!/usr/bin/env python3
"""
Add API Training Endpoints to Your Existing NFL App
This adds training capabilities to your current app.py
"""

from flask import request, jsonify
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
import pickle
import logging
import json
from datetime import datetime

# Set up logging
logger = logging.getLogger(__name__)

# Add these methods to your existing NFLMLModel class in app.py

class APITrainingMixin:
    """Mixin to add API training capabilities to NFLMLModel"""
    
    def retrain_model_via_api(self, training_data: dict, position: str) -> dict:
        """Retrain a specific position model with new data from API"""
        
        try:
            logger.info(f"🔄 Retraining {position} model via API with {len(training_data.get('features', []))} new samples...")
            
            # Validate input data
            features = np.array(training_data['features'])
            targets = np.array(training_data['targets'])
            
            if len(features) != len(targets):
                raise ValueError("Features and targets must have same length")
            
            if len(features) < 10:
                raise ValueError("Need at least 10 samples for retraining")
            
            # Prepare data
            X = features
            y = targets
            
            # Split for validation (80/20)
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Configure XGBoost based on position and data size
            if position == 'QB':
                params = {'max_depth': 4, 'learning_rate': 0.08, 'n_estimators': min(150, len(X_train) * 2)}
            elif position == 'RB':
                params = {'max_depth': 3, 'learning_rate': 0.1, 'n_estimators': min(120, len(X_train) * 2)}
            elif position in ['WR', 'TE']:
                params = {'max_depth': 3, 'learning_rate': 0.09, 'n_estimators': min(140, len(X_train) * 2)}
            
            # Train new model
            new_model = xgb.XGBRegressor(
                **params,
                reg_alpha=0.1,
                reg_lambda=0.1,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=42
            )
            
            new_model.fit(X_train_scaled, y_train)
            
            # Validate performance
            y_pred = new_model.predict(X_test_scaled)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Betting accuracy
            line = np.median(y_test)
            betting_acc = accuracy_score(y_test > line, y_pred > line)
            
            # Get old model performance for comparison
            old_performance = None
            if position in self.models:
                try:
                    old_pred = self.models[position].predict(X_test_scaled)
                    old_r2 = r2_score(y_test, old_pred)
                    old_betting_acc = accuracy_score(y_test > line, old_pred > line)
                    old_performance = {'r2': old_r2, 'betting_accuracy': old_betting_acc}
                except:
                    old_performance = None
            
            # Update model if performance improved or no existing model
            performance_improved = (
                old_performance is None or 
                r2 > old_performance.get('r2', -1) or
                betting_acc > old_performance.get('betting_accuracy', 0)
            )
            
            if performance_improved:
                # Update the model
                self.models[position] = new_model
                
                # Update scaler
                if not hasattr(self, 'scalers'):
                    self.scalers = {}
                self.scalers[position] = scaler
                
                # Save updated model
                self._save_retrained_model(position, new_model, scaler, {
                    'mae': mae,
                    'r2': r2,
                    'betting_accuracy': betting_acc,
                    'training_samples': len(X_train),
                    'test_samples': len(X_test),
                    'retrained_at': datetime.now().isoformat()
                })
                
                logger.info(f"✅ {position} model updated! New performance: R²={r2:.3f}, Betting Acc={betting_acc:.1%}")
                
                return {
                    'success': True,
                    'position': position,
                    'performance': {
                        'mae': mae,
                        'r2': r2,
                        'betting_accuracy': betting_acc,
                        'training_samples': len(X_train),
                        'test_samples': len(X_test)
                    },
                    'improvement': {
                        'r2_change': r2 - old_performance.get('r2', 0) if old_performance else r2,
                        'betting_acc_change': betting_acc - old_performance.get('betting_accuracy', 0) if old_performance else betting_acc
                    },
                    'model_updated': True
                }
            else:
                logger.info(f"⚠️ {position} model not updated - performance didn't improve")
                return {
                    'success': True,
                    'position': position,
                    'performance': {
                        'mae': mae,
                        'r2': r2,
                        'betting_accuracy': betting_acc
                    },
                    'model_updated': False,
                    'reason': 'Performance did not improve over existing model'
                }
                
        except Exception as e:
            logger.error(f"❌ Error retraining {position} model: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _save_retrained_model(self, position: str, model, scaler, performance: dict):
        """Save retrained model to disk"""
        
        model_data = {
            'model': model,
            'scaler': scaler,
            'performance': performance,
            'position': position,
            'model_type': f'API-retrained {position} XGBoost',
            'retrained_at': datetime.now().isoformat()
        }
        
        filename = f'retrained_{position.lower()}_model.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"💾 Retrained {position} model saved to {filename}")
    
    def bulk_retrain_from_recent_games(self, weeks: int = 4):
        """Retrain all models using recent game data (simulated)"""
        
        logger.info(f"🔄 Bulk retraining all models with data from last {weeks} weeks...")
        
        results = {}
        
        # Simulate recent game data for each position
        positions = ['QB', 'RB', 'WR', 'TE']
        
        for position in positions:
            # Generate simulated recent performance data
            n_samples = np.random.randint(20, 50)  # 20-50 recent performances
            
            if position == 'QB':
                # QB features: age, experience, prev_yards_per_game, etc.
                features = []
                targets = []
                
                for _ in range(n_samples):
                    age = np.random.randint(22, 40)
                    exp = max(0, age - 22)
                    prev_yards = np.random.normal(250, 60)
                    
                    feature_vector = [age, exp, prev_yards, 1 if age >= 32 else 0]
                    target_yards = prev_yards + np.random.normal(0, 30)  # Current game
                    
                    features.append(feature_vector)
                    targets.append(max(0, target_yards))
                
            elif position == 'RB':
                # RB features
                features = []
                targets = []
                
                for _ in range(n_samples):
                    age = np.random.randint(21, 35)
                    exp = max(0, age - 21)
                    prev_yards = np.random.normal(70, 25)
                    
                    feature_vector = [age, exp, prev_yards, 1 if age >= 30 else 0]
                    target_yards = prev_yards + np.random.normal(0, 20)
                    
                    features.append(feature_vector)
                    targets.append(max(0, target_yards))
                    
            elif position == 'WR':
                # WR features
                features = []
                targets = []
                
                for _ in range(n_samples):
                    age = np.random.randint(21, 36)
                    exp = max(0, age - 21)
                    prev_yards = np.random.normal(60, 20)
                    
                    feature_vector = [age, exp, prev_yards, 1 if age >= 32 else 0]
                    target_yards = prev_yards + np.random.normal(0, 15)
                    
                    features.append(feature_vector)
                    targets.append(max(0, target_yards))
                    
            elif position == 'TE':
                # TE features
                features = []
                targets = []
                
                for _ in range(n_samples):
                    age = np.random.randint(22, 37)
                    exp = max(0, age - 22)
                    prev_yards = np.random.normal(35, 15)
                    
                    feature_vector = [age, exp, prev_yards, 1 if age >= 33 else 0]
                    target_yards = prev_yards + np.random.normal(0, 10)
                    
                    features.append(feature_vector)
                    targets.append(max(0, target_yards))
            
            # Retrain the position model
            training_data = {
                'features': features,
                'targets': targets
            }
            
            result = self.retrain_model_via_api(training_data, position)
            results[position] = result
        
        logger.info(f"🎯 Bulk retraining complete!")
        return results

# Add these Flask endpoints to your existing app.py

def add_training_endpoints_to_app(app, nfl_model):
    """Add training endpoints to existing Flask app"""
    
    @app.route('/api/retrain/<position>', methods=['POST'])
    def retrain_position_model(position):
        """API endpoint to retrain a specific position model"""
        
        try:
            # Validate position
            if position.upper() not in ['QB', 'RB', 'WR', 'TE']:
                return jsonify({
                    'success': False,
                    'error': 'Invalid position. Must be QB, RB, WR, or TE'
                }), 400
            
            # Get training data from request
            data = request.json
            
            if not data or 'features' not in data or 'targets' not in data:
                return jsonify({
                    'success': False,
                    'error': 'Missing training data. Provide features and targets arrays.'
                }), 400
            
            # Retrain the model
            result = nfl_model.retrain_model_via_api(data, position.upper())
            
            if result['success']:
                return jsonify(result), 200
            else:
                return jsonify(result), 400
                
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/retrain/bulk', methods=['POST'])
    def bulk_retrain_models():
        """Retrain all models with recent data"""
        
        try:
            data = request.json or {}
            weeks = data.get('weeks', 4)
            
            results = nfl_model.bulk_retrain_from_recent_games(weeks)
            
            return jsonify({
                'success': True,
                'results': results,
                'summary': {
                    'positions_retrained': len(results),
                    'models_updated': sum(1 for r in results.values() if r.get('model_updated', False))
                }
            }), 200
            
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/models/status', methods=['GET'])
    def get_model_status():
        """Get status of all trained models"""
        
        model_status = {}
        
        for position in ['QB', 'RB', 'WR', 'TE']:
            if position in nfl_model.models:
                model_status[position] = {
                    'loaded': True,
                    'type': 'XGBoost',
                    'has_scaler': hasattr(nfl_model, 'scalers') and position in nfl_model.scalers
                }
            else:
                model_status[position] = {
                    'loaded': False
                }
        
        return jsonify({
            'success': True,
            'models': model_status,
            'total_loaded': sum(1 for status in model_status.values() if status['loaded'])
        })

if __name__ == "__main__":
    print("🔧 API Training Integration Example")
    print("=" * 50)
    
    print("\n📋 To add API training to your existing app.py:")
    print("1. Add the APITrainingMixin to your NFLMLModel class")
    print("2. Add the training endpoints to your Flask app")
    print("3. Test the endpoints with sample data")
    
    print("\n🌐 Example API calls:")
    print("POST /api/retrain/QB - Retrain QB model")
    print("POST /api/retrain/bulk - Retrain all models")
    print("GET /api/models/status - Check model status")
    
    print("\n📊 Example training data format:")
    example_data = {
        "features": [
            [28, 5, 220.5, 0],  # age, experience, prev_yards_per_game, veteran
            [25, 2, 180.3, 0],
            [32, 10, 285.7, 1]
        ],
        "targets": [
            235.2,  # actual yards for each player
            195.8,
            275.3
        ]
    }
    print(json.dumps(example_data, indent=2))
    
    print("\n✅ Ready to integrate API training into your NFL app!")