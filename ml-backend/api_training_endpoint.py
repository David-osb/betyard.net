#!/usr/bin/env python3
"""
XGBoost Model Training via API Endpoints
Demonstrates multiple approaches to training models with API data
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import requests
import json
import pickle
import os
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
from flask import Flask, request, jsonify
import asyncio
import aiohttp
from typing import Dict, List, Optional

# ============================================================================
# APPROACH 1: Training Endpoint (Accept data via API to train models)
# ============================================================================

class APITrainingService:
    """Service for training XGBoost models via API endpoints"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.training_history = []
    
    def train_model_from_api_data(self, training_data: Dict, model_config: Dict = None):
        """Train XGBoost model from data received via API"""
        
        try:
            # Extract training data
            features = np.array(training_data['features'])
            targets = np.array(training_data['targets'])
            position = training_data.get('position', 'QB')
            
            print(f"🚀 Training {position} model with {len(features)} samples from API data...")
            
            # Data validation
            if len(features) != len(targets):
                raise ValueError("Features and targets must have same length")
            
            if len(features) < 20:
                raise ValueError("Need at least 20 samples for training")
            
            # Split data (80/20 train/test)
            split_idx = int(len(features) * 0.8)
            X_train, X_test = features[:split_idx], features[split_idx:]
            y_train, y_test = targets[:split_idx], targets[split_idx:]
            
            # Feature scaling
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Configure XGBoost model
            default_config = {
                'n_estimators': 100,
                'max_depth': 4,
                'learning_rate': 0.1,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'random_state': 42
            }
            
            if model_config:
                default_config.update(model_config)
            
            # Train model
            model = xgb.XGBRegressor(**default_config)
            model.fit(X_train_scaled, y_train)
            
            # Validate
            y_pred = model.predict(X_test_scaled)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Betting accuracy
            line = np.median(y_test)
            betting_acc = accuracy_score(y_test > line, y_pred > line)
            
            # Store model
            self.models[position] = model
            self.scalers[position] = scaler
            
            # Log training
            training_record = {
                'timestamp': datetime.now().isoformat(),
                'position': position,
                'samples': len(features),
                'mae': mae,
                'r2': r2,
                'betting_accuracy': betting_acc,
                'config': default_config
            }
            self.training_history.append(training_record)
            
            print(f"✅ {position} model trained successfully!")
            print(f"   Performance: MAE={mae:.2f}, R²={r2:.3f}, Betting Acc={betting_acc:.1%}")
            
            return {
                'success': True,
                'position': position,
                'performance': {
                    'mae': mae,
                    'r2': r2,
                    'betting_accuracy': betting_acc,
                    'training_samples': len(X_train),
                    'test_samples': len(X_test)
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def save_models(self, filepath: str = 'api_trained_models.pkl'):
        """Save all API-trained models"""
        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'training_history': self.training_history,
            'created_at': datetime.now().isoformat()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"📁 API-trained models saved to {filepath}")

# ============================================================================
# APPROACH 2: Consuming External APIs for Training Data
# ============================================================================

class ExternalAPITrainer:
    """Train models by consuming external APIs for data"""
    
    def __init__(self):
        self.api_keys = {
            'tank01': os.getenv('TANK01_API_KEY', 'demo'),
            'rapidapi': os.getenv('RAPIDAPI_KEY', 'demo'),
            'nfl_api': os.getenv('NFL_API_KEY', 'demo')
        }
    
    async def fetch_nfl_data_from_apis(self, weeks: int = 10):
        """Fetch NFL data from multiple APIs for training"""
        
        print(f"🌐 Fetching NFL data from APIs for last {weeks} weeks...")
        
        all_data = []
        
        # Tank01 API (Free tier)
        tank01_data = await self._fetch_tank01_data(weeks)
        if tank01_data:
            all_data.extend(tank01_data)
        
        # ESPN API (Free)
        espn_data = await self._fetch_espn_data(weeks)
        if espn_data:
            all_data.extend(espn_data)
        
        # NFL.com API (Limited free)
        nfl_data = await self._fetch_nfl_official_data(weeks)
        if nfl_data:
            all_data.extend(nfl_data)
        
        print(f"📊 Collected {len(all_data)} player records from APIs")
        return all_data
    
    async def _fetch_tank01_data(self, weeks: int):
        """Fetch data from Tank01 API"""
        try:
            async with aiohttp.ClientSession() as session:
                # Get recent games
                url = "https://tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com/getNFLGamesForWeek"
                headers = {
                    "X-RapidAPI-Key": self.api_keys['rapidapi'],
                    "X-RapidAPI-Host": "tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com"
                }
                
                player_data = []
                current_week = 10  # Current week
                
                for week in range(max(1, current_week - weeks), current_week + 1):
                    params = {
                        "week": str(week),
                        "seasonType": "reg",
                        "season": "2024"
                    }
                    
                    async with session.get(url, headers=headers, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            # Process games and extract player stats
                            for game in data.get('body', []):
                                # Extract player stats from game data
                                # This would need to be customized based on Tank01's actual response structure
                                if 'playerStats' in game:
                                    player_data.extend(self._process_tank01_player_stats(game['playerStats']))
                    
                    # Rate limiting
                    await asyncio.sleep(0.5)
                
                return player_data
                
        except Exception as e:
            print(f"❌ Error fetching Tank01 data: {e}")
            return []
    
    async def _fetch_espn_data(self, weeks: int):
        """Fetch data from ESPN API (free, no key required)"""
        try:
            async with aiohttp.ClientSession() as session:
                player_data = []
                
                # ESPN NFL API endpoints
                base_url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"
                
                # Get recent scores
                async with session.get(f"{base_url}/scoreboard") as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Extract player stats from games
                        for event in data.get('events', []):
                            # Process ESPN game data for player stats
                            if 'competitions' in event:
                                for comp in event['competitions']:
                                    if 'competitors' in comp:
                                        for team in comp['competitors']:
                                            # Extract team and player stats
                                            team_stats = self._process_espn_team_stats(team)
                                            if team_stats:
                                                player_data.extend(team_stats)
                
                return player_data
                
        except Exception as e:
            print(f"❌ Error fetching ESPN data: {e}")
            return []
    
    async def _fetch_nfl_official_data(self, weeks: int):
        """Fetch data from NFL.com official API"""
        try:
            # NFL.com has limited free access
            # This is a placeholder for the actual implementation
            print("📡 NFL.com API integration would go here...")
            return []
            
        except Exception as e:
            print(f"❌ Error fetching NFL.com data: {e}")
            return []
    
    def _process_tank01_player_stats(self, player_stats):
        """Process Tank01 player statistics into training format"""
        processed = []
        
        for player in player_stats:
            # Convert Tank01 format to training data format
            if player.get('position') in ['QB', 'RB', 'WR', 'TE']:
                processed_player = {
                    'name': player.get('name'),
                    'position': player.get('position'),
                    'age': player.get('age', 25),
                    'experience': player.get('exp', 3),
                    'yards': player.get('yards', 0),
                    'touchdowns': player.get('tds', 0),
                    'games': 1,  # Single game data
                    'team': player.get('team')
                }
                processed.append(processed_player)
        
        return processed
    
    def _process_espn_team_stats(self, team_data):
        """Process ESPN team data into player training format"""
        # This would process ESPN's team data structure
        # Placeholder implementation
        return []
    
    def train_from_api_data(self, position: str = 'QB'):
        """Train XGBoost model using data fetched from APIs"""
        
        print(f"🤖 Training {position} model using API data...")
        
        # Fetch data asynchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        api_data = loop.run_until_complete(self.fetch_nfl_data_from_apis(weeks=8))
        loop.close()
        
        if not api_data:
            print("❌ No API data available for training")
            return None
        
        # Filter data for position
        position_data = [p for p in api_data if p.get('position') == position]
        
        if len(position_data) < 20:
            print(f"❌ Insufficient {position} data: {len(position_data)} samples")
            return None
        
        # Convert to training format
        features = []
        targets = []
        
        for player in position_data:
            # Create feature vector
            feature_vector = [
                player.get('age', 25),
                player.get('experience', 3),
                player.get('games', 1),
                1 if player.get('age', 25) >= 32 else 0,  # veteran
                player.get('yards', 0) / max(player.get('games', 1), 1)  # prev yards per game
            ]
            features.append(feature_vector)
            targets.append(player.get('yards', 0))
        
        # Train model
        trainer = APITrainingService()
        training_data = {
            'features': features,
            'targets': targets,
            'position': position
        }
        
        result = trainer.train_model_from_api_data(training_data)
        
        if result['success']:
            trainer.save_models(f'api_trained_{position.lower()}_model.pkl')
        
        return result

# ============================================================================
# APPROACH 3: Flask API Endpoints for Training
# ============================================================================

app = Flask(__name__)
training_service = APITrainingService()

@app.route('/train', methods=['POST'])
def train_model_endpoint():
    """API endpoint to train XGBoost model with provided data"""
    
    try:
        # Get training data from request
        data = request.json
        
        required_fields = ['features', 'targets']
        if not all(field in data for field in required_fields):
            return jsonify({
                'success': False,
                'error': 'Missing required fields: features, targets'
            }), 400
        
        # Optional model configuration
        model_config = data.get('model_config', {})
        
        # Train model
        result = training_service.train_model_from_api_data(data, model_config)
        
        if result['success']:
            # Save models
            training_service.save_models()
            return jsonify(result), 200
        else:
            return jsonify(result), 400
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/train/from-apis/<position>', methods=['POST'])
def train_from_external_apis(position):
    """Train model by fetching data from external APIs"""
    
    try:
        trainer = ExternalAPITrainer()
        result = trainer.train_from_api_data(position.upper())
        
        if result and result['success']:
            return jsonify(result), 200
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to train from API data'
            }), 400
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/models', methods=['GET'])
def get_models():
    """Get information about trained models"""
    
    model_info = {
        'available_models': list(training_service.models.keys()),
        'training_history': training_service.training_history,
        'total_models': len(training_service.models)
    }
    
    return jsonify(model_info)

@app.route('/predict/<position>', methods=['POST'])
def predict_with_api_model(position):
    """Make predictions using API-trained models"""
    
    try:
        if position.upper() not in training_service.models:
            return jsonify({
                'success': False,
                'error': f'No trained model for {position}'
            }), 404
        
        data = request.json
        features = np.array(data['features']).reshape(1, -1)
        
        # Scale features
        scaled_features = training_service.scalers[position.upper()].transform(features)
        
        # Make prediction
        prediction = training_service.models[position.upper()].predict(scaled_features)[0]
        
        return jsonify({
            'success': True,
            'position': position.upper(),
            'prediction': float(prediction),
            'model_type': 'API-trained XGBoost'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("🚀 XGBoost API Training Examples")
    print("=" * 50)
    
    # Example 1: Train model with provided data
    print("\n1️⃣ Training model with API data...")
    
    # Simulate training data
    sample_training_data = {
        'features': np.random.randn(100, 5).tolist(),  # 100 samples, 5 features
        'targets': (np.random.randn(100) * 50 + 200).tolist(),  # Target yards
        'position': 'QB'
    }
    
    trainer = APITrainingService()
    result = trainer.train_model_from_api_data(sample_training_data)
    print(f"Training result: {result}")
    
    # Example 2: Train from external APIs
    print("\n2️⃣ Training from external APIs...")
    api_trainer = ExternalAPITrainer()
    # Uncomment to actually fetch from APIs:
    # api_result = api_trainer.train_from_api_data('QB')
    print("API training setup complete (uncomment to run actual API calls)")
    
    # Example 3: Start Flask API server
    print("\n3️⃣ Starting Flask API server...")
    print("Available endpoints:")
    print("  POST /train - Train model with provided data")
    print("  POST /train/from-apis/<position> - Train from external APIs")
    print("  GET /models - Get model information")
    print("  POST /predict/<position> - Make predictions")
    
    # Uncomment to start server:
    # app.run(debug=True, port=5001)
    
    print("\n✅ All examples ready!")
    print("\n📚 BENEFITS of API-based training:")
    print("1. Real-time model updates with fresh data")
    print("2. Distributed training across multiple data sources")
    print("3. Easy integration with existing systems")
    print("4. Scalable and automated retraining")
    print("5. Version control for model iterations")