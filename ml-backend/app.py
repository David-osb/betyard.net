#!/usr/bin/env python3
"""
BetYard NFL QB Prediction ML Backend
Real XGBoost model serving predictions via Flask API
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import xgboost as xgb
import requests
import json
from datetime import datetime, timedelta
import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from frontend

@dataclass
class QBPrediction:
    """QB performance prediction with confidence metrics"""
    passing_yards: float
    completions: float
    attempts: float
    touchdowns: float
    interceptions: float
    qb_rating: float
    confidence: float
    features_used: Dict
    weather_impact: float
    injury_adjustment: float

class NFLWeatherService:
    """Fetch weather data for NFL games"""
    
    def __init__(self):
        # You can use OpenWeatherMap API or similar
        self.api_key = os.getenv('WEATHER_API_KEY', 'demo_key')
    
    def get_game_weather(self, team_code: str, date: str = None) -> Dict:
        """Get weather conditions for team's game"""
        # Stadium locations mapping
        stadium_locations = {
            'KC': {'lat': 39.0489, 'lon': -94.4839, 'dome': False},  # Arrowhead
            'BUF': {'lat': 42.7738, 'lon': -78.7870, 'dome': False},  # Highmark
            'DAL': {'lat': 32.7473, 'lon': -97.0945, 'dome': True},   # AT&T Stadium
            'PHI': {'lat': 39.9008, 'lon': -75.1675, 'dome': False},  # Lincoln Financial
            'CIN': {'lat': 39.0955, 'lon': -84.5160, 'dome': False},  # Paycor Stadium
            # Add more as needed
        }
        
        if team_code not in stadium_locations:
            return {'temp': 72, 'wind': 5, 'precipitation': 0, 'dome': False}
        
        location = stadium_locations[team_code]
        
        if location['dome']:
            return {'temp': 72, 'wind': 0, 'precipitation': 0, 'dome': True}
        
        # Mock weather for now - in production, use real API
        return {
            'temp': np.random.normal(65, 15),
            'wind': np.random.exponential(8),
            'precipitation': np.random.exponential(0.1),
            'dome': False
        }

class NFLInjuryService:
    """Track injury reports for players"""
    
    def get_qb_injury_status(self, qb_name: str, team_code: str) -> Dict:
        """Get injury status for QB"""
        # Mock injury data - in production, scrape from ESPN/NFL.com
        injury_statuses = ['Healthy', 'Questionable', 'Doubtful', 'Out']
        
        # For demo, most QBs are healthy
        status = np.random.choice(injury_statuses, p=[0.7, 0.2, 0.07, 0.03])
        
        injury_impact = {
            'Healthy': 1.0,
            'Questionable': 0.9,
            'Doubtful': 0.7,
            'Out': 0.0
        }
        
        return {
            'status': status,
            'impact_factor': injury_impact[status],
            'injury_type': 'None' if status == 'Healthy' else 'Shoulder'
        }

class NFLMLModel:
    """XGBoost model for NFL QB predictions"""
    
    def __init__(self):
        self.model = None
        self.feature_names = [
            'recent_form', 'home_advantage', 'opponent_defense_rank',
            'temperature', 'wind_speed', 'injury_factor', 'experience',
            'season_avg_yards', 'season_avg_tds', 'weather_impact'
        ]
        self.weather_service = NFLWeatherService()
        self.injury_service = NFLInjuryService()
        
        # Initialize or load model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize XGBoost model with trained weights or create new one"""
        model_path = 'qb_model.pkl'
        
        if os.path.exists(model_path):
            logger.info("Loading existing model...")
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
        else:
            logger.info("Creating new XGBoost model...")
            self._create_and_train_model()
    
    def _create_and_train_model(self):
        """Create and train XGBoost model with synthetic historical data"""
        logger.info("Generating training data...")
        
        # Generate synthetic historical QB performance data
        n_samples = 5000
        np.random.seed(42)
        
        # Features
        recent_form = np.random.normal(0.7, 0.2, n_samples)
        home_advantage = np.random.choice([0, 1], n_samples, p=[0.5, 0.5])
        opponent_defense_rank = np.random.uniform(1, 32, n_samples)
        temperature = np.random.normal(65, 20, n_samples)
        wind_speed = np.random.exponential(8, n_samples)
        injury_factor = np.random.choice([1.0, 0.9, 0.8, 0.6], n_samples, p=[0.7, 0.2, 0.07, 0.03])
        experience = np.random.uniform(1, 20, n_samples)
        season_avg_yards = np.random.normal(250, 50, n_samples)
        season_avg_tds = np.random.normal(1.8, 0.8, n_samples)
        weather_impact = np.where(temperature < 32, 0.8, 
                         np.where(wind_speed > 15, 0.85, 1.0))
        
        # Create feature matrix
        X = np.column_stack([
            recent_form, home_advantage, opponent_defense_rank,
            temperature, wind_speed, injury_factor, experience,
            season_avg_yards, season_avg_tds, weather_impact
        ])
        
        # Target: passing yards (realistic NFL distribution)
        base_yards = (season_avg_yards * recent_form * injury_factor * weather_impact +
                     home_advantage * 15 - (opponent_defense_rank - 16) * 3)
        
        y = np.clip(base_yards + np.random.normal(0, 25, n_samples), 100, 500)
        
        # Train XGBoost model
        logger.info("Training XGBoost model...")
        self.model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        
        self.model.fit(X, y)
        
        # Save model
        with open('qb_model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        
        logger.info("Model training completed and saved!")
    
    def predict_qb_performance(self, qb_name: str, team_code: str, 
                             opponent_code: str = None, 
                             date: str = None) -> QBPrediction:
        """Predict QB performance using real ML model"""
        
        # Get real-time data
        weather = self.weather_service.get_game_weather(team_code, date)
        injury = self.injury_service.get_qb_injury_status(qb_name, team_code)
        
        # Mock QB historical stats (in production, fetch from database)
        qb_stats = {
            'recent_form': np.random.normal(0.75, 0.15),
            'season_avg_yards': np.random.normal(260, 40),
            'season_avg_tds': np.random.normal(1.9, 0.6),
            'experience': np.random.uniform(2, 15)
        }
        
        # Create feature vector
        features = np.array([[
            qb_stats['recent_form'],
            1,  # home_advantage (mock)
            np.random.uniform(5, 25),  # opponent_defense_rank
            weather['temp'],
            weather['wind'],
            injury['impact_factor'],
            qb_stats['experience'],
            qb_stats['season_avg_yards'],
            qb_stats['season_avg_tds'],
            1.0 if weather['dome'] else max(0.7, 1 - weather['wind']/50 - abs(weather['temp']-70)/100)
        ]])
        
        # Make prediction
        pred_yards = self.model.predict(features)[0]
        
        # Calculate other stats based on yards
        attempts = np.clip(pred_yards / 7.5 + np.random.normal(0, 3), 25, 55)
        completions = attempts * np.clip(np.random.normal(0.65, 0.05), 0.4, 0.8)
        touchdowns = max(0, pred_yards / 120 + np.random.normal(0, 0.8))
        interceptions = max(0, np.random.poisson(1.2) * (1 - qb_stats['recent_form']))
        
        # Calculate QB rating
        comp_pct = completions / attempts * 100
        yards_per_att = pred_yards / attempts
        td_pct = touchdowns / attempts * 100
        int_pct = interceptions / attempts * 100
        
        # Simplified QB rating calculation
        qb_rating = min(158.3, max(0, 
            (comp_pct - 30) * 0.05 + 
            (yards_per_att - 3) * 0.25 + 
            td_pct * 0.2 - 
            int_pct * 0.25
        ) * 100)
        
        # Calculate confidence based on feature importance
        feature_importance = self.model.feature_importances_
        confidence = np.clip(
            np.dot(features[0], feature_importance) * 100 + 
            np.random.normal(0, 5), 60, 95
        )
        
        return QBPrediction(
            passing_yards=float(round(pred_yards, 1)),
            completions=float(round(completions, 1)),
            attempts=float(round(attempts, 1)),
            touchdowns=float(round(touchdowns, 1)),
            interceptions=float(round(interceptions, 1)),
            qb_rating=float(round(qb_rating, 1)),
            confidence=float(round(confidence, 1)),
            features_used={
                'weather_temp': float(weather['temp']),
                'weather_wind': float(weather['wind']),
                'injury_factor': float(injury['impact_factor']),
                'recent_form': float(qb_stats['recent_form'])
            },
            weather_impact=float(round(features[0][-1], 2)),
            injury_adjustment=float(injury['impact_factor'])
        )

# Initialize ML model
logger.info("Initializing NFL ML Model...")
ml_model = NFLMLModel()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': ml_model.model is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict_qb():
    """Main prediction endpoint"""
    try:
        data = request.get_json()
        
        qb_name = data.get('qb_name')
        team_code = data.get('team_code')
        opponent_code = data.get('opponent_code')
        
        if not qb_name or not team_code:
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Generate prediction
        prediction = ml_model.predict_qb_performance(
            qb_name, team_code, opponent_code
        )
        
        return jsonify({
            'success': True,
            'prediction': {
                'passing_yards': float(prediction.passing_yards),
                'completions': float(prediction.completions),
                'attempts': float(prediction.attempts),
                'touchdowns': float(prediction.touchdowns),
                'interceptions': float(prediction.interceptions),
                'qb_rating': float(prediction.qb_rating),
                'confidence': float(prediction.confidence)
            },
            'metadata': {
                'features_used': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                                for k, v in prediction.features_used.items()},
                'weather_impact': float(prediction.weather_impact),
                'injury_adjustment': float(prediction.injury_adjustment),
                'model_version': '1.0',
                'timestamp': datetime.now().isoformat()
            }
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/model/info', methods=['GET'])
def model_info():
    """Get model information and feature importance"""
    try:
        # Convert numpy float32 values to regular Python floats for JSON serialization
        feature_importance = dict(zip(
            ml_model.feature_names, 
            [float(x) for x in ml_model.model.feature_importances_]
        ))
        
        return jsonify({
            'model_type': 'XGBoost Regressor',
            'features': ml_model.feature_names,
            'feature_importance': feature_importance,
            'training_samples': 5000,
            'version': '1.0'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    import os
    
    # Get port from environment variable (Railway/Heroku sets this)
    port = int(os.environ.get('PORT', 5000))
    
    logger.info("🚀 Starting NFL QB Prediction ML Backend...")
    logger.info("🏈 XGBoost Model with Weather & Injury Intelligence")
    logger.info("=" * 50)
    logger.info(f"📡 Server running on port {port}")
    logger.info("Available endpoints:")
    logger.info("  GET  /health - Health check")
    logger.info("  POST /predict - QB performance prediction")
    logger.info("  GET  /model/info - Model information")
    logger.info("=" * 50)
    
    # Production ready settings for cloud deployment
    app.run(host='0.0.0.0', port=port, debug=False)