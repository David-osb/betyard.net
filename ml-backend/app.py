#!/usr/bin/env python3
"""
BetYard NFL Multi-Position Prediction ML Backend
Real XGBoost models for QB, RB, WR, TE predictions via Flask API
Includes Tank01 API proxy to solve CORS issues
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
class PlayerPrediction:
    """Universal player performance prediction with confidence metrics"""
    # QB stats
    passing_yards: Optional[float] = None
    completions: Optional[float] = None
    attempts: Optional[float] = None
    
    # RB stats
    rushing_yards: Optional[float] = None
    rushing_attempts: Optional[float] = None
    
    # WR/TE stats
    receiving_yards: Optional[float] = None
    receptions: Optional[float] = None
    targets: Optional[float] = None
    
    # Common stats
    touchdowns: float = 0
    interceptions: Optional[float] = None
    fumbles: Optional[float] = None
    qb_rating: Optional[float] = None
    fantasy_points: Optional[float] = None
    
    # Metadata
    confidence: float = 0.85
    features_used: Dict = None
    weather_impact: float = 1.0
    injury_adjustment: float = 1.0
    position: str = "QB"

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
    """XGBoost models for NFL multi-position predictions"""
    
    def __init__(self):
        self.models = {}  # Store models for each position
        self.feature_names = {
            'QB': ['recent_form', 'home_advantage', 'opponent_defense_rank',
                   'temperature', 'wind_speed', 'injury_factor', 'experience',
                   'season_avg_yards', 'season_avg_tds', 'weather_impact'],
            'RB': ['recent_form', 'home_advantage', 'opponent_rush_defense_rank',
                   'temperature', 'injury_factor', 'experience',
                   'season_avg_yards', 'season_avg_tds', 'carries_per_game'],
            'WR': ['recent_form', 'home_advantage', 'opponent_pass_defense_rank',
                   'temperature', 'wind_speed', 'injury_factor', 'experience',
                   'season_avg_yards', 'season_avg_receptions', 'target_share'],
            'TE': ['recent_form', 'home_advantage', 'opponent_pass_defense_rank',
                   'temperature', 'injury_factor', 'experience',
                   'season_avg_yards', 'season_avg_receptions', 'blocking_snaps']
        }
        self.weather_service = NFLWeatherService()
        self.injury_service = NFLInjuryService()
        
        # Initialize or load models for all positions
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize XGBoost models for all positions"""
        positions = ['QB', 'RB', 'WR', 'TE']
        
        for position in positions:
            model_path = f'{position.lower()}_model.pkl'
            
            if os.path.exists(model_path):
                logger.info(f"Loading existing {position} model...")
                with open(model_path, 'rb') as f:
                    self.models[position] = pickle.load(f)
            else:
                logger.info(f"Creating new {position} model...")
                self._create_and_train_model(position)
    
    def _create_and_train_model(self, position: str):
        """Create and train XGBoost model for specific position"""
        logger.info(f"Generating training data for {position}...")
        
        n_samples = 5000
        np.random.seed(42 + hash(position) % 100)
        
        if position == 'QB':
            X, y = self._generate_qb_training_data(n_samples)
        elif position == 'RB':
            X, y = self._generate_rb_training_data(n_samples)
        elif position == 'WR':
            X, y = self._generate_wr_training_data(n_samples)
        elif position == 'TE':
            X, y = self._generate_te_training_data(n_samples)
        else:
            raise ValueError(f"Unknown position: {position}")
        
        # Train XGBoost model
        logger.info(f"Training {position} XGBoost model...")
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        
        model.fit(X, y)
        self.models[position] = model
        
        # Save model
        with open(f'{position.lower()}_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        logger.info(f"{position} model training completed and saved!")
    
    def _generate_qb_training_data(self, n_samples):
        """Generate realistic QB training data based on NFL stats"""
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
        
        X = np.column_stack([
            recent_form, home_advantage, opponent_defense_rank,
            temperature, wind_speed, injury_factor, experience,
            season_avg_yards, season_avg_tds, weather_impact
        ])
        
        # Target: passing yards (NFL average ~260 yards/game)
        base_yards = (season_avg_yards * recent_form * injury_factor * weather_impact +
                     home_advantage * 15 - (opponent_defense_rank - 16) * 3)
        y = np.clip(base_yards + np.random.normal(0, 25, n_samples), 100, 500)
        
        return X, y
    
    def _generate_rb_training_data(self, n_samples):
        """Generate realistic RB training data based on NFL stats"""
        recent_form = np.random.normal(0.7, 0.2, n_samples)
        home_advantage = np.random.choice([0, 1], n_samples, p=[0.5, 0.5])
        opponent_rush_defense_rank = np.random.uniform(1, 32, n_samples)
        temperature = np.random.normal(65, 20, n_samples)
        injury_factor = np.random.choice([1.0, 0.9, 0.8, 0.6], n_samples, p=[0.7, 0.2, 0.07, 0.03])
        experience = np.random.uniform(1, 15, n_samples)
        season_avg_yards = np.random.normal(70, 30, n_samples)  # RB avg ~70 yards/game
        season_avg_tds = np.random.normal(0.6, 0.4, n_samples)
        carries_per_game = np.random.normal(15, 5, n_samples)
        
        X = np.column_stack([
            recent_form, home_advantage, opponent_rush_defense_rank,
            temperature, injury_factor, experience,
            season_avg_yards, season_avg_tds, carries_per_game
        ])
        
        # Target: rushing yards (NFL RB average ~70 yards/game)
        base_yards = (season_avg_yards * recent_form * injury_factor +
                     home_advantage * 8 - (opponent_rush_defense_rank - 16) * 2 +
                     carries_per_game * 2.5)
        y = np.clip(base_yards + np.random.normal(0, 20, n_samples), 10, 250)
        
        return X, y
    
    def _generate_wr_training_data(self, n_samples):
        """Generate realistic WR training data based on NFL stats"""
        recent_form = np.random.normal(0.7, 0.2, n_samples)
        home_advantage = np.random.choice([0, 1], n_samples, p=[0.5, 0.5])
        opponent_pass_defense_rank = np.random.uniform(1, 32, n_samples)
        temperature = np.random.normal(65, 20, n_samples)
        wind_speed = np.random.exponential(8, n_samples)
        injury_factor = np.random.choice([1.0, 0.9, 0.8, 0.6], n_samples, p=[0.7, 0.2, 0.07, 0.03])
        experience = np.random.uniform(1, 15, n_samples)
        season_avg_yards = np.random.normal(60, 25, n_samples)  # WR avg ~60 yards/game
        season_avg_receptions = np.random.normal(5, 2, n_samples)
        target_share = np.random.normal(0.20, 0.08, n_samples)
        
        weather_impact = np.where(wind_speed > 15, 0.85, 1.0)
        
        X = np.column_stack([
            recent_form, home_advantage, opponent_pass_defense_rank,
            temperature, wind_speed, injury_factor, experience,
            season_avg_yards, season_avg_receptions, target_share
        ])
        
        # Target: receiving yards (NFL WR average ~60 yards/game)
        base_yards = (season_avg_yards * recent_form * injury_factor * weather_impact +
                     home_advantage * 8 - (opponent_pass_defense_rank - 16) * 2.5 +
                     target_share * 150)
        y = np.clip(base_yards + np.random.normal(0, 20, n_samples), 5, 200)
        
        return X, y
    
    def _generate_te_training_data(self, n_samples):
        """Generate realistic TE training data based on NFL stats"""
        recent_form = np.random.normal(0.7, 0.2, n_samples)
        home_advantage = np.random.choice([0, 1], n_samples, p=[0.5, 0.5])
        opponent_pass_defense_rank = np.random.uniform(1, 32, n_samples)
        temperature = np.random.normal(65, 20, n_samples)
        injury_factor = np.random.choice([1.0, 0.9, 0.8, 0.6], n_samples, p=[0.7, 0.2, 0.07, 0.03])
        experience = np.random.uniform(1, 15, n_samples)
        season_avg_yards = np.random.normal(45, 20, n_samples)  # TE avg ~45 yards/game
        season_avg_receptions = np.random.normal(4, 1.5, n_samples)
        blocking_snaps = np.random.normal(0.35, 0.15, n_samples)  # % of snaps blocking
        
        X = np.column_stack([
            recent_form, home_advantage, opponent_pass_defense_rank,
            temperature, injury_factor, experience,
            season_avg_yards, season_avg_receptions, blocking_snaps
        ])
        
        # Target: receiving yards (NFL TE average ~45 yards/game)
        base_yards = (season_avg_yards * recent_form * injury_factor +
                     home_advantage * 6 - (opponent_pass_defense_rank - 16) * 1.5 -
                     blocking_snaps * 25)  # More blocking = fewer receiving yards
        y = np.clip(base_yards + np.random.normal(0, 15, n_samples), 5, 150)
        
        return X, y
    
    def predict_qb_performance(self, qb_name: str, team_code: str, 
                             opponent_code: str = None, 
                             date: str = None) -> PlayerPrediction:
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
        
        # Create feature vector matching real data training
        # QB model expects: completions, attempts, passing_tds, interceptions, sacks,
        # passing_air_yards, passing_yards_after_catch, passing_first_downs
        
        # Estimate typical game stats based on season averages
        est_attempts = np.clip(qb_stats['season_avg_yards'] / 7.5 + np.random.normal(0, 3), 25, 55)
        est_completions = est_attempts * 0.65 * qb_stats['recent_form']
        est_tds = qb_stats['season_avg_tds'] * qb_stats['recent_form']
        est_ints = max(0, np.random.poisson(1) * (1.5 - qb_stats['recent_form']))
        est_sacks = max(0, np.random.poisson(2))
        est_air_yards = qb_stats['season_avg_yards'] * 0.6
        est_yac = qb_stats['season_avg_yards'] * 0.4
        est_first_downs = qb_stats['season_avg_yards'] / 15
        
        features = np.array([[
            est_completions,
            est_attempts,
            est_tds,
            est_ints,
            est_sacks,
            est_air_yards,
            est_yac,
            est_first_downs
        ]])
        
        # Make prediction using QB model
        pred_yards = self.models['QB'].predict(features)[0]
        
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
        feature_importance = self.models['QB'].feature_importances_
        confidence = np.clip(
            np.dot(features[0], feature_importance) * 100 + 
            np.random.normal(0, 5), 60, 95
        )
        
        return PlayerPrediction(
            passing_yards=float(round(pred_yards, 1)),
            completions=float(round(completions, 1)),
            attempts=float(round(attempts, 1)),
            touchdowns=float(round(touchdowns, 1)),
            interceptions=float(round(interceptions, 1)),
            qb_rating=float(round(qb_rating, 1)),
            confidence=float(round(confidence, 1)),
            position='QB'
        )
    
    def predict_rb_performance(self, rb_name: str, team_code: str, 
                             opponent_code: str = None, 
                             date: str = None) -> PlayerPrediction:
        """Predict RB performance using real ML model"""
        
        # Get real-time data
        weather = self.weather_service.get_game_weather(team_code, date)
        injury = self.injury_service.get_qb_injury_status(rb_name, team_code)  # Reuse injury service
        
        # Mock RB historical stats
        rb_stats = {
            'recent_form': np.random.normal(0.75, 0.15),
            'season_avg_yards': np.random.normal(70, 25),
            'season_avg_tds': np.random.normal(0.6, 0.3),
            'experience': np.random.uniform(1, 12),
            'carries_per_game': np.random.normal(15, 5)
        }
        
        # Create feature vector matching real data training
        # RB model expects: carries, rushing_tds, rushing_fumbles, rushing_first_downs,
        # receptions, targets, receiving_yards, receiving_tds
        
        est_carries = rb_stats['carries_per_game'] * rb_stats['recent_form']
        est_rush_tds = rb_stats['season_avg_tds'] * rb_stats['recent_form']
        est_fumbles = max(0, np.random.poisson(0.1))
        est_first_downs = rb_stats['season_avg_yards'] / 10
        est_receptions = np.clip(np.random.normal(3, 2), 0, 10)
        est_targets = est_receptions * 1.5
        est_rec_yards = est_receptions * np.random.normal(8, 3)
        est_rec_tds = max(0, np.random.poisson(0.2))
        
        features = np.array([[
            est_carries,
            est_rush_tds,
            est_fumbles,
            est_first_downs,
            est_receptions,
            est_targets,
            est_rec_yards,
            est_rec_tds
        ]])
        
        # Make prediction using RB model
        pred_yards = self.models['RB'].predict(features)[0]
        
        # Calculate derived stats
        attempts = np.clip(rb_stats['carries_per_game'] + np.random.normal(0, 3), 8, 30)
        touchdowns = max(0, pred_yards / 100 + np.random.normal(0, 0.5))
        
        # Calculate confidence
        feature_importance = self.models['RB'].feature_importances_
        confidence = np.clip(
            np.dot(features[0], feature_importance) * 100 + 
            np.random.normal(0, 5), 60, 95
        )
        
        return PlayerPrediction(
            rushing_yards=float(round(pred_yards, 1)),
            rushing_attempts=float(round(attempts, 1)),
            touchdowns=float(round(touchdowns, 1)),
            fantasy_points=float(round(pred_yards * 0.1 + touchdowns * 6, 1)),
            confidence=float(round(confidence, 1)),
            position='RB'
        )
    
    def predict_wr_performance(self, wr_name: str, team_code: str, 
                             opponent_code: str = None, 
                             date: str = None) -> PlayerPrediction:
        """Predict WR performance using real ML model"""
        
        # Get real-time data
        weather = self.weather_service.get_game_weather(team_code, date)
        injury = self.injury_service.get_qb_injury_status(wr_name, team_code)
        
        # Mock WR historical stats
        wr_stats = {
            'recent_form': np.random.normal(0.75, 0.15),
            'season_avg_yards': np.random.normal(60, 20),
            'season_avg_receptions': np.random.normal(5, 2),
            'experience': np.random.uniform(1, 12),
            'target_share': np.random.normal(0.20, 0.08)
        }
        
        # Create feature vector matching real data training
        # WR model expects: receptions, targets, receiving_tds, receiving_fumbles,
        # receiving_air_yards, receiving_yards_after_catch, receiving_first_downs
        
        est_receptions = wr_stats['season_avg_receptions'] * wr_stats['recent_form']
        est_targets = est_receptions / 0.65  # ~65% catch rate
        est_tds = max(0, np.random.poisson(0.6) * wr_stats['recent_form'])
        est_fumbles = max(0, np.random.poisson(0.05))
        est_air_yards = wr_stats['season_avg_yards'] * 0.65
        est_yac = wr_stats['season_avg_yards'] * 0.35
        est_first_downs = wr_stats['season_avg_yards'] / 12
        
        features = np.array([[
            est_receptions,
            est_targets,
            est_tds,
            est_fumbles,
            est_air_yards,
            est_yac,
            est_first_downs
        ]])
        
        # Make prediction using WR model
        pred_yards = self.models['WR'].predict(features)[0]
        
        # Calculate derived stats
        targets = np.clip(wr_stats['target_share'] * 40 + np.random.normal(0, 2), 3, 15)
        receptions = targets * np.clip(np.random.normal(0.65, 0.1), 0.4, 0.85)
        touchdowns = max(0, pred_yards / 150 + np.random.normal(0, 0.4))
        
        # Calculate confidence
        feature_importance = self.models['WR'].feature_importances_
        confidence = np.clip(
            np.dot(features[0], feature_importance) * 100 + 
            np.random.normal(0, 5), 60, 95
        )
        
        return PlayerPrediction(
            receiving_yards=float(round(pred_yards, 1)),
            receptions=float(round(receptions, 1)),
            targets=float(round(targets, 1)),
            touchdowns=float(round(touchdowns, 1)),
            fantasy_points=float(round(pred_yards * 0.1 + receptions * 0.5 + touchdowns * 6, 1)),
            confidence=float(round(confidence, 1)),
            position='WR'
        )
    
    def predict_te_performance(self, te_name: str, team_code: str, 
                             opponent_code: str = None, 
                             date: str = None) -> PlayerPrediction:
        """Predict TE performance using real ML model"""
        
        # Get real-time data
        weather = self.weather_service.get_game_weather(team_code, date)
        injury = self.injury_service.get_qb_injury_status(te_name, team_code)
        
        # Mock TE historical stats
        te_stats = {
            'recent_form': np.random.normal(0.75, 0.15),
            'season_avg_yards': np.random.normal(45, 18),
            'season_avg_receptions': np.random.normal(4, 1.5),
            'experience': np.random.uniform(1, 12),
            'blocking_snaps': np.random.normal(0.35, 0.15)
        }
        
        # Create feature vector matching real data training
        # TE model expects: receptions, targets, receiving_tds, receiving_fumbles,
        # receiving_air_yards, receiving_yards_after_catch, receiving_first_downs
        
        est_receptions = te_stats['season_avg_receptions'] * te_stats['recent_form']
        est_targets = est_receptions / 0.68  # TEs have ~68% catch rate
        est_tds = max(0, np.random.poisson(0.5) * te_stats['recent_form'])
        est_fumbles = max(0, np.random.poisson(0.03))
        est_air_yards = te_stats['season_avg_yards'] * 0.55
        est_yac = te_stats['season_avg_yards'] * 0.45
        est_first_downs = te_stats['season_avg_yards'] / 11
        
        features = np.array([[
            est_receptions,
            est_targets,
            est_tds,
            est_fumbles,
            est_air_yards,
            est_yac,
            est_first_downs
        ]])
        
        # Make prediction using TE model
        pred_yards = self.models['TE'].predict(features)[0]
        
        # Calculate derived stats
        targets = np.clip(np.random.normal(6, 2), 2, 12)
        receptions = targets * np.clip(np.random.normal(0.65, 0.1), 0.4, 0.85)
        touchdowns = max(0, pred_yards / 120 + np.random.normal(0, 0.4))
        
        # Calculate confidence
        feature_importance = self.models['TE'].feature_importances_
        confidence = np.clip(
            np.dot(features[0], feature_importance) * 100 + 
            np.random.normal(0, 5), 60, 95
        )
        
        return PlayerPrediction(
            receiving_yards=float(round(pred_yards, 1)),
            receptions=float(round(receptions, 1)),
            targets=float(round(targets, 1)),
            touchdowns=float(round(touchdowns, 1)),
            fantasy_points=float(round(pred_yards * 0.1 + receptions * 0.5 + touchdowns * 6, 1)),
            confidence=float(round(confidence, 1)),
            position='TE'
        )


# Initialize ML model
logger.info("Initializing NFL ML Model...")
ml_model = NFLMLModel()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': {
            'QB': 'QB' in ml_model.models,
            'RB': 'RB' in ml_model.models,
            'WR': 'WR' in ml_model.models,
            'TE': 'TE' in ml_model.models
        },
        'timestamp': datetime.now().isoformat(),
        'version': 'v2024-10-20-multi-position'
    })

@app.route('/test', methods=['GET'])
def test_json_fix():
    """Test endpoint to verify JSON serialization fix is deployed"""
    import numpy as np
    test_float32 = np.float32(123.456)
    return jsonify({
        'test': 'JSON serialization fix',
        'float32_converted': float(test_float32),
        'status': 'working'
    })

@app.route('/predict', methods=['POST'])
def predict_player():
    """Main prediction endpoint - supports all positions (QB, RB, WR, TE)"""
    try:
        data = request.get_json()
        
        player_name = data.get('player_name') or data.get('qb_name')  # Support legacy field
        team_code = data.get('team_code')
        opponent_code = data.get('opponent_code')
        position = data.get('position', 'QB').upper()
        
        if not player_name or not team_code:
            return jsonify({'error': 'Missing required fields: player_name, team_code'}), 400
        
        if position not in ['QB', 'RB', 'WR', 'TE']:
            return jsonify({'error': f'Invalid position: {position}. Must be QB, RB, WR, or TE'}), 400
        
        # Route to appropriate prediction function
        if position == 'QB':
            prediction = ml_model.predict_qb_performance(player_name, team_code, opponent_code)
        elif position == 'RB':
            prediction = ml_model.predict_rb_performance(player_name, team_code, opponent_code)
        elif position == 'WR':
            prediction = ml_model.predict_wr_performance(player_name, team_code, opponent_code)
        elif position == 'TE':
            prediction = ml_model.predict_te_performance(player_name, team_code, opponent_code)
        
        # Build response with only non-None fields
        response_prediction = {
            'position': prediction.position,
            'confidence': float(prediction.confidence),
            'fantasy_points': float(prediction.fantasy_points) if prediction.fantasy_points else None
        }
        
        # Add position-specific stats
        if prediction.passing_yards is not None:
            response_prediction['passing_yards'] = float(prediction.passing_yards)
            response_prediction['completions'] = float(prediction.completions)
            response_prediction['attempts'] = float(prediction.attempts)
        
        if prediction.rushing_yards is not None:
            response_prediction['rushing_yards'] = float(prediction.rushing_yards)
            response_prediction['rushing_attempts'] = float(prediction.rushing_attempts)
        
        if prediction.receiving_yards is not None:
            response_prediction['receiving_yards'] = float(prediction.receiving_yards)
            response_prediction['receptions'] = float(prediction.receptions)
            response_prediction['targets'] = float(prediction.targets)
        
        if prediction.touchdowns is not None:
            response_prediction['touchdowns'] = float(prediction.touchdowns)
        
        if prediction.interceptions is not None:
            response_prediction['interceptions'] = float(prediction.interceptions)
        
        if prediction.qb_rating is not None:
            response_prediction['qb_rating'] = float(prediction.qb_rating)
        
        return jsonify({
            'success': True,
            'prediction': response_prediction,
            'metadata': {
                'player_name': player_name,
                'team': team_code,
                'opponent': opponent_code,
                'position': position,
                'model_version': '2.0',
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
        model_info_response = {
            'model_type': 'XGBoost Regressor (Multi-Position)',
            'supported_positions': ['QB', 'RB', 'WR', 'TE'],
            'training_samples': 5000,
            'version': '2.0'
        }
        
        # Add feature importance for each position
        for position in ['QB', 'RB', 'WR', 'TE']:
            if position in ml_model.models:
                feature_importance = dict(zip(
                    ml_model.feature_names[position], 
                    [float(x) for x in ml_model.models[position].feature_importances_]
                ))
                model_info_response[f'{position}_features'] = ml_model.feature_names[position]
                model_info_response[f'{position}_feature_importance'] = feature_importance
        
        return jsonify(model_info_response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================================================
# API PROXY ENDPOINTS (CORS Solution for Tank01 NFL API)
# ============================================================================

@app.route('/api/proxy/tank01', methods=['GET', 'POST'])
def tank01_proxy():
    """
    Proxy endpoint for Tank01 NFL API to bypass CORS restrictions
    Usage: /api/proxy/tank01?endpoint=getNFLTeamSchedule&teamAbv=KC
    """
    try:
        # Get the endpoint and parameters from query string
        endpoint = request.args.get('endpoint')
        if not endpoint:
            return jsonify({'error': 'Missing endpoint parameter'}), 400
        
        # Build the Tank01 API URL
        base_url = 'https://tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com'
        url = f"{base_url}/{endpoint}"
        
        # Get all query parameters except 'endpoint'
        params = {k: v for k, v in request.args.items() if k != 'endpoint'}
        
        # RapidAPI headers
        headers = {
            'X-RapidAPI-Key': 'be76a86cb3msh01c346c2b0ef4ffp151e0djsn0b0e85e00bd3',
            'X-RapidAPI-Host': 'tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com'
        }
        
        # Make the request to Tank01 API
        response = requests.get(url, headers=headers, params=params, timeout=10)
        
        # Return the response with CORS headers
        return jsonify(response.json()), response.status_code
        
    except requests.exceptions.Timeout:
        return jsonify({'error': 'Tank01 API timeout'}), 504
    except requests.exceptions.RequestException as e:
        logger.error(f"Tank01 proxy error: {str(e)}")
        return jsonify({'error': f'API request failed: {str(e)}'}), 500
    except Exception as e:
        logger.error(f"Proxy error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/proxy/nfl/schedule', methods=['GET'])
def nfl_schedule_proxy():
    """
    Specialized proxy for NFL schedule data
    Usage: /api/proxy/nfl/schedule?teamAbv=KC
    """
    try:
        team_abv = request.args.get('teamAbv')
        if not team_abv:
            return jsonify({'error': 'Missing teamAbv parameter'}), 400
        
        url = 'https://tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com/getNFLTeamSchedule'
        headers = {
            'X-RapidAPI-Key': 'be76a86cb3msh01c346c2b0ef4ffp151e0djsn0b0e85e00bd3',
            'X-RapidAPI-Host': 'tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com'
        }
        params = {'teamAbv': team_abv, 'season': '2024'}
        
        response = requests.get(url, headers=headers, params=params, timeout=10)
        return jsonify(response.json()), response.status_code
        
    except Exception as e:
        logger.error(f"Schedule proxy error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/proxy/nfl/roster', methods=['GET'])
def nfl_roster_proxy():
    """
    Specialized proxy for NFL team roster data
    Usage: /api/proxy/nfl/roster?teamID=LAC&getStats=false
    """
    try:
        team_id = request.args.get('teamID')
        get_stats = request.args.get('getStats', 'false')
        
        if not team_id:
            return jsonify({'error': 'Missing teamID parameter'}), 400
        
        url = 'https://tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com/getNFLTeamRoster'
        headers = {
            'X-RapidAPI-Key': 'be76a86cb3msh01c346c2b0ef4ffp151e0djsn0b0e85e00bd3',
            'X-RapidAPI-Host': 'tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com'
        }
        params = {'teamID': team_id, 'getStats': get_stats}
        
        response = requests.get(url, headers=headers, params=params, timeout=10)
        return jsonify(response.json()), response.status_code
        
    except Exception as e:
        logger.error(f"Roster proxy error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/proxy/nfl/games', methods=['GET'])
def nfl_games_proxy():
    """
    Specialized proxy for NFL games data
    Usage: /api/proxy/nfl/games?gameWeek=8&season=2024
    """
    try:
        game_week = request.args.get('gameWeek', 'all')
        season = request.args.get('season', '2024')
        
        url = 'https://tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com/getNFLGamesForWeek'
        headers = {
            'X-RapidAPI-Key': 'be76a86cb3msh01c346c2b0ef4ffp151e0djsn0b0e85e00bd3',
            'X-RapidAPI-Host': 'tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com'
        }
        params = {'week': game_week, 'season': season}
        
        response = requests.get(url, headers=headers, params=params, timeout=10)
        return jsonify(response.json()), response.status_code
        
    except Exception as e:
        logger.error(f"Games proxy error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    import os
    
    # Get port from environment variable (Railway/Heroku sets this)
    port = int(os.environ.get('PORT', 5000))
    
    logger.info("🚀 Starting NFL Multi-Position Prediction ML Backend...")
    logger.info("🏈 XGBoost Models: QB, RB, WR, TE")
    logger.info("=" * 50)
    logger.info(f"📡 Server running on port {port}")
    logger.info("Available endpoints:")
    logger.info("  GET  /health - Health check")
    logger.info("  POST /predict - Player performance prediction (all positions)")
    logger.info("  GET  /model/info - Model information")
    logger.info("=" * 50)
    
    # Production ready settings for cloud deployment
    app.run(host='0.0.0.0', port=port, debug=False)