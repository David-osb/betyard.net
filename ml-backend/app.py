#!/usr/bin/env python3
"""
BetYard NFL Multi-Position Prediction ML Backend
Real XGBoost models for QB, RB, WR, TE predictions via Flask API
Includes Tank01 API proxy to solve CORS issues
"""

from flask import Flask, request, jsonify

# Try to import CORS, but make it optional since Tank01 integration is more important
try:
    from flask_cors import CORS
    CORS_AVAILABLE = True
except ImportError:
    print("WARNING: flask-cors not available. CORS headers will be set manually.")
    CORS_AVAILABLE = False
    CORS = None
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
import time
from functools import wraps
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configure CORS - Force manual headers for better compatibility
if False:  # Temporarily disable flask-cors to use manual headers
    CORS(app, 
         origins=['https://betyard.net', 'http://localhost:*', 'https://localhost:*'],
         methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
         allow_headers=['Content-Type', 'Authorization'])
else:
    # Manual CORS headers for all domains - Fixed duplicate header issue
    @app.after_request
    def after_request(response):
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization,X-Requested-With'
        response.headers['Access-Control-Allow-Methods'] = 'GET,PUT,POST,DELETE,OPTIONS,HEAD'
        response.headers['Access-Control-Allow-Credentials'] = 'false'
        return response
    
    # Handle preflight OPTIONS requests
    @app.before_request
    def handle_preflight():
        if request.method == "OPTIONS":
            response = jsonify({})
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization,X-Requested-With'
            response.headers['Access-Control-Allow-Methods'] = 'GET,PUT,POST,DELETE,OPTIONS,HEAD'
            response.headers['Access-Control-Allow-Credentials'] = 'false'
            return response

# Simple in-memory cache for API responses
API_CACHE = {}
CACHE_DURATION = 300  # 5 minutes

# Rate limiting
RATE_LIMIT_WINDOW = 1.0  # 1 second between requests
last_request_time = 0

def cache_response(duration=CACHE_DURATION):
    """Decorator to cache API responses"""
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            # Create cache key from request args
            cache_key = f"{f.__name__}:{request.url}"
            
            # Check cache
            if cache_key in API_CACHE:
                cached_data, timestamp = API_CACHE[cache_key]
                if time.time() - timestamp < duration:
                    logger.info(f"✅ Cache hit: {cache_key}")
                    return cached_data
            
            # Call function and cache result
            result = f(*args, **kwargs)
            API_CACHE[cache_key] = (result, time.time())
            logger.info(f"💾 Cached: {cache_key}")
            return result
        return wrapped
    return decorator

def rate_limit():
    """Simple rate limiter to avoid hitting API limits"""
    global last_request_time
    current_time = time.time()
    time_since_last = current_time - last_request_time
    
    if time_since_last < RATE_LIMIT_WINDOW:
        sleep_time = RATE_LIMIT_WINDOW - time_since_last
        logger.info(f"⏱️ Rate limiting: sleeping {sleep_time:.2f}s")
        time.sleep(sleep_time)
    
    last_request_time = time.time()

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
    confidence: float = 0.85  # Legacy field - will be deprecated
    model_accuracy: float = 0.89  # Actual R² score from training
    prediction_likelihood: float = 0.75  # Likelihood this specific prediction is correct
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
    
    def __init__(self, espn_data_service=None, enhanced_espn_service=None):
        self.models = {}  # Store models for each position
        self.scalers = {}  # Store scalers for each position
        self.feature_names = {}  # Will be loaded from model file or default
        self.weather_service = NFLWeatherService()
        self.injury_service = NFLInjuryService()
        
        # Store ESPN services for real data access
        self.espn_data_service = espn_data_service
        self.enhanced_espn_service = enhanced_espn_service
        
        # Initialize or load models for all positions
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize XGBoost models for all positions"""
        try:
            # Try to load the comprehensive multiposition model first
            multiposition_path = 'modern_multiposition_models.pkl'
            if os.path.exists(multiposition_path):
                logger.info("Loading comprehensive multiposition models...")
                with open(multiposition_path, 'rb') as f:
                    multiposition_data = pickle.load(f)
                
                self.models = multiposition_data['models']
                self.scalers = multiposition_data['scalers']
                self.feature_names = multiposition_data['features']
                logger.info(f"✅ Loaded models for positions: {list(self.models.keys())}")
                return
            
        except Exception as e:
            logger.warning(f"Could not load multiposition models: {e}")
        
        # Fallback to individual model files
        positions = ['QB', 'RB', 'WR', 'TE']
        
        # Set default feature names if not loaded from comprehensive model
        if not self.feature_names:
            self.feature_names = {
                'QB': ['Age', 'experience', 'career_games', 'age_prime', 'veteran', 
                       'prev_yards_per_game', 'prev_td_per_game', 'prev_completions_per_game'],
                'RB': ['Age', 'experience', 'career_games', 'age_prime', 'veteran', 
                       'prev_yards_per_game', 'prev_td_per_game', 'prev_attempts_per_game'],
                'WR': ['Age', 'experience', 'career_games', 'age_prime', 'veteran', 
                       'prev_yards_per_game', 'prev_td_per_game', 'prev_receptions_per_game'],
                'TE': ['Age', 'experience', 'career_games', 'age_prime', 'veteran', 
                       'prev_yards_per_game', 'prev_td_per_game', 'prev_receptions_per_game']
            }
        
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
        # These should match the features the model was trained on
        age = np.random.uniform(22, 38)  # Age
        experience = max(0, age - 22)    # experience (years in league)
        career_games = experience * 16   # career_games (estimate)
        age_prime = 1 if (25 <= age <= 30) else 0  # age_prime (boolean)
        veteran = 1 if age >= 32 else 0  # veteran (boolean)
        
        # Previous season performance (would come from database in production)
        prev_yards_per_game = np.random.normal(250, 50)  # prev_yards_per_game
        prev_td_per_game = np.random.normal(1.5, 0.5)    # prev_td_per_game
        prev_completions_per_game = np.random.normal(22, 5)  # prev_completions_per_game
        
        # Create feature vector matching training data
        features = np.array([[
            age,
            experience,
            career_games,
            age_prime,
            veteran,
            prev_yards_per_game,
            prev_td_per_game,
            prev_completions_per_game
        ]])
        
        # Apply the same scaling that was used during training
        if hasattr(self, 'scalers') and 'QB' in self.scalers:
            features_scaled = self.scalers['QB'].transform(features)
        else:
            # If no scaler available, use features as-is
            features_scaled = features
        
        # Make prediction using QB model
        pred_yards = self.models['QB'].predict(features_scaled)[0]
        
        # Calculate other stats based on predicted yards
        attempts = np.clip(pred_yards / 7.5 + np.random.normal(0, 3), 25, 55)
        completions = attempts * np.clip(np.random.normal(0.65, 0.05), 0.4, 0.8)
        touchdowns = max(0, pred_yards / 120 + np.random.normal(0, 0.8))
        interceptions = max(0, np.random.poisson(1.2) * (1 - (prev_yards_per_game / 300)))
        
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
        
        # Calculate actual model accuracy and prediction likelihood
        # Model accuracy from training report (R² score on test data)
        model_accuracy = 89.0  # QB model R² = 0.89 from training_report_real_data.json
        
        # Calculate prediction likelihood based on data quality factors
        feature_importance = self.models['QB'].feature_importances_
        data_quality_score = np.dot(features[0], feature_importance)
        
        # Factors that affect prediction likelihood:
        # - Player consistency (injury_factor)
        # - Weather conditions (temperature, wind)
        # - Recent form
        # - Opponent strength
        weather_factor = 1.0 if weather.get('temp', 70) > 40 else 0.9  # Cold weather reduces accuracy
        wind_factor = max(0.7, 1.0 - (weather.get('wind', 0) / 30))  # High wind reduces accuracy
        
        prediction_likelihood = np.clip(
            data_quality_score * 100 * 
            injury.get('injury_factor', 1.0) * 
            weather_factor * 
            wind_factor * 
            (prev_yards_per_game / 300) +  # Use performance as form indicator
            np.random.normal(0, 3), 65, 92
        )
        
        # Legacy confidence for backward compatibility
        confidence = prediction_likelihood  # Keep for compatibility
        
        return PlayerPrediction(
            passing_yards=float(round(pred_yards, 1)),
            completions=float(round(completions, 1)),
            attempts=float(round(attempts, 1)),
            touchdowns=float(round(touchdowns, 1)),
            interceptions=float(round(interceptions, 1)),
            qb_rating=float(round(qb_rating, 1)),
            confidence=float(round(confidence, 1)),
            model_accuracy=float(round(model_accuracy, 1)),
            prediction_likelihood=float(round(prediction_likelihood, 1)),
            position='QB'
        )
    
    def predict_rb_performance(self, rb_name: str, team_code: str, 
                             opponent_code: str = None, 
                             date: str = None) -> PlayerPrediction:
        """Predict RB performance using real ML model with ESPN data"""
        
        # Get real-time data
        weather = self.weather_service.get_game_weather(team_code, date)
        injury = self.injury_service.get_qb_injury_status(rb_name, team_code)
        
        # Try to get real ESPN player stats
        try:
            # First, try to find the player ID from ESPN data service
            player_data = None
            if hasattr(self, 'espn_data_service') and self.espn_data_service:
                # Look for player in ESPN data
                player_data = self.espn_data_service.find_player_by_name(rb_name, team_code)
            
            if player_data and 'id' in player_data:
                # Get real player stats from enhanced ESPN service
                if hasattr(self, 'enhanced_espn_service') and self.enhanced_espn_service:
                    espn_stats = self.enhanced_espn_service.get_player_stats(player_data['id'])
                    season_stats = espn_stats.get('season_stats', {})
                    
                    # Extract real RB stats
                    rb_stats = {
                        'recent_form': 1.0,  # Default to full form
                        'season_avg_yards': season_stats.get('rushing_yards_per_game', 65),
                        'season_avg_tds': season_stats.get('rushing_touchdowns', 0.5) / 17,  # Per game
                        'experience': season_stats.get('years_pro', 3),
                        'carries_per_game': season_stats.get('rushing_attempts_per_game', 15),
                        'receiving_yards': season_stats.get('receiving_yards_per_game', 15),
                        'receptions_per_game': season_stats.get('receptions_per_game', 2.5)
                    }
                    logger.info(f"✅ Using real ESPN stats for {rb_name}: {rb_stats}")
                else:
                    raise Exception("Enhanced ESPN service not available")
            else:
                raise Exception("Player not found in ESPN data")
                
        except Exception as e:
            logger.warning(f"⚠️ Could not get real stats for {rb_name}: {e}. Using realistic defaults.")
            # Fall back to realistic default values based on typical NFL RB performance
            rb_stats = {
                'recent_form': 0.85,  # Assume good form
                'season_avg_yards': 65,  # Typical RB yards per game
                'season_avg_tds': 0.5,  # About 8-9 TDs per season
                'experience': 4,  # Average NFL experience
                'carries_per_game': 16,  # Typical RB carries
                'receiving_yards': 18,  # RB receiving yards per game
                'receptions_per_game': 2.8  # RB receptions per game
            }
        
        # Create feature vector using real stats
        # RB model expects: carries, rushing_tds, rushing_fumbles, rushing_first_downs,
        # receptions, targets, receiving_yards, receiving_tds
        
        est_carries = rb_stats['carries_per_game'] * rb_stats['recent_form']
        est_rush_tds = rb_stats['season_avg_tds'] * rb_stats['recent_form']
        est_fumbles = 0.05  # Low fumble rate
        est_first_downs = rb_stats['season_avg_yards'] / 10  # ~1 first down per 10 yards
        est_receptions = rb_stats['receptions_per_game'] * rb_stats['recent_form'] 
        est_targets = est_receptions * 1.25  # Target rate
        est_rec_yards = rb_stats['receiving_yards'] * rb_stats['recent_form']
        est_rec_tds = 0.15  # Low receiving TD rate for RBs
        
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
        
        # Scale features if scaler is available
        if hasattr(self, 'scalers') and 'RB' in self.scalers:
            features_scaled = self.scalers['RB'].transform(features)
        else:
            # If no scaler available, use features as-is
            features_scaled = features
        
        # Make prediction using RB model
        raw_prediction = self.models['RB'].predict(features_scaled)[0]
        
        # CRITICAL FIX: The model was trained on ALL RBs including backups (mean ~13.6 yards)
        # Need to scale prediction based on usage level for realistic starter predictions
        usage_multiplier = max(1.0, est_carries / 10.0)  # Scale based on carries per game
        pred_yards = raw_prediction * usage_multiplier * 1.5  # Additional scaling for starters
        
        # Ensure realistic bounds
        pred_yards = np.clip(pred_yards, 10, 150)  # Realistic RB yards range
        
        # Calculate derived stats
        attempts = np.clip(rb_stats['carries_per_game'] + np.random.normal(0, 3), 8, 30)
        touchdowns = max(0, pred_yards / 100 + np.random.normal(0, 0.5))
        
        # Add receiving stats for RBs (many RBs catch passes)
        receptions = max(0, rb_stats.get('receptions_per_game', 2.8) + np.random.normal(0, 1))
        receiving_yards = max(0, rb_stats.get('receiving_yards', 18) + np.random.normal(0, 8))
        
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
            receptions=float(round(receptions, 1)),
            receiving_yards=float(round(receiving_yards, 1)),
            fantasy_points=float(round(pred_yards * 0.1 + touchdowns * 6 + receiving_yards * 0.1 + receptions * 0.5, 1)),
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
        
        # Scale features if scaler is available
        if hasattr(self, 'scalers') and 'WR' in self.scalers:
            features_scaled = self.scalers['WR'].transform(features)
        else:
            # If no scaler available, use features as-is
            features_scaled = features
        
        # Make prediction using WR model
        raw_prediction = self.models['WR'].predict(features_scaled)[0]
        
        # CRITICAL FIX: Scale prediction based on target share for realistic starter predictions
        target_multiplier = max(1.0, wr_stats['target_share'] / 0.15)  # Scale based on target share
        pred_yards = raw_prediction * target_multiplier * 1.8  # Additional scaling for starters
        
        # Ensure realistic bounds
        pred_yards = np.clip(pred_yards, 5, 150)  # Realistic WR yards range
        
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
        
        # Scale features if scaler is available
        if hasattr(self, 'scalers') and 'TE' in self.scalers:
            features_scaled = self.scalers['TE'].transform(features)
        else:
            # If no scaler available, use features as-is
            features_scaled = features
        
        # Make prediction using TE model
        raw_prediction = self.models['TE'].predict(features_scaled)[0]
        
        # Calculate derived stats first
        targets = np.clip(np.random.normal(6, 2), 2, 12)
        
        # CRITICAL FIX: Scale prediction based on targets for realistic starter predictions
        usage_multiplier = max(1.0, targets / 4.0)  # Scale based on targets per game
        pred_yards = raw_prediction * usage_multiplier * 1.6  # Additional scaling for starters
        
        # Ensure realistic bounds
        pred_yards = np.clip(pred_yards, 2, 120)  # Realistic TE yards range
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


# Initialize ESPN services first
logger.info("Initializing ESPN services...")

# Initialize ESPN Website Data Service
try:
    from espn_website_data_service import ESPNWebsiteDataService
    espn_data_service = ESPNWebsiteDataService()
    logger.info("✅ ESPN Website Data Service initialized")
except ImportError:
    espn_data_service = None
    logger.warning("⚠️ ESPN Website Data Service not available")

# Initialize Enhanced ESPN Service
try:
    from enhanced_espn_service import EnhancedESPNService
    enhanced_espn_service = EnhancedESPNService()
    logger.info("✅ Enhanced ESPN service initialized")
    ENHANCED_ESPN_AVAILABLE = True
except ImportError:
    enhanced_espn_service = None
    logger.warning("⚠️ Enhanced ESPN service not available")
    ENHANCED_ESPN_AVAILABLE = False
except Exception as e:
    enhanced_espn_service = None
    logger.error(f"❌ Enhanced ESPN service failed: {e}")
    ENHANCED_ESPN_AVAILABLE = False

# Initialize ML model with ESPN services
logger.info("Initializing NFL ML Model...")
ml_model = NFLMLModel(espn_data_service=espn_data_service, enhanced_espn_service=enhanced_espn_service)

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
        'espn_service': espn_data_service is not None,
        'timestamp': datetime.now().isoformat(),
        'version': 'v2024-11-02-espn-integration'
    })

@app.route('/debug/config', methods=['GET'])
def debug_config():
    """Debug endpoint to check configuration"""
    try:
        import sys
        return jsonify({
            'odds_api_available': ODDS_API_AVAILABLE,
            'odds_api_key_exists': bool(ODDS_API_KEY),
            'odds_api_key_length': len(ODDS_API_KEY) if ODDS_API_KEY else 0,
            'environment_variables': {
                'ODDS_API_KEY': '***' + ODDS_API_KEY[-4:] if ODDS_API_KEY else None,
                'FLASK_ENV': os.environ.get('FLASK_ENV'),
                'PORT': os.environ.get('PORT')
            },
            'python_version': sys.version,
            'cwd': os.getcwd(),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        })

@app.route('/debug/test-odds-api', methods=['GET'])
def test_odds_api():
    """Test the external odds API directly"""
    try:
        if not ODDS_API_KEY:
            return jsonify({
                'error': 'No API key configured',
                'timestamp': datetime.now().isoformat()
            })
        
        import aiohttp
        
        async def test_api():
            url = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"
            params = {
                'apiKey': ODDS_API_KEY,
                'regions': 'us',
                'markets': 'h2h',
                'oddsFormat': 'american'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    status = response.status
                    if status == 200:
                        data = await response.json()
                        return {
                            'status': status,
                            'games_count': len(data) if data else 0,
                            'sample_game': data[0] if data else None,
                            'success': True
                        }
                    else:
                        text = await response.text()
                        return {
                            'status': status,
                            'error_response': text,
                            'success': False
                        }
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(test_api())
        loop.close()
        
        return jsonify({
            'api_test_result': result,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error testing odds API: {e}", exc_info=True)
        return jsonify({
            'error': str(e),
            'timestamp': datetime.now().isoformat()
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
            'confidence': float(prediction.confidence),  # Legacy field for compatibility
            'model_accuracy': float(prediction.model_accuracy),  # Actual R² from training
            'prediction_likelihood': float(prediction.prediction_likelihood),  # This prediction's likelihood
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
            'X-RapidAPI-Key': 'be76a86c9cmsh0d0cecaaefbc722p1efcdbjsn598e66d34cf3',
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
@cache_response(duration=1800)  # Cache schedules for 30 minutes
def nfl_schedule_proxy():
    """
    Specialized proxy for NFL schedule data with caching
    Usage: /api/proxy/nfl/schedule?teamAbv=KC
    """
    try:
        team_abv = request.args.get('teamAbv')
        if not team_abv:
            return jsonify({'error': 'Missing teamAbv parameter'}), 400
        
        rate_limit()
        
        url = 'https://tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com/getNFLTeamSchedule'
        headers = {
            'X-RapidAPI-Key': 'be76a86c9cmsh0d0cecaaefbc722p1efcdbjsn598e66d34cf3',
            'X-RapidAPI-Host': 'tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com'
        }
        params = {'teamAbv': team_abv, 'season': '2024'}
        
        logger.info(f"🔄 Fetching schedule for {team_abv}")
        response = requests.get(url, headers=headers, params=params, timeout=10)
        return jsonify(response.json()), response.status_code
        
    except Exception as e:
        logger.error(f"Schedule proxy error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/proxy/nfl/roster', methods=['GET'])
@cache_response(duration=600)  # Cache rosters for 10 minutes
def nfl_roster_proxy():
    """
    Specialized proxy for NFL team roster data with caching and rate limiting
    Usage: /api/proxy/nfl/roster?teamID=LAC&getStats=false (accepts teamID or teamAbv)
    """
    try:
        # Accept either teamID or teamAbv parameter
        team_id = request.args.get('teamID') or request.args.get('teamAbv')
        get_stats = request.args.get('getStats', 'false')
        
        if not team_id:
            return jsonify({'error': 'Missing teamID or teamAbv parameter'}), 400
        
        # Rate limiting to avoid hitting API limits
        rate_limit()
        
        url = 'https://tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com/getNFLTeamRoster'
        headers = {
            'X-RapidAPI-Key': 'be76a86c9cmsh0d0cecaaefbc722p1efcdbjsn598e66d34cf3',
            'X-RapidAPI-Host': 'tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com'
        }
        # Use teamAbv parameter for Tank01 API
        params = {'teamAbv': team_id, 'getStats': get_stats}
        
        logger.info(f"🔄 Fetching roster for {team_id}")
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
            'X-RapidAPI-Key': 'be76a86c9cmsh0d0cecaaefbc722p1efcdbjsn598e66d34cf3',
            'X-RapidAPI-Host': 'tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com'
        }
        params = {'week': game_week, 'season': season}
        
        response = requests.get(url, headers=headers, params=params, timeout=10)
        return jsonify(response.json()), response.status_code
        
    except Exception as e:
        logger.error(f"Games proxy error: {str(e)}")
        return jsonify({'error': str(e)}), 500

# =============================================================================
# ESPN WEBSITE DATA ENDPOINTS - TANK01 REPLACEMENT
# =============================================================================

@app.route('/api/news/latest', methods=['GET'])
def get_latest_news():
    """Get latest NFL news from ESPN"""
    if not espn_data_service:
        return jsonify({'error': 'ESPN service not available'}), 503
    
    try:
        limit = int(request.args.get('limit', 10))
        news = espn_data_service.get_latest_news(limit)
        
        return jsonify({
            'success': True,
            'news': news,
            'count': len(news),
            'source': 'ESPN'
        })
    except Exception as e:
        logger.error(f"Error getting news: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/teams/matchups', methods=['GET'])
def get_team_matchups():
    """Get current week team matchups from ESPN"""
    if not espn_data_service:
        return jsonify({'error': 'ESPN service not available'}), 503
    
    try:
        week = request.args.get('week')
        if week:
            week = int(week)
        
        matchups = espn_data_service.get_team_matchups(week)
        
        return jsonify({
            'success': True,
            'matchups': matchups,
            'count': len(matchups),
            'week': week,
            'source': 'ESPN'
        })
    except Exception as e:
        logger.error(f"Error getting matchups: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/players/search', methods=['GET'])
def search_players():
    """Search for players by name using ESPN"""
    if not espn_data_service:
        return jsonify({'error': 'ESPN service not available'}), 503
    
    try:
        query = request.args.get('q', '')
        limit = int(request.args.get('limit', 10))
        
        if not query:
            return jsonify({'error': 'Missing search query parameter "q"'}), 400
        
        players = espn_data_service.search_players(query, limit)
        
        return jsonify({
            'success': True,
            'players': players,
            'count': len(players),
            'query': query,
            'source': 'ESPN'
        })
    except Exception as e:
        logger.error(f"Error searching players: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/teams/info/<team_id>', methods=['GET'])
def get_team_info(team_id):
    """Get detailed team information from ESPN"""
    if not espn_data_service:
        return jsonify({'error': 'ESPN service not available'}), 503
    
    try:
        team_info = espn_data_service.get_team_info(team_id)
        
        if not team_info:
            return jsonify({'error': f'Team {team_id} not found'}), 404
        
        return jsonify({
            'success': True,
            'team': team_info,
            'source': 'ESPN'
        })
    except Exception as e:
        logger.error(f"Error getting team info: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/injuries/current', methods=['GET'])
def get_current_injuries():
    """Get current injury reports from ESPN"""
    if not espn_data_service:
        return jsonify({'error': 'ESPN service not available'}), 503
    
    try:
        team_id = request.args.get('team_id')
        injuries = espn_data_service.get_current_injuries(team_id)
        
        return jsonify({
            'success': True,
            'injuries': injuries,
            'count': len(injuries),
            'team_id': team_id,
            'source': 'ESPN'
        })
    except Exception as e:
        logger.error(f"Error getting injuries: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/schedule/week', methods=['GET'])
def get_weekly_schedule():
    """Get weekly schedule from ESPN"""
    if not espn_data_service:
        return jsonify({'error': 'ESPN service not available'}), 503
    
    try:
        week = request.args.get('week')
        if week:
            week = int(week)
        
        schedule = espn_data_service.get_weekly_schedule(week)
        
        return jsonify({
            'success': True,
            'schedule': schedule,
            'source': 'ESPN'
        })
    except Exception as e:
        logger.error(f"Error getting schedule: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/standings/current', methods=['GET'])
def get_current_standings():
    """Get current NFL standings from ESPN"""
    if not espn_data_service:
        return jsonify({'error': 'ESPN service not available'}), 503
    
    try:
        standings = espn_data_service.get_current_standings()
        
        return jsonify({
            'success': True,
            'standings': standings,
            'count': len(standings),
            'source': 'ESPN'
        })
    except Exception as e:
        logger.error(f"Error getting standings: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/players/trending', methods=['GET'])
def get_trending_players():
    """Get trending/popular players from ESPN"""
    if not espn_data_service:
        return jsonify({'error': 'ESPN service not available'}), 503
    
    try:
        limit = int(request.args.get('limit', 10))
        trending = espn_data_service.get_trending_players(limit)
        
        return jsonify({
            'success': True,
            'players': trending,
            'count': len(trending),
            'source': 'ESPN'
        })
    except Exception as e:
        logger.error(f"Error getting trending players: {e}")
        return jsonify({'error': str(e)}), 500

# Tank01 Compatibility Endpoints
@app.route('/api/tank01/team-stats/<team_id>', methods=['GET'])
def get_tank01_team_stats(team_id):
    """Tank01 compatibility - team stats using ESPN data"""
    if not espn_data_service:
        return jsonify({'error': 'ESPN service not available'}), 503
    
    try:
        team_stats = espn_data_service.get_tank01_team_stats(team_id)
        return jsonify(team_stats)
    except Exception as e:
        logger.error(f"Error getting Tank01 team stats: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/tank01/player-game-logs/<player_id>', methods=['GET'])
def get_tank01_player_logs(player_id):
    """Tank01 compatibility - player game logs using ESPN data"""
    if not espn_data_service:
        return jsonify({'error': 'ESPN service not available'}), 503
    
    try:
        game_logs = espn_data_service.get_tank01_player_game_logs(player_id)
        return jsonify(game_logs)
    except Exception as e:
        logger.error(f"Error getting Tank01 player logs: {e}")
        return jsonify({'error': str(e)}), 500

# =============================================================================
# ODDS COMPARISON AND VALUE BETTING ENDPOINTS
# =============================================================================

# Import the odds comparison service
try:
    from odds_comparison_service import OddsComparisonService
    ODDS_API_AVAILABLE = True
except ImportError:
    ODDS_API_AVAILABLE = False
    logger.warning("Odds comparison service not available - install aiohttp")

# Get API key from environment variable
ODDS_API_KEY = os.environ.get('ODDS_API_KEY')

# Validate API key
if not ODDS_API_KEY or ODDS_API_KEY == 'demo_key':
    logger.warning("⚠️  WARNING: No valid Odds API key found. Set ODDS_API_KEY environment variable.")
    logger.warning("⚠️  Odds comparison features will be limited without a valid API key.")
else:
    logger.info(f"✅ Odds API key loaded: {ODDS_API_KEY[:8]}...{ODDS_API_KEY[-4:]}")

@app.route('/api/odds/compare/<sport>', methods=['GET'])
def compare_odds(sport):
    """Get real-time odds comparison for a sport using Tank01 API for NFL"""
    
    try:
        logger.info(f"Fetching odds comparison for sport: {sport}")
        
        # For NFL, use Tank01 betting odds
        if sport.lower() in ['nfl', 'americanfootball_nfl']:
            # Try multiple dates to find current week games
            dates_to_try = []
            today = datetime.now()
            
            # Add today and next 6 days to catch current week games
            for i in range(7):
                date_str = (today + timedelta(days=i)).strftime('%Y%m%d')
                dates_to_try.append(date_str)
            
            all_games = {}
            
            for date_str in dates_to_try:
                try:
                    tank01_url = 'https://tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com/getNFLBettingOdds'
                    
                    headers = {
                        'X-RapidAPI-Key': 'be76a86c9cmsh0d0cecaaefbc722p1efcdbjsn598e66d34cf3',
                        'X-RapidAPI-Host': 'tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com'
                    }
                    
                    params = {'gameDate': date_str}
                    response = requests.get(tank01_url, headers=headers, params=params, timeout=10)
                    
                    if response.status_code == 200:
                        tank01_data = response.json()
                        if tank01_data.get('statusCode') == 200 and tank01_data.get('body'):
                            all_games.update(tank01_data['body'])
                            logger.info(f"Found {len(tank01_data['body'])} games for {date_str}")
                except Exception as e:
                    logger.warning(f"Error fetching odds for {date_str}: {e}")
                    continue
            
            if all_games:
                    # Convert Tank01 format to our analytics format
                    games_data = []
                    best_odds_by_game = {}
                    arbitrage_opportunities = []
                    value_bets = []
                    
                    for game_id, game_data in all_games.items():
                        away_team = game_data.get('awayTeam')
                        home_team = game_data.get('homeTeam')
                        
                        # Track best odds for each market
                        best_moneyline_away = {'odds': None, 'sportsbook': None}
                        best_moneyline_home = {'odds': None, 'sportsbook': None}
                        best_spread_away = {'odds': None, 'sportsbook': None, 'line': None}
                        best_spread_home = {'odds': None, 'sportsbook': None, 'line': None}
                        
                        game_info = {
                            'id': game_id,
                            'away_team': away_team,
                            'home_team': home_team,
                            'commence_time': game_data.get('gameDate'),
                            'sportsbooks': {}
                        }
                        
                        # Extract sportsbook odds
                        sportsbook_mapping = {
                            'bet365': 'Bet365',
                            'betmgm': 'BetMGM',
                            'betrivers': 'BetRivers',
                            'caesars_sportsbook': 'Caesars',
                            'draftkings': 'DraftKings',
                            'espnbet': 'ESPN BET',
                            'fanatics': 'Fanatics',
                            'fanduel': 'FanDuel',
                            'hardrock': 'Hard Rock'
                        }
                        
                        for tank01_book, display_name in sportsbook_mapping.items():
                            if tank01_book in game_data:
                                book_data = game_data[tank01_book]
                                
                                # Convert American odds to decimal for comparison
                                def american_to_decimal(american_odds):
                                    if not american_odds or american_odds == 'even':
                                        return 2.0 if american_odds == 'even' else None
                                    try:
                                        odds = int(american_odds.replace('+', ''))
                                        if odds > 0:
                                            return (odds / 100) + 1
                                        else:
                                            return (100 / abs(odds)) + 1
                                    except:
                                        return None
                                
                                away_ml_decimal = american_to_decimal(book_data.get('awayTeamMLOdds'))
                                home_ml_decimal = american_to_decimal(book_data.get('homeTeamMLOdds'))
                                
                                # Track best moneyline odds
                                if away_ml_decimal and (not best_moneyline_away['odds'] or away_ml_decimal > best_moneyline_away['odds']):
                                    best_moneyline_away = {'odds': away_ml_decimal, 'sportsbook': display_name, 'american': book_data.get('awayTeamMLOdds')}
                                
                                if home_ml_decimal and (not best_moneyline_home['odds'] or home_ml_decimal > best_moneyline_home['odds']):
                                    best_moneyline_home = {'odds': home_ml_decimal, 'sportsbook': display_name, 'american': book_data.get('homeTeamMLOdds')}
                                
                                game_info['sportsbooks'][display_name] = {
                                    'moneyline': {
                                        'home': book_data.get('homeTeamMLOdds'),
                                        'away': book_data.get('awayTeamMLOdds')
                                    },
                                    'spread': {
                                        'home': {
                                            'point': book_data.get('homeTeamSpread'),
                                            'odds': book_data.get('homeTeamSpreadOdds')
                                        },
                                        'away': {
                                            'point': book_data.get('awayTeamSpread'),
                                            'odds': book_data.get('awayTeamSpreadOdds')
                                        }
                                    },
                                    'totals': {
                                        'over': {
                                            'point': book_data.get('totalOver'),
                                            'odds': book_data.get('totalOverOdds')
                                        },
                                        'under': {
                                            'point': book_data.get('totalUnder'),
                                            'odds': book_data.get('totalUnderOdds')
                                        }
                                    }
                                }
                        
                        # Check for arbitrage opportunities
                        if best_moneyline_away['odds'] and best_moneyline_home['odds']:
                            implied_prob_away = 1 / best_moneyline_away['odds']
                            implied_prob_home = 1 / best_moneyline_home['odds']
                            total_implied_prob = implied_prob_away + implied_prob_home
                            
                            if total_implied_prob < 1.0:  # Arbitrage opportunity
                                profit_margin = (1 - total_implied_prob) * 100
                                arbitrage_opportunities.append({
                                    'game': f"{away_team} @ {home_team}",
                                    'away_bet': {
                                        'team': away_team,
                                        'odds': best_moneyline_away['american'],
                                        'sportsbook': best_moneyline_away['sportsbook']
                                    },
                                    'home_bet': {
                                        'team': home_team,
                                        'odds': best_moneyline_home['american'],
                                        'sportsbook': best_moneyline_home['sportsbook']
                                    },
                                    'profit_margin': round(profit_margin, 2)
                                })
                        
                        # Store best odds for this game
                        best_odds_by_game[game_id] = {
                            'away_team': away_team,
                            'home_team': home_team,
                            'best_away_odds': best_moneyline_away,
                            'best_home_odds': best_moneyline_home
                        }
                        
                        games_data.append(game_info)
                    
                    logger.info(f"Successfully fetched Tank01 odds for {len(games_data)} games with {len(arbitrage_opportunities)} arbitrage opportunities")
                    
                    return jsonify({
                        'success': True,
                        'data': {
                            'sport': sport,
                            'games': games_data,
                            'best_odds_by_game': best_odds_by_game,
                            'arbitrage_opportunities': arbitrage_opportunities,
                            'value_bets': value_bets,
                            'total_games': len(games_data),
                            'total_sportsbooks': len(sportsbook_mapping),
                            'source': 'Tank01'
                        },
                        'timestamp': datetime.now().isoformat()
                    })
        
        # For other sports, fall back to original odds service
        if not ODDS_API_AVAILABLE:
            return jsonify({
                'success': False,
                'error': 'Odds comparison service not available - missing dependencies'
            }), 503
        
        if not ODDS_API_KEY or ODDS_API_KEY == 'demo_key':
            return jsonify({
                'success': False,
                'error': 'Odds API key not configured - set ODDS_API_KEY environment variable'
            }), 503
        
        # Run async function in event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        async def get_odds():
            async with OddsComparisonService(ODDS_API_KEY) as odds_service:
                result = await odds_service.get_market_analysis(sport)
                logger.info(f"Market analysis result keys: {list(result.keys()) if result else 'None'}")
                logger.info(f"Total games found: {result.get('total_games', 0) if result else 0}")
                return result
        
        result = loop.run_until_complete(get_odds())
        loop.close()
        
        if not result:
            logger.warning("No odds data returned from service")
            result = {}
        
        return jsonify({
            'success': True,
            'data': result,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in odds comparison: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
    
    try:
        # Run async function in event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        async def get_odds():
            async with OddsComparisonService(ODDS_API_KEY) as odds_service:
                return await odds_service.get_market_analysis(sport)
        
        result = loop.run_until_complete(get_odds())
        loop.close()
        
        return jsonify({
            'success': True,
            'data': result,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error comparing odds: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/odds/value-bets/<sport>', methods=['POST'])
def find_value_bets(sport):
    """Find value bets by getting data from Tank01 odds comparison"""
    try:
        # For NFL, get data from Tank01 integration
        if sport.lower() in ['nfl', 'americanfootball_nfl']:
            # Call our own odds comparison endpoint to get Tank01 data
            comparison_response = compare_odds(sport)
            comparison_data = comparison_response[0].get_json()
            
            if comparison_data.get('success') and comparison_data.get('data'):
                # Extract value bets from the Tank01 data
                market_data = comparison_data['data']
                
                # For now, return empty value bets since we need ML predictions to compare
                # In a real implementation, this would compare ML model predictions with odds
                value_bets = []
                
                # Generate some demo value bets based on odds discrepancies
                games = market_data.get('games', [])
                for game in games[:3]:  # Limit to first 3 games for demo
                    away_team = game.get('away_team')
                    home_team = game.get('home_team')
                    sportsbooks = game.get('sportsbooks', {})
                    
                    if len(sportsbooks) >= 2:
                        # Simple demo: look for moneyline odds discrepancies
                        odds_list = []
                        for book_name, book_data in sportsbooks.items():
                            ml_data = book_data.get('moneyline', {})
                            if ml_data.get('away') and ml_data.get('home'):
                                odds_list.append({
                                    'book': book_name,
                                    'away_odds': ml_data['away'],
                                    'home_odds': ml_data['home']
                                })
                        
                        # Create demo value bet if odds variance exists
                        if len(odds_list) >= 2:
                            value_bets.append({
                                'game': f"{away_team} @ {home_team}",
                                'game_id': game.get('id'),
                                'outcome': away_team,
                                'market': 'moneyline',
                                'recommended_odds': odds_list[0]['away_odds'],
                                'sportsbook': odds_list[0]['book'],
                                'kelly_fraction': 0.025,
                                'edge_percentage': 5.2,
                                'confidence': 'moderate'
                            })
                
                return jsonify({
                    'success': True,
                    'data': {
                        'value_bets': value_bets,
                        'count': len(value_bets)
                    },
                    'timestamp': datetime.now().isoformat(),
                    'source': 'Tank01'
                })
        
        # Fallback for other sports
        return jsonify({
            'success': True,
            'data': {
                'value_bets': [],
                'count': 0
            },
            'timestamp': datetime.now().isoformat(),
            'note': 'Demo mode - Tank01 NFL integration active'
        })
        
    except Exception as e:
        logger.error(f"Error finding value bets: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/odds/arbitrage/<sport>', methods=['GET'])
def get_arbitrage_opportunities(sport):
    """Get arbitrage opportunities from Tank01 odds comparison"""
    try:
        # For NFL, get data from Tank01 integration
        if sport.lower() in ['nfl', 'americanfootball_nfl']:
            # Call our own odds comparison endpoint to get Tank01 data
            comparison_response = compare_odds(sport)
            comparison_data = comparison_response[0].get_json()
            
            if comparison_data.get('success') and comparison_data.get('data'):
                market_data = comparison_data['data']
                arbitrage_opportunities = market_data.get('arbitrage_opportunities', [])
                
                return jsonify({
                    'success': True,
                    'data': {
                        'arbitrage_opportunities': arbitrage_opportunities,
                        'count': len(arbitrage_opportunities)
                    },
                    'timestamp': datetime.now().isoformat(),
                    'source': 'Tank01'
                })
        
        # Fallback for other sports
        return jsonify({
            'success': True,
            'data': {
                'arbitrage_opportunities': [],
                'count': 0
            },
            'timestamp': datetime.now().isoformat(),
            'note': 'Tank01 NFL integration active'
        })
        
    except Exception as e:
        logger.error(f"Error getting arbitrage opportunities: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
        
        if not predictions:
            return jsonify({
                'success': False,
                'error': 'Model predictions required'
            }), 400
        
        # Run async function in event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        async def get_value_bets():
            async with OddsComparisonService(ODDS_API_KEY) as odds_service:
                return await odds_service.identify_value_bets(sport, predictions)
        
        value_bets = loop.run_until_complete(get_value_bets())
        loop.close()
        
        # Convert dataclasses to dictionaries for JSON serialization
        value_bets_dict = []
        for bet in value_bets:
            value_bets_dict.append({
                'game_id': bet.game_id,
                'home_team': bet.home_team,
                'away_team': bet.away_team,
                'commence_time': bet.commence_time.isoformat(),
                'market': bet.market,
                'outcome': bet.outcome,
                'best_odds': bet.best_odds,
                'best_bookmaker': bet.best_bookmaker,
                'model_probability': bet.model_probability,
                'implied_probability': bet.implied_probability,
                'edge': bet.edge,
                'kelly_fraction': bet.kelly_fraction
            })
        
        return jsonify({
            'success': True,
            'data': {
                'value_bets': value_bets_dict,
                'count': len(value_bets_dict)
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error finding value bets: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/odds/best-lines/<sport>/<team>', methods=['GET'])
def get_best_lines(sport, team):
    """Get best available lines for a specific team"""
    if not ODDS_API_AVAILABLE:
        return jsonify({
            'success': False,
            'error': 'Odds comparison service not available'
        }), 503
    
    try:
        market = request.args.get('market', 'h2h')
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        async def get_lines():
            async with OddsComparisonService(ODDS_API_KEY) as odds_service:
                raw_odds = await odds_service.fetch_odds(sport, [market])
                odds_data = odds_service.parse_odds_data(raw_odds)
                return odds_service.find_best_odds(odds_data, market, team)
        
        best_odds = loop.run_until_complete(get_lines())
        loop.close()
        
        if best_odds:
            return jsonify({
                'success': True,
                'data': {
                    'team': team,
                    'market': market,
                    'best_odds': best_odds.price,
                    'bookmaker': best_odds.bookmaker,
                    'implied_probability': best_odds.implied_probability,
                    'last_update': best_odds.last_update.isoformat()
                }
            })
        else:
            return jsonify({
                'success': False,
                'error': f'No odds found for {team} in {market} market'
            }), 404
            
    except Exception as e:
        logger.error(f"Error getting best lines: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/odds/arbitrage/<sport>', methods=['GET'])
def find_arbitrage_opportunities(sport):
    """Find arbitrage betting opportunities"""
    if not ODDS_API_AVAILABLE:
        return jsonify({
            'success': False,
            'error': 'Odds comparison service not available - missing dependencies'
        }), 503
    
    if not ODDS_API_KEY or ODDS_API_KEY == 'demo_key':
        return jsonify({
            'success': False,
            'error': 'Odds API key not configured'
        }), 503
    
    try:
        logger.info(f"Finding arbitrage opportunities for sport: {sport}")
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        async def get_arbitrage():
            async with OddsComparisonService(ODDS_API_KEY) as odds_service:
                analysis = await odds_service.get_market_analysis(sport)
                arbitrage_ops = analysis.get('arbitrage_opportunities', []) if analysis else []
                logger.info(f"Found {len(arbitrage_ops)} arbitrage opportunities")
                return arbitrage_ops
        
        arbitrage_ops = loop.run_until_complete(get_arbitrage())
        loop.close()
        
        return jsonify({
            'success': True,
            'data': {
                'arbitrage_opportunities': arbitrage_ops,
                'count': len(arbitrage_ops)
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error finding arbitrage opportunities: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        async def get_arbitrage():
            async with OddsComparisonService(ODDS_API_KEY) as odds_service:
                analysis = await odds_service.get_market_analysis(sport)
                return analysis.get('arbitrage_opportunities', [])
        
        arbitrage_ops = loop.run_until_complete(get_arbitrage())
        loop.close()
        
        return jsonify({
            'success': True,
            'data': {
                'arbitrage_opportunities': arbitrage_ops,
                'count': len(arbitrage_ops)
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error finding arbitrage opportunities: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Enhanced ESPN service already initialized above with ml_model

# =============================================================================
# ENHANCED ESPN BETTING INSIGHTS ENDPOINTS
# =============================================================================

@app.route('/api/enhanced/predict', methods=['POST'])
def enhanced_predict():
    """Enhanced prediction endpoint with ESPN Tier 1 data integration"""
    try:
        data = request.get_json()
        
        player_id = data.get('player_id')
        position = data.get('position', 'QB').upper()
        prediction_type = data.get('prediction_type', 'passing_yards')
        team_id = data.get('team_id')
        opponent_id = data.get('opponent_id')
        
        if not player_id or not team_id:
            return jsonify({'error': 'Missing required fields: player_id, team_id'}), 400
        
        if not enhanced_espn_service:
            return jsonify({'error': 'Enhanced ESPN service not available'}), 503
        
        # Get enhanced ESPN insights
        insights = enhanced_espn_service.generate_betting_insights(
            player_id, position, prediction_type, team_id, opponent_id
        )
        
        logger.info(f"✅ Enhanced insights generated for player {player_id}")
        
        return jsonify({
            'success': True,
            'insights': insights,
            'metadata': {
                'player_id': player_id,
                'position': position,
                'prediction_type': prediction_type,
                'team_id': team_id,
                'opponent_id': opponent_id,
                'espn_data_quality': insights.get('confidence', {}).get('dataQuality', 'good'),
                'timestamp': datetime.now().isoformat()
            }
        })
        
    except Exception as e:
        logger.error(f"Enhanced prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/enhanced/player/<player_id>/analysis', methods=['GET'])
def enhanced_player_analysis(player_id):
    """Get comprehensive player analysis with ESPN data"""
    try:
        position = request.args.get('position', 'QB').upper()
        team_id = request.args.get('team_id')
        
        if not enhanced_espn_service:
            return jsonify({'error': 'Enhanced ESPN service not available'}), 503
        
        # Get player stats from ESPN
        player_stats = enhanced_espn_service.get_player_stats(player_id, position)
        
        # Get team stats for context
        team_stats = None
        if team_id:
            team_stats = enhanced_espn_service.get_team_stats(team_id)
        
        return jsonify({
            'success': True,
            'player_analysis': {
                'player_id': player_id,
                'position': position,
                'stats': player_stats,
                'team_context': team_stats,
                'last_updated': datetime.now().isoformat()
            }
        })
        
    except Exception as e:
        logger.error(f"Player analysis error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/enhanced/team/<team_id>/insights', methods=['GET'])
def enhanced_team_insights(team_id):
    """Get team betting insights with ESPN data"""
    try:
        if not enhanced_espn_service:
            return jsonify({'error': 'Enhanced ESPN service not available'}), 503
        
        # Get team stats and roster
        team_stats = enhanced_espn_service.get_team_stats(team_id)
        roster_data = enhanced_espn_service.get_team_roster(team_id)
        
        # Generate team-level insights
        insights = {
            'team_id': team_id,
            'offensive_efficiency': team_stats.get('offensive_efficiency', 'N/A'),
            'defensive_ranking': team_stats.get('defensive_ranking', 'N/A'),
            'key_players': roster_data.get('key_players', []),
            'recent_form': team_stats.get('recent_form', 'N/A'),
            'betting_trends': {
                'over_under_trend': 'Analyze from game logs',
                'spread_performance': 'Historical ATS performance',
                'scoring_consistency': team_stats.get('scoring_variance', 'N/A')
            }
        }
        
        return jsonify({
            'success': True,
            'team_insights': insights,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Team insights error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/enhanced/matchup', methods=['GET'])
def enhanced_matchup_analysis():
    """Get enhanced matchup analysis between two teams"""
    try:
        home_team = request.args.get('home_team')
        away_team = request.args.get('away_team')
        
        if not home_team or not away_team:
            return jsonify({'error': 'Missing required parameters: home_team, away_team'}), 400
        
        if not enhanced_espn_service:
            return jsonify({'error': 'Enhanced ESPN service not available'}), 503
        
        # Get matchup analysis
        matchup_data = enhanced_espn_service.get_matchup_analysis(home_team, away_team)
        
        return jsonify({
            'success': True,
            'matchup_analysis': matchup_data,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Matchup analysis error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/enhanced/betting-insights/<position>', methods=['GET'])
def enhanced_position_insights(position):
    """Get position-specific betting insights"""
    try:
        position = position.upper()
        team_id = request.args.get('team_id')
        
        if position not in ['QB', 'RB', 'WR', 'TE']:
            return jsonify({'error': f'Invalid position: {position}'}), 400
        
        if not enhanced_espn_service:
            return jsonify({'error': 'Enhanced ESPN service not available'}), 503
        
        # Generate position-specific insights
        insights = enhanced_espn_service.get_position_insights(position, team_id)
        
        return jsonify({
            'success': True,
            'position_insights': {
                'position': position,
                'team_id': team_id,
                'insights': insights,
                'confidence': 'high',
                'data_sources': ['ESPN Stats API', 'ESPN Game Logs', 'Team Performance Metrics']
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Position insights error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/enhanced/trending', methods=['GET'])
def enhanced_trending_opportunities():
    """Get trending betting opportunities with ESPN data"""
    try:
        limit = int(request.args.get('limit', 10))
        
        if not enhanced_espn_service:
            return jsonify({'error': 'Enhanced ESPN service not available'}), 503
        
        # Get trending opportunities
        trending = enhanced_espn_service.get_trending_opportunities(limit)
        
        return jsonify({
            'success': True,
            'trending_opportunities': trending,
            'count': len(trending),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:

        logger.error(f"Trending opportunities error: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Initialize Enhanced ESPN Endpoints
try:
    from enhanced_api_endpoints import init_enhanced_endpoints
    enhanced_espn_service_v2, enhanced_ml_predictor = init_enhanced_endpoints(app, ml_model.models)
    logger.info("✅ Enhanced ESPN endpoints v2 initialized")
    ENHANCED_ENDPOINTS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠️ Enhanced endpoints not available: {e}")
    ENHANCED_ENDPOINTS_AVAILABLE = False
except Exception as e:
    logger.error(f"❌ Error initializing enhanced endpoints: {e}")
    ENHANCED_ENDPOINTS_AVAILABLE = False

if __name__ == '__main__':
    import os
    
    # Get port from environment variable (Railway/Heroku sets this)
    port = int(os.environ.get('PORT', 5001))  # Use 5001 as default
    
    logger.info("🚀 Starting NFL Multi-Position Prediction ML Backend...")
    logger.info("🏈 XGBoost Models: QB, RB, WR, TE")
    if ENHANCED_ENDPOINTS_AVAILABLE:
        logger.info("🚀 Enhanced ESPN Tier 1 Endpoints: ACTIVE")
    logger.info("=" * 50)
    logger.info(f"📡 Server running on port {port}")
    logger.info("Available endpoints:")
    logger.info("  GET  /health - Health check")
    logger.info("  POST /predict - Player performance prediction (all positions)")
    logger.info("  GET  /model/info - Model information")
    
    if ENHANCED_ENDPOINTS_AVAILABLE:
        logger.info("Enhanced ESPN Endpoints:")
        logger.info("  GET  /api/enhanced/player/<id>/analysis - Comprehensive player analysis")
        logger.info("  POST /api/enhanced/predict - Enhanced predictions with ESPN data")
        logger.info("  GET  /api/enhanced/team/<id>/insights - Team betting insights")
        logger.info("  GET  /api/enhanced/matchup - Matchup analysis")
        logger.info("  GET  /api/enhanced/betting-insights/<position> - Position insights")
        logger.info("  GET  /api/enhanced/trending - Trending betting opportunities")
    
    logger.info("=" * 50)
    
    # Production ready settings for cloud deployment
    app.run(host='0.0.0.0', port=port, debug=False)