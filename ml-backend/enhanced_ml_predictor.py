"""
Enhanced ML Model with Comprehensive ESPN Integration
Uses Tier 1 ESPN endpoints for maximum prediction accuracy
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
from enhanced_espn_service import EnhancedESPNService, BettingInsight

logger = logging.getLogger(__name__)

@dataclass
class EnhancedPrediction:
    """Enhanced prediction with comprehensive ESPN data"""
    player_name: str
    position: str
    prediction_type: str
    predicted_value: float
    confidence: float
    range_low: float
    range_high: float
    key_factors: List[str]
    recent_trends: List[str]
    matchup_factors: List[str]
    situational_modifiers: Dict[str, float]
    betting_insights: BettingInsight
    model_features: Dict[str, float]

class EnhancedMLPredictor:
    """
    Enhanced ML predictor with comprehensive ESPN data integration
    """
    
    def __init__(self, models: Dict, espn_service: EnhancedESPNService = None):
        self.models = models
        self.espn_service = espn_service or EnhancedESPNService()
        
        # Feature engineering weights for different data sources
        self.feature_weights = {
            'recent_form': 0.35,      # Last 5 games
            'season_stats': 0.25,     # Season averages
            'situational': 0.20,      # Home/away, matchups
            'team_context': 0.15,     # Team offensive/defensive rankings
            'consistency': 0.05       # Player consistency metrics
        }
        
        logger.info("🧠 Enhanced ML Predictor initialized with ESPN integration")
    
    def predict_enhanced_performance(self, player_id: str, position: str, 
                                   prediction_type: str, team_id: str,
                                   opponent_id: str = None, is_home: bool = True,
                                   weather_conditions: Dict = None) -> EnhancedPrediction:
        """
        Generate enhanced prediction using comprehensive ESPN data
        """
        logger.info(f"🎯 Enhanced prediction for Player {player_id} ({position}) - {prediction_type}")
        
        # Gather comprehensive data from ESPN
        player_stats = self.espn_service.get_player_stats(player_id)
        game_logs = self.espn_service.get_player_game_log(player_id)
        player_splits = self.espn_service.get_player_splits(player_id)
        team_stats = self.espn_service.get_team_statistics(team_id)
        opponent_stats = self.espn_service.get_team_statistics(opponent_id) if opponent_id else None
        
        # Generate betting insights
        betting_insights = self.espn_service.generate_betting_insights(
            player_id, prediction_type, opponent_id
        )
        
        # Build comprehensive feature set
        features = self._build_comprehensive_features(
            player_stats, game_logs, player_splits, team_stats, opponent_stats,
            is_home, weather_conditions, position, prediction_type
        )
        
        # Make model prediction
        base_prediction = self._get_model_prediction(position, features, prediction_type)
        
        # Apply situational modifiers
        modified_prediction, modifiers = self._apply_situational_modifiers(
            base_prediction, player_splits, team_stats, opponent_stats, 
            is_home, weather_conditions, game_logs
        )
        
        # Calculate confidence and prediction range
        confidence = self._calculate_confidence(game_logs, player_splits, features)
        prediction_range = self._calculate_prediction_range(
            modified_prediction, confidence, game_logs
        )
        
        # Generate explanatory factors
        key_factors = self._identify_key_factors(features, player_splits, team_stats)
        recent_trends = self._analyze_recent_trends(game_logs, prediction_type)
        matchup_factors = self._analyze_matchup_factors(team_stats, opponent_stats)
        
        return EnhancedPrediction(
            player_name=f"Player_{player_id}",
            position=position,
            prediction_type=prediction_type,
            predicted_value=modified_prediction,
            confidence=confidence,
            range_low=prediction_range[0],
            range_high=prediction_range[1],
            key_factors=key_factors,
            recent_trends=recent_trends,
            matchup_factors=matchup_factors,
            situational_modifiers=modifiers,
            betting_insights=betting_insights,
            model_features=features
        )
    
    def _build_comprehensive_features(self, player_stats: Dict, game_logs: List,
                                    player_splits: Dict, team_stats: Any, 
                                    opponent_stats: Any, is_home: bool,
                                    weather_conditions: Dict, position: str,
                                    prediction_type: str) -> Dict[str, float]:
        """Build comprehensive feature set from all ESPN data sources"""
        
        features = {}
        
        # 1. Recent Form Features (35% weight)
        recent_games = game_logs[:5] if game_logs else []
        if recent_games:
            recent_values = [game.stats.get(prediction_type, 0) for game in recent_games]
            features['recent_avg'] = np.mean(recent_values)
            features['recent_trend'] = self._calculate_trend(recent_values)
            features['recent_consistency'] = 1 - (np.std(recent_values) / max(np.mean(recent_values), 1))
            features['last_game_performance'] = recent_values[0] if recent_values else 0
        else:
            features.update({
                'recent_avg': 0, 'recent_trend': 0, 
                'recent_consistency': 0.5, 'last_game_performance': 0
            })
        
        # 2. Season Statistics Features (25% weight)
        season_stats = player_stats.get('season_stats', {})
        features['season_avg'] = season_stats.get(prediction_type, 0)
        features['games_played'] = season_stats.get('games_played', 0)
        features['season_high'] = max([game.stats.get(prediction_type, 0) for game in game_logs], default=0)
        
        # Position-specific features
        if position == 'QB':
            features['completion_percentage'] = season_stats.get('completion_percentage', 65.0)
            features['qb_rating'] = season_stats.get('passer_rating', 90.0)
            features['yards_per_attempt'] = season_stats.get('yards_per_attempt', 7.0)
        elif position == 'RB':
            features['yards_per_carry'] = season_stats.get('yards_per_carry', 4.0)
            features['rush_attempts'] = season_stats.get('rush_attempts', 15.0)
            features['receiving_targets'] = season_stats.get('receiving_targets', 3.0)
        elif position in ['WR', 'TE']:
            features['targets_per_game'] = season_stats.get('targets_per_game', 6.0)
            features['catch_percentage'] = season_stats.get('catch_percentage', 65.0)
            features['yards_per_reception'] = season_stats.get('yards_per_reception', 12.0)
        
        # 3. Situational Features (20% weight)
        home_away_key = 'home' if is_home else 'away'
        if hasattr(player_splits, 'home_away') and home_away_key in player_splits.home_away:
            features['home_away_modifier'] = player_splits.home_away[home_away_key].get(prediction_type, features['season_avg'])
        else:
            features['home_away_modifier'] = features['season_avg']
        
        features['is_home'] = 1.0 if is_home else 0.0
        
        # Weather modifiers
        if weather_conditions:
            features['temperature'] = weather_conditions.get('temperature', 70)
            features['wind_speed'] = weather_conditions.get('wind_speed', 5)
            features['precipitation'] = 1.0 if weather_conditions.get('precipitation', False) else 0.0
        else:
            features.update({'temperature': 70, 'wind_speed': 5, 'precipitation': 0.0})
        
        # 4. Team Context Features (15% weight)
        if team_stats:
            # Offensive support
            features['team_off_rank'] = team_stats.offensive_rankings.get('total_offense', 16) / 32.0
            features['team_off_efficiency'] = team_stats.efficiency_metrics.get('off_yards_per_play', 5.5)
            
            if position == 'QB':
                features['oline_rank'] = team_stats.offensive_rankings.get('pass_protection', 16) / 32.0
                features['weapons_quality'] = team_stats.efficiency_metrics.get('off_passing_yards', 250) / 300.0
            elif position == 'RB':
                features['oline_rush_rank'] = team_stats.offensive_rankings.get('rushing_offense', 16) / 32.0
                features['goal_line_usage'] = team_stats.red_zone_stats.get('rushing_attempts', 5) / 10.0
            elif position in ['WR', 'TE']:
                features['pass_volume'] = team_stats.efficiency_metrics.get('off_pass_attempts', 35) / 50.0
                features['target_share'] = min(features.get('targets_per_game', 6) / 35.0, 1.0)
        else:
            # Default team context
            features.update({
                'team_off_rank': 0.5, 'team_off_efficiency': 5.5,
                'oline_rank': 0.5, 'weapons_quality': 0.5,
                'oline_rush_rank': 0.5, 'goal_line_usage': 0.5,
                'pass_volume': 0.7, 'target_share': 0.2
            })
        
        # Opponent defense features
        if opponent_stats:
            features['opp_def_rank'] = opponent_stats.defensive_rankings.get('total_defense', 16) / 32.0
            
            if position == 'QB':
                features['opp_pass_def_rank'] = opponent_stats.defensive_rankings.get('pass_defense', 16) / 32.0
                features['opp_sacks_per_game'] = opponent_stats.efficiency_metrics.get('def_sacks_per_game', 2.5)
            elif position == 'RB':
                features['opp_rush_def_rank'] = opponent_stats.defensive_rankings.get('rush_defense', 16) / 32.0
                features['opp_run_stuff_rate'] = opponent_stats.efficiency_metrics.get('def_run_stuff_rate', 0.2)
            elif position in ['WR', 'TE']:
                features['opp_pass_def_rank'] = opponent_stats.defensive_rankings.get('pass_defense', 16) / 32.0
                features['opp_int_rate'] = opponent_stats.turnover_stats.get('interceptions_per_game', 1.0)
        else:
            features.update({
                'opp_def_rank': 0.5, 'opp_pass_def_rank': 0.5, 'opp_sacks_per_game': 2.5,
                'opp_rush_def_rank': 0.5, 'opp_run_stuff_rate': 0.2, 'opp_int_rate': 1.0
            })
        
        # 5. Consistency Features (5% weight)
        if game_logs:
            all_values = [game.stats.get(prediction_type, 0) for game in game_logs]
            features['season_consistency'] = 1 - (np.std(all_values) / max(np.mean(all_values), 1))
            features['floor_performance'] = np.percentile(all_values, 25) if all_values else 0
            features['ceiling_performance'] = np.percentile(all_values, 75) if all_values else 0
        else:
            features.update({'season_consistency': 0.5, 'floor_performance': 0, 'ceiling_performance': 0})
        
        return features
    
    def _get_model_prediction(self, position: str, features: Dict, prediction_type: str) -> float:
        """Get base prediction from XGBoost model"""
        
        if position not in self.models:
            logger.warning(f"No model available for position {position}")
            return features.get('season_avg', 0)
        
        # Convert features to array format expected by model
        # This would need to match the exact feature order used in training
        feature_array = np.array([list(features.values())]).reshape(1, -1)
        
        try:
            prediction = self.models[position].predict(feature_array)[0]
            return max(0, prediction)  # Ensure non-negative
        except Exception as e:
            logger.error(f"Model prediction error: {e}")
            return features.get('season_avg', 0)
    
    def _apply_situational_modifiers(self, base_prediction: float, player_splits: Any,
                                   team_stats: Any, opponent_stats: Any, is_home: bool,
                                   weather_conditions: Dict, game_logs: List) -> Tuple[float, Dict]:
        """Apply situational modifiers to base prediction"""
        
        modifiers = {}
        modified_prediction = base_prediction
        
        # Home/Away modifier
        if hasattr(player_splits, 'home_away'):
            home_away_key = 'home' if is_home else 'away'
            if home_away_key in player_splits.home_away:
                home_away_avg = list(player_splits.home_away[home_away_key].values())
                if home_away_avg:
                    season_avg = sum([sum(game.stats.values()) for game in game_logs]) / max(len(game_logs), 1)
                    modifier = np.mean(home_away_avg) / max(season_avg, 1)
                    modifiers['home_away'] = modifier
                    modified_prediction *= modifier
        
        # Weather modifier (primarily affects passing and kicking)
        if weather_conditions:
            wind_speed = weather_conditions.get('wind_speed', 5)
            precipitation = weather_conditions.get('precipitation', False)
            
            if wind_speed > 15:  # High wind
                modifiers['wind'] = 0.9
                modified_prediction *= 0.9
            
            if precipitation:  # Rain/snow
                modifiers['precipitation'] = 0.85
                modified_prediction *= 0.85
        
        # Recent momentum modifier
        if len(game_logs) >= 3:
            recent_trend = self._calculate_trend([game.stats.get('primary_stat', 0) for game in game_logs[:3]])
            if recent_trend > 0.1:  # Strong upward trend
                modifiers['momentum'] = 1.1
                modified_prediction *= 1.1
            elif recent_trend < -0.1:  # Strong downward trend
                modifiers['momentum'] = 0.9
                modified_prediction *= 0.9
        
        return modified_prediction, modifiers
    
    def _calculate_confidence(self, game_logs: List, player_splits: Any, features: Dict) -> float:
        """Calculate prediction confidence based on data quality and consistency"""
        
        confidence_factors = []
        
        # Data availability factor
        if len(game_logs) >= 8:
            confidence_factors.append(0.9)
        elif len(game_logs) >= 4:
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.5)
        
        # Consistency factor
        consistency = features.get('season_consistency', 0.5)
        confidence_factors.append(consistency)
        
        # Recent form stability
        recent_consistency = features.get('recent_consistency', 0.5)
        confidence_factors.append(recent_consistency)
        
        # Situational data availability
        if hasattr(player_splits, 'home_away') and player_splits.home_away:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.6)
        
        return np.mean(confidence_factors)
    
    def _calculate_prediction_range(self, prediction: float, confidence: float, 
                                  game_logs: List) -> Tuple[float, float]:
        """Calculate prediction range based on historical variance"""
        
        if game_logs:
            all_values = [sum(game.stats.values()) for game in game_logs]
            std_dev = np.std(all_values)
        else:
            std_dev = prediction * 0.3  # Default to 30% of prediction
        
        # Adjust range based on confidence
        range_multiplier = (1 - confidence) + 0.5  # Between 0.5 and 1.5
        
        range_low = max(0, prediction - (std_dev * range_multiplier))
        range_high = prediction + (std_dev * range_multiplier)
        
        return (range_low, range_high)
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend coefficient for recent performance"""
        if len(values) < 2:
            return 0
        
        x = np.arange(len(values))
        coeffs = np.polyfit(x, values, 1)
        return coeffs[0]  # Slope of trend line
    
    def _identify_key_factors(self, features: Dict, player_splits: Any, team_stats: Any) -> List[str]:
        """Identify key factors influencing the prediction"""
        factors = []
        
        # Recent form
        if features.get('recent_trend', 0) > 0.1:
            factors.append("📈 Strong recent momentum")
        elif features.get('recent_trend', 0) < -0.1:
            factors.append("📉 Recent struggles")
        
        # Team context
        team_off_rank = features.get('team_off_rank', 0.5)
        if team_off_rank > 0.8:
            factors.append("🔥 Elite offensive support")
        elif team_off_rank < 0.3:
            factors.append("⚠️ Limited offensive support")
        
        # Matchup
        opp_def_rank = features.get('opp_def_rank', 0.5)
        if opp_def_rank < 0.3:
            factors.append("✅ Favorable matchup vs weak defense")
        elif opp_def_rank > 0.8:
            factors.append("❌ Tough matchup vs elite defense")
        
        # Consistency
        consistency = features.get('season_consistency', 0.5)
        if consistency > 0.8:
            factors.append("🎯 Highly reliable performer")
        elif consistency < 0.4:
            factors.append("🎲 Boom-or-bust potential")
        
        return factors
    
    def _analyze_recent_trends(self, game_logs: List, prediction_type: str) -> List[str]:
        """Analyze recent performance trends"""
        trends = []
        
        if len(game_logs) < 3:
            return ["📊 Limited recent data"]
        
        recent_values = [game.stats.get(prediction_type, 0) for game in game_logs[:5]]
        
        # Streak analysis
        improving_games = 0
        declining_games = 0
        
        for i in range(1, len(recent_values)):
            if recent_values[i-1] > recent_values[i]:
                improving_games += 1
            elif recent_values[i-1] < recent_values[i]:
                declining_games += 1
        
        if improving_games >= 3:
            trends.append("🔥 Hot streak - 3+ improving games")
        elif declining_games >= 3:
            trends.append("❄️ Cold streak - 3+ declining games")
        
        # Volatility analysis
        std_dev = np.std(recent_values)
        mean_val = np.mean(recent_values)
        
        if std_dev / max(mean_val, 1) < 0.2:
            trends.append("📊 Very consistent recent performance")
        elif std_dev / max(mean_val, 1) > 0.5:
            trends.append("📈📉 Highly volatile recent performance")
        
        return trends
    
    def _analyze_matchup_factors(self, team_stats: Any, opponent_stats: Any) -> List[str]:
        """Analyze team vs opponent matchup factors"""
        factors = []
        
        if not team_stats or not opponent_stats:
            return ["📊 Limited matchup data available"]
        
        # Offensive vs defensive rankings comparison
        team_off = getattr(team_stats, 'offensive_rankings', {})
        opp_def = getattr(opponent_stats, 'defensive_rankings', {})
        
        if team_off and opp_def:
            # Compare total offense vs total defense
            off_rank = team_off.get('total_offense', 16)
            def_rank = opp_def.get('total_defense', 16)
            
            if off_rank <= 10 and def_rank >= 20:
                factors.append("🎯 Elite offense vs weak defense")
            elif off_rank >= 20 and def_rank <= 10:
                factors.append("⚔️ Struggling offense vs elite defense")
            elif off_rank <= 10 and def_rank <= 10:
                factors.append("🔥 Elite matchup - top offense vs top defense")
        
        # Red zone efficiency
        team_rz = getattr(team_stats, 'red_zone_stats', {})
        if team_rz.get('red_zone_efficiency', 0) > 0.6:
            factors.append("🏈 Strong red zone offense")
        
        return factors