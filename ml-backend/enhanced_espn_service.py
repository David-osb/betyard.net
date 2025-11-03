"""
Enhanced ESPN API Service for Comprehensive Betting Insights
Uses Tier 1 ESPN endpoints for maximum accuracy and comprehensive data
"""

import requests
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class PlayerGameLog:
    """Player game log entry"""
    date: str
    opponent: str
    stats: Dict[str, float]
    is_home: bool
    game_result: str

@dataclass
class PlayerSplits:
    """Player situational splits"""
    home_away: Dict[str, Dict]
    vs_division: Dict[str, Dict]
    weather_conditions: Dict[str, Dict]
    recent_form: Dict[str, Dict]

@dataclass
class TeamStats:
    """Comprehensive team statistics"""
    offensive_rankings: Dict[str, int]
    defensive_rankings: Dict[str, int]
    efficiency_metrics: Dict[str, float]
    red_zone_stats: Dict[str, float]
    turnover_stats: Dict[str, float]

@dataclass
class BettingInsight:
    """Enhanced betting insight with ESPN data"""
    player_name: str
    position: str
    prediction_type: str
    predicted_value: float
    confidence: float
    key_factors: List[str]
    recent_trends: List[str]
    matchup_advantages: List[str]
    injury_considerations: List[str]
    weather_impact: Optional[str]
    betting_recommendation: str

class EnhancedESPNService:
    """
    Enhanced ESPN API service using Tier 1 endpoints for comprehensive betting insights
    """
    
    def __init__(self):
        # Use the official ESPN API endpoints from the document
        self.base_url = "https://site.web.api.espn.com/apis"
        self.common_url = f"{self.base_url}/common/v3/sports/football/nfl"
        self.site_url = f"{self.base_url}/site/v2/sports/football/nfl"
        
        self.session = requests.Session()
        self.cache = {}
        self.cache_duration = 300  # 5 minutes for most data
        self.player_cache_duration = 1800  # 30 minutes for player stats
        
        logger.info("🚀 Enhanced ESPN Service initialized with Tier 1 endpoints")
    
    def _make_request(self, endpoint: str, params: Dict = None, cache_duration: int = None) -> Dict:
        """Make cached API request with configurable cache duration"""
        cache_key = f"{endpoint}_{str(params)}"
        cache_ttl = cache_duration or self.cache_duration
        
        if cache_key in self.cache:
            cached_time, data = self.cache[cache_key]
            if time.time() - cached_time < cache_ttl:
                return data
        
        try:
            if endpoint.startswith('http'):
                url = endpoint
            elif '/athletes/' in endpoint or '/teams/' in endpoint:
                url = f"{self.common_url}/{endpoint}"
            else:
                url = f"{self.site_url}/{endpoint}"
            
            response = self.session.get(url, params=params or {}, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            self.cache[cache_key] = (time.time(), data)
            return data
            
        except Exception as e:
            logger.error(f"ESPN API Error for {endpoint}: {e}")
            return {}
    
    def get_player_stats(self, player_id: str) -> Dict[str, Any]:
        """
        TIER 1: Get comprehensive player statistics
        Endpoint: /athletes/{player_id}/stats
        """
        endpoint = f"athletes/{player_id}/stats"
        data = self._make_request(endpoint, cache_duration=self.player_cache_duration)
        
        if not data:
            return {}
        
        # Parse career and seasonal stats
        stats_data = {
            'career_stats': {},
            'season_stats': {},
            'splits': {},
            'rankings': {}
        }
        
        # Extract different stat categories
        for split in data.get('splits', {}).get('categories', []):
            category_name = split.get('name', '')
            category_stats = {}
            
            for stat in split.get('stats', []):
                stat_name = stat.get('name', '')
                stat_value = stat.get('value', 0)
                category_stats[stat_name] = stat_value
            
            if 'career' in category_name.lower():
                stats_data['career_stats'].update(category_stats)
            elif 'season' in category_name.lower() or datetime.now().year in category_name:
                stats_data['season_stats'].update(category_stats)
        
        return stats_data
    
    def get_player_game_log(self, player_id: str, season: int = None) -> List[PlayerGameLog]:
        """
        TIER 1: Get player game-by-game performance
        Endpoint: /athletes/{player_id}/gamelog
        """
        endpoint = f"athletes/{player_id}/gamelog"
        params = {'season': season or datetime.now().year}
        data = self._make_request(endpoint, params, cache_duration=self.player_cache_duration)
        
        game_logs = []
        
        for event in data.get('events', []):
            # Parse game details
            competition = event.get('competition', {})
            competitors = competition.get('competitors', [])
            
            # Determine home/away and opponent
            player_team_id = data.get('athlete', {}).get('team', {}).get('id')
            is_home = False
            opponent = "Unknown"
            
            for competitor in competitors:
                team = competitor.get('team', {})
                if team.get('id') == player_team_id:
                    is_home = competitor.get('homeAway') == 'home'
                else:
                    opponent = team.get('displayName', 'Unknown')
            
            # Parse stats
            stats = {}
            for stat_category in event.get('stats', []):
                for stat in stat_category.get('stats', []):
                    stat_name = stat.get('name', '')
                    stat_value = stat.get('value', 0)
                    try:
                        stats[stat_name] = float(stat_value)
                    except (ValueError, TypeError):
                        stats[stat_name] = 0
            
            game_log = PlayerGameLog(
                date=event.get('date', ''),
                opponent=opponent,
                stats=stats,
                is_home=is_home,
                game_result=competition.get('status', {}).get('type', {}).get('description', '')
            )
            game_logs.append(game_log)
        
        return game_logs
    
    def get_player_splits(self, player_id: str) -> PlayerSplits:
        """
        TIER 2: Get situational performance splits
        Endpoint: /athletes/{player_id}/splits
        """
        endpoint = f"athletes/{player_id}/splits"
        data = self._make_request(endpoint, cache_duration=self.player_cache_duration)
        
        splits = PlayerSplits(
            home_away={},
            vs_division={},
            weather_conditions={},
            recent_form={}
        )
        
        # Parse different split categories
        for category in data.get('categories', []):
            category_name = category.get('name', '').lower()
            category_stats = {}
            
            for stat in category.get('stats', []):
                stat_name = stat.get('name', '')
                stat_value = stat.get('value', 0)
                try:
                    category_stats[stat_name] = float(stat_value)
                except (ValueError, TypeError):
                    category_stats[stat_name] = 0
            
            if 'home' in category_name or 'away' in category_name:
                splits.home_away[category_name] = category_stats
            elif 'division' in category_name:
                splits.vs_division[category_name] = category_stats
            elif 'weather' in category_name or 'outdoor' in category_name:
                splits.weather_conditions[category_name] = category_stats
            elif 'last' in category_name or 'recent' in category_name:
                splits.recent_form[category_name] = category_stats
        
        return splits
    
    def get_team_statistics(self, team_id: str = None) -> Dict[str, TeamStats]:
        """
        TIER 1: Get comprehensive team statistics
        Endpoint: /statistics/byteam
        """
        endpoint = "statistics/byteam"
        data = self._make_request(endpoint, cache_duration=600)  # 10 minutes cache
        
        team_stats = {}
        
        for team_data in data.get('statistics', []):
            team_info = team_data.get('team', {})
            team_id_current = str(team_info.get('id', ''))
            team_name = team_info.get('displayName', '')
            
            # Skip if we're looking for a specific team and this isn't it
            if team_id and team_id_current != str(team_id):
                continue
            
            # Parse offensive and defensive stats
            offensive_rankings = {}
            defensive_rankings = {}
            efficiency_metrics = {}
            red_zone_stats = {}
            turnover_stats = {}
            
            for category in team_data.get('statistics', []):
                category_name = category.get('name', '').lower()
                
                for stat in category.get('stats', []):
                    stat_name = stat.get('name', '')
                    stat_value = stat.get('value', 0)
                    rank = stat.get('rank', 999)
                    
                    try:
                        stat_value = float(stat_value)
                    except (ValueError, TypeError):
                        stat_value = 0
                    
                    # Categorize stats
                    if 'offense' in category_name or 'passing' in category_name or 'rushing' in category_name:
                        offensive_rankings[stat_name] = rank
                        efficiency_metrics[f"off_{stat_name}"] = stat_value
                    elif 'defense' in category_name:
                        defensive_rankings[stat_name] = rank
                        efficiency_metrics[f"def_{stat_name}"] = stat_value
                    elif 'red zone' in category_name:
                        red_zone_stats[stat_name] = stat_value
                    elif 'turnover' in category_name or 'interception' in category_name or 'fumble' in category_name:
                        turnover_stats[stat_name] = stat_value
            
            team_stats[team_id_current] = TeamStats(
                offensive_rankings=offensive_rankings,
                defensive_rankings=defensive_rankings,
                efficiency_metrics=efficiency_metrics,
                red_zone_stats=red_zone_stats,
                turnover_stats=turnover_stats
            )
        
        return team_stats if not team_id else team_stats.get(str(team_id), TeamStats({}, {}, {}, {}, {}))
    
    def get_roster_with_depth(self, team_id: str) -> Dict[str, List[Dict]]:
        """
        TIER 1: Get team roster with depth chart information
        Endpoint: /teams/{team_id}/roster
        """
        endpoint = f"teams/{team_id}/roster"
        data = self._make_request(endpoint)
        
        roster_by_position = {}
        
        for athlete in data.get('athletes', []):
            position = athlete.get('position', {}).get('abbreviation', 'UNKNOWN')
            
            player_info = {
                'id': athlete.get('id'),
                'name': athlete.get('displayName', ''),
                'jersey': athlete.get('jersey', ''),
                'status': athlete.get('status', {}),
                'experience': athlete.get('experience', 0),
                'age': athlete.get('age', 0),
                'height': athlete.get('height', 0),
                'weight': athlete.get('weight', 0),
                'depth_rank': len(roster_by_position.get(position, [])) + 1  # Depth based on roster order
            }
            
            if position not in roster_by_position:
                roster_by_position[position] = []
            
            roster_by_position[position].append(player_info)
        
        return roster_by_position
    
    def generate_betting_insights(self, player_id: str, prediction_type: str, 
                                matchup_team_id: str = None) -> BettingInsight:
        """
        Generate comprehensive betting insights using all Tier 1 ESPN data
        """
        logger.info(f"🔍 Generating betting insights for player {player_id}")
        
        # Gather all available data
        player_stats = self.get_player_stats(player_id)
        game_logs = self.get_player_game_log(player_id)
        player_splits = self.get_player_splits(player_id)
        
        # Analyze recent form (last 5 games)
        recent_games = game_logs[:5] if game_logs else []
        recent_stats = []
        for game in recent_games:
            if prediction_type in game.stats:
                recent_stats.append(game.stats[prediction_type])
        
        # Calculate prediction value and confidence
        if recent_stats:
            recent_avg = np.mean(recent_stats)
            recent_std = np.std(recent_stats) if len(recent_stats) > 1 else 0
            season_avg = player_stats.get('season_stats', {}).get(prediction_type, recent_avg)
            
            # Weight recent form more heavily
            predicted_value = (recent_avg * 0.7) + (season_avg * 0.3)
            confidence = max(0.5, 1.0 - (recent_std / max(recent_avg, 1)) * 0.5)
        else:
            predicted_value = player_stats.get('season_stats', {}).get(prediction_type, 0)
            confidence = 0.6
        
        # Analyze trends and factors
        key_factors = []
        recent_trends = []
        matchup_advantages = []
        injury_considerations = []
        
        # Recent form analysis
        if len(recent_stats) >= 3:
            if recent_stats[-1] > recent_stats[-3]:
                recent_trends.append("📈 Trending upward in recent games")
            elif recent_stats[-1] < recent_stats[-3]:
                recent_trends.append("📉 Declining performance lately")
            else:
                recent_trends.append("➡️ Consistent recent performance")
        
        # Home/Away splits analysis
        if player_splits.home_away:
            home_avg = player_splits.home_away.get('home', {}).get(prediction_type, 0)
            away_avg = player_splits.home_away.get('away', {}).get(prediction_type, 0)
            
            if home_avg > away_avg * 1.1:
                matchup_advantages.append(f"🏠 Significantly better at home (+{((home_avg/away_avg)-1)*100:.1f}%)")
            elif away_avg > home_avg * 1.1:
                matchup_advantages.append(f"✈️ Road warrior (+{((away_avg/home_avg)-1)*100:.1f}% better away)")
        
        # Consistency analysis
        if recent_stats:
            consistency = 1 - (np.std(recent_stats) / max(np.mean(recent_stats), 1))
            if consistency > 0.8:
                key_factors.append("🎯 Highly consistent performer")
            elif consistency < 0.5:
                key_factors.append("🎲 Volatile performer - higher risk/reward")
        
        # Generate betting recommendation
        if confidence > 0.8:
            betting_recommendation = f"🟢 STRONG BET: High confidence in {predicted_value:.1f}"
        elif confidence > 0.65:
            betting_recommendation = f"🟡 MODERATE BET: Good value at {predicted_value:.1f}"
        else:
            betting_recommendation = f"🔴 PROCEED WITH CAUTION: Low confidence"
        
        return BettingInsight(
            player_name=f"Player_{player_id}",  # Would get from roster data
            position="",  # Would get from roster data
            prediction_type=prediction_type,
            predicted_value=predicted_value,
            confidence=confidence,
            key_factors=key_factors,
            recent_trends=recent_trends,
            matchup_advantages=matchup_advantages,
            injury_considerations=injury_considerations,
            weather_impact=None,  # Would integrate weather data
            betting_recommendation=betting_recommendation
        )
    
    def get_comprehensive_player_analysis(self, player_id: str) -> Dict[str, Any]:
        """
        Get complete player analysis using all Tier 1 endpoints
        """
        analysis = {
            'basic_stats': self.get_player_stats(player_id),
            'game_logs': self.get_player_game_log(player_id),
            'situational_splits': self.get_player_splits(player_id),
            'generated_at': datetime.now().isoformat()
        }
        
        # Add derived insights
        game_logs = analysis['game_logs']
        if game_logs:
            # Calculate momentum
            recent_performance = [game.stats for game in game_logs[:3]]
            season_performance = [game.stats for game in game_logs]
            
            analysis['momentum_indicators'] = {
                'games_analyzed': len(game_logs),
                'recent_games': len(recent_performance),
                'home_vs_away_splits': len([g for g in game_logs if g.is_home]) / max(len(game_logs), 1),
                'consistency_score': self._calculate_consistency(season_performance)
            }
        
        return analysis
    
    def _calculate_consistency(self, performance_data: List[Dict]) -> float:
        """Calculate player consistency score"""
        if not performance_data:
            return 0.0
        
        # Get all numeric stats
        all_stats = {}
        for game in performance_data:
            for stat_name, stat_value in game.items():
                if isinstance(stat_value, (int, float)):
                    if stat_name not in all_stats:
                        all_stats[stat_name] = []
                    all_stats[stat_name].append(stat_value)
        
        # Calculate coefficient of variation for each stat
        consistency_scores = []
        for stat_name, values in all_stats.items():
            if len(values) > 1:
                mean_val = np.mean(values)
                std_val = np.std(values)
                if mean_val > 0:
                    cv = std_val / mean_val
                    consistency_scores.append(1 - min(cv, 1))  # Invert CV and cap at 1
        
        return np.mean(consistency_scores) if consistency_scores else 0.5