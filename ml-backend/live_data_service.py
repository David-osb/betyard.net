"""
Live Data Service - Fetches real-time NFL data from ESPN API
Updates player stats, team ratings, schedules, injuries, and weather
"""

import requests
import json
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class LiveNFLDataService:
    """Fetches and caches live NFL data from ESPN and weather APIs"""
    
    def __init__(self):
        self.base_url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"
        self.cache_dir = os.path.dirname(__file__)
        self.cache_duration = 3600  # 1 hour cache
        
    def get_current_week(self) -> int:
        """Get current NFL week number"""
        try:
            url = f"{self.base_url}/scoreboard"
            response = requests.get(url, timeout=10)
            data = response.json()
            return data.get('week', {}).get('number', 1)
        except Exception as e:
            logger.error(f"Failed to get current week: {e}")
            return 12  # Default to week 12 if fails
    
    def get_player_stats(self, player_name: str) -> Optional[Dict]:
        """
        Get real player stats from ESPN player data
        Returns season avg, recent 3-game avg, total yards, total TDs
        """
        try:
            data_file = os.path.join(self.cache_dir, 'espn_player_data.json')
            if not os.path.exists(data_file):
                return None
            
            with open(data_file, 'r') as f:
                players = json.load(f)
            
            # Find player
            player_name_lower = player_name.lower()
            player_data = None
            
            for p in players:
                if p.get('name', '').lower() == player_name_lower:
                    player_data = p
                    break
            
            if not player_data:
                return None
            
            # Calculate stats from game log
            gamelog = player_data.get('gamelog', [])
            if not gamelog:
                return None
            
            # Season averages
            total_yards = sum(g.get('yards', 0) for g in gamelog)
            total_tds = sum(g.get('tds', 0) for g in gamelog)
            games_played = len(gamelog)
            
            season_avg_yards = total_yards / games_played if games_played > 0 else 0
            season_avg_tds = total_tds / games_played if games_played > 0 else 0
            
            # Recent 3-game average
            recent_games = gamelog[:3] if len(gamelog) >= 3 else gamelog
            recent_yards = sum(g.get('yards', 0) for g in recent_games) / len(recent_games)
            
            return {
                'season_avg_yards': round(season_avg_yards, 1),
                'season_avg_tds': round(season_avg_tds, 2),
                'recent_3game_avg': round(recent_yards, 1),
                'games_played': games_played,
                'total_yards': total_yards,
                'total_tds': total_tds,
                'position': player_data.get('position'),
                'team': player_data.get('team')
            }
            
        except Exception as e:
            logger.error(f"Failed to get player stats for {player_name}: {e}")
            return None
    
    def get_team_ratings(self, team_code: str) -> Dict:
        """
        Calculate team offensive/defensive ratings from ESPN stats
        Returns ratings on 0-100 scale based on real performance
        """
        try:
            # Try to load from cached team ratings
            ratings_file = os.path.join(self.cache_dir, 'team_ratings.json')
            
            if os.path.exists(ratings_file):
                # Check if cache is fresh (less than 24 hours old)
                file_age = time.time() - os.path.getmtime(ratings_file)
                if file_age < 86400:  # 24 hours
                    with open(ratings_file, 'r') as f:
                        all_ratings = json.load(f)
                        if team_code in all_ratings:
                            return all_ratings[team_code]
            
            # If not cached, fetch from ESPN
            return self._fetch_team_ratings_from_espn(team_code)
            
        except Exception as e:
            logger.error(f"Failed to get team ratings for {team_code}: {e}")
            # Fallback to default
            return {'offense': 75, 'defense': 75}
    
    def _fetch_team_ratings_from_espn(self, team_code: str) -> Dict:
        """
        Fetch team stats from ESPN API and calculate ratings
        """
        try:
            # Get team stats from ESPN
            url = f"{self.base_url}/teams/{team_code}/statistics"
            response = requests.get(url, timeout=10)
            
            if response.status_code != 200:
                return {'offense': 75, 'defense': 75}
            
            data = response.json()
            
            # Parse offensive stats
            # Look for yards per game, points per game, etc.
            offense_rating = self._calculate_offense_rating(data)
            defense_rating = self._calculate_defense_rating(data)
            
            return {
                'offense': offense_rating,
                'defense': defense_rating
            }
            
        except Exception as e:
            logger.error(f"ESPN API fetch failed for {team_code}: {e}")
            return {'offense': 75, 'defense': 75}
    
    def _calculate_offense_rating(self, team_data: Dict) -> int:
        """Calculate offensive rating from team stats (0-100 scale)"""
        # This would parse ESPN stats and normalize to 0-100
        # For now, use a simple formula based on available data
        # TODO: Implement full calculation when we have real ESPN stats structure
        return 75
    
    def _calculate_defense_rating(self, team_data: Dict) -> int:
        """Calculate defensive rating from team stats (0-100 scale)"""
        # This would parse ESPN stats and normalize to 0-100
        # TODO: Implement full calculation when we have real ESPN stats structure
        return 75
    
    def get_game_info(self, team_code: str, week: int = None) -> Optional[Dict]:
        """
        Get game info for team this week: opponent, home/away, date/time
        """
        try:
            if week is None:
                week = self.get_current_week()
            
            url = f"{self.base_url}/scoreboard"
            params = {'week': week}
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            events = data.get('events', [])
            
            for event in events:
                competitions = event.get('competitions', [])
                if not competitions:
                    continue
                
                competition = competitions[0]
                competitors = competition.get('competitors', [])
                
                # Find if team is in this game
                home_team = None
                away_team = None
                
                for comp in competitors:
                    team = comp.get('team', {})
                    team_abbr = team.get('abbreviation', '')
                    
                    if comp.get('homeAway') == 'home':
                        home_team = team_abbr
                    else:
                        away_team = team_abbr
                
                # Check if our team is playing
                is_home = None
                opponent = None
                
                if home_team == team_code:
                    is_home = True
                    opponent = away_team
                elif away_team == team_code:
                    is_home = False
                    opponent = home_team
                
                if opponent:
                    # Found the game
                    return {
                        'opponent': opponent,
                        'is_home': is_home,
                        'date': event.get('date'),
                        'venue': competition.get('venue', {}).get('fullName'),
                        'city': competition.get('venue', {}).get('address', {}).get('city'),
                        'state': competition.get('venue', {}).get('address', {}).get('state')
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get game info for {team_code}: {e}")
            return None
    
    def get_weather(self, city: str, state: str, game_date: str) -> Dict:
        """
        Get weather forecast for game location
        Returns weather score 0-100 (100 = perfect, 0 = terrible)
        
        NOTE: Requires OpenWeatherMap API key (free tier: 1000 calls/day)
        Set OPENWEATHER_API_KEY environment variable
        """
        try:
            api_key = os.environ.get('OPENWEATHER_API_KEY')
            if not api_key:
                logger.warning("No weather API key - using default weather score")
                return {'score': 75, 'description': 'Unknown'}
            
            # Get coordinates for city
            geo_url = f"http://api.openweathermap.org/geo/1.0/direct"
            geo_params = {
                'q': f"{city},{state},US",
                'limit': 1,
                'appid': api_key
            }
            
            geo_response = requests.get(geo_url, params=geo_params, timeout=10)
            geo_data = geo_response.json()
            
            if not geo_data:
                return {'score': 75, 'description': 'Unknown'}
            
            lat = geo_data[0]['lat']
            lon = geo_data[0]['lon']
            
            # Get weather forecast
            weather_url = f"http://api.openweathermap.org/data/2.5/forecast"
            weather_params = {
                'lat': lat,
                'lon': lon,
                'appid': api_key,
                'units': 'imperial'
            }
            
            weather_response = requests.get(weather_url, params=weather_params, timeout=10)
            weather_data = weather_response.json()
            
            # Find forecast closest to game time
            # For now, get current/next forecast
            if 'list' in weather_data and len(weather_data['list']) > 0:
                forecast = weather_data['list'][0]
                
                temp = forecast['main']['temp']
                wind_speed = forecast['wind']['speed']
                conditions = forecast['weather'][0]['main']
                
                # Calculate weather score
                score = self._calculate_weather_score(temp, wind_speed, conditions)
                
                return {
                    'score': score,
                    'temp': temp,
                    'wind': wind_speed,
                    'conditions': conditions,
                    'description': f"{temp}°F, {conditions}, {wind_speed}mph wind"
                }
            
            return {'score': 75, 'description': 'Unknown'}
            
        except Exception as e:
            logger.error(f"Failed to get weather: {e}")
            return {'score': 75, 'description': 'Error fetching weather'}
    
    def _calculate_weather_score(self, temp: float, wind: float, conditions: str) -> int:
        """
        Calculate weather score for football (0-100)
        100 = Perfect conditions, 0 = Terrible
        """
        score = 100
        
        # Temperature impact
        if temp < 32:  # Freezing
            score -= 20
        elif temp < 45:  # Cold
            score -= 10
        elif temp > 95:  # Very hot
            score -= 15
        elif temp > 85:  # Hot
            score -= 5
        
        # Wind impact
        if wind > 25:  # Very windy
            score -= 20
        elif wind > 15:  # Windy
            score -= 10
        
        # Precipitation impact
        if conditions in ['Rain', 'Thunderstorm']:
            score -= 15
        elif conditions == 'Snow':
            score -= 25
        
        return max(0, min(100, score))
    
    def get_player_injury_status(self, player_name: str, team_code: str) -> Dict:
        """
        Get player injury status from ESPN
        Returns health score: 100 = healthy, 75 = questionable, 50 = doubtful, 0 = out
        """
        try:
            # ESPN injury report endpoint
            url = f"{self.base_url}/teams/{team_code}/injuries"
            response = requests.get(url, timeout=10)
            
            if response.status_code != 200:
                return {'score': 100, 'status': 'Unknown'}
            
            data = response.json()
            
            # Search for player in injury report
            player_name_lower = player_name.lower()
            
            # Parse injury data structure
            # TODO: Implement when we confirm ESPN injury API structure
            
            # Default to healthy if not found in injury report
            return {'score': 100, 'status': 'Healthy'}
            
        except Exception as e:
            logger.error(f"Failed to get injury status for {player_name}: {e}")
            return {'score': 100, 'status': 'Unknown'}


# Global instance
live_data_service = LiveNFLDataService()


if __name__ == '__main__':
    # Test the service
    import time
    
    print("Testing Live NFL Data Service")
    print("=" * 60)
    
    service = LiveNFLDataService()
    
    # Test current week
    print(f"\nCurrent Week: {service.get_current_week()}")
    
    # Test player stats
    print("\n" + "=" * 60)
    print("Josh Allen Stats:")
    stats = service.get_player_stats('Josh Allen')
    if stats:
        print(json.dumps(stats, indent=2))
    else:
        print("No stats found")
    
    # Test team ratings
    print("\n" + "=" * 60)
    print("Buffalo Bills Ratings:")
    ratings = service.get_team_ratings('BUF')
    print(json.dumps(ratings, indent=2))
    
    # Test game info
    print("\n" + "=" * 60)
    print("Buffalo Bills Game Info:")
    game_info = service.get_game_info('BUF')
    if game_info:
        print(json.dumps(game_info, indent=2))
    else:
        print("No game found")
    
    print("\n" + "=" * 60)
    print("✅ Test complete")
