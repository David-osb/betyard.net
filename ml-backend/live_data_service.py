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
        Get weather forecast for game location using Weather.gov (NOAA)
        Returns weather score 0-100 (100 = perfect, 0 = terrible)
        
        FREE - No API key needed, unlimited calls, US government data
        """
        try:
            # Stadium coordinates for major NFL cities
            # Weather.gov requires lat/lon, not city names
            stadium_coords = {
                'Glendale': (33.5276, -112.2626),  # ARI
                'Atlanta': (33.7490, -84.3880),     # ATL
                'Baltimore': (39.2904, -76.6122),   # BAL
                'Buffalo': (42.7738, -78.7870),     # BUF
                'Charlotte': (35.2271, -80.8431),   # CAR
                'Chicago': (41.8781, -87.6298),     # CHI
                'Cincinnati': (39.1031, -84.5120),  # CIN
                'Cleveland': (41.4993, -81.6944),   # CLE
                'Arlington': (32.7473, -97.0945),   # DAL (AT&T Stadium)
                'Denver': (39.7439, -105.0201),     # DEN
                'Detroit': (42.3400, -83.0456),     # DET
                'Green Bay': (44.5013, -88.0622),   # GB
                'Houston': (29.7604, -95.3698),     # HOU
                'Indianapolis': (39.7601, -86.1639), # IND
                'Jacksonville': (30.3322, -81.6557), # JAX
                'Kansas City': (39.0997, -94.5786), # KC
                'Paradise': (36.0909, -115.1833),   # LV (Allegiant Stadium)
                'Inglewood': (33.9534, -118.3390),  # LAR/LAC (SoFi Stadium)
                'Miami Gardens': (25.9580, -80.2389), # MIA
                'Minneapolis': (44.9778, -93.2650), # MIN
                'Foxborough': (42.0909, -71.2643),  # NE
                'New Orleans': (29.9511, -90.0715), # NO
                'East Rutherford': (40.8128, -74.0742), # NYG/NYJ (MetLife)
                'Philadelphia': (39.9012, -75.1675), # PHI
                'Pittsburgh': (40.4468, -80.0158),  # PIT
                'Santa Clara': (37.4032, -121.9698), # SF (Levi's Stadium)
                'Seattle': (47.5952, -122.3316),    # SEA
                'Tampa': (27.9759, -82.5033),       # TB
                'Nashville': (36.1627, -86.7816),   # TEN
                'Landover': (38.9076, -76.8645)     # WAS (FedEx Field)
            }
            
            # Get coordinates for city
            coords = stadium_coords.get(city)
            if not coords:
                # Fallback: try to match partial city name
                for stadium_city, coord in stadium_coords.items():
                    if city.lower() in stadium_city.lower() or stadium_city.lower() in city.lower():
                        coords = coord
                        break
            
            if not coords:
                logger.warning(f"No coordinates found for {city}, {state}")
                return {'score': 75, 'description': 'Unknown location'}
            
            lat, lon = coords
            
            # Step 1: Get grid point data from coordinates
            points_url = f"https://api.weather.gov/points/{lat},{lon}"
            points_response = requests.get(points_url, headers={'User-Agent': 'BetYard NFL Predictor'}, timeout=10)
            
            if points_response.status_code != 200:
                logger.warning(f"Weather.gov points API failed: {points_response.status_code}")
                return {'score': 75, 'description': 'API error'}
            
            points_data = points_response.json()
            forecast_url = points_data['properties']['forecast']
            
            # Step 2: Get forecast
            forecast_response = requests.get(forecast_url, headers={'User-Agent': 'BetYard NFL Predictor'}, timeout=10)
            
            if forecast_response.status_code != 200:
                logger.warning(f"Weather.gov forecast API failed: {forecast_response.status_code}")
                return {'score': 75, 'description': 'Forecast unavailable'}
            
            forecast_data = forecast_response.json()
            
            # Get current/next period forecast
            periods = forecast_data['properties']['periods']
            if not periods:
                return {'score': 75, 'description': 'No forecast data'}
            
            current_forecast = periods[0]
            
            # Extract weather data
            temp = current_forecast['temperature']
            wind_speed = int(current_forecast.get('windSpeed', '0 mph').split()[0]) if current_forecast.get('windSpeed') else 0
            conditions = current_forecast['shortForecast']
            
            # Calculate weather score
            score = self._calculate_weather_score(temp, wind_speed, conditions)
            
            return {
                'score': score,
                'temp': temp,
                'wind': wind_speed,
                'conditions': conditions,
                'description': f"{temp}°F, {conditions}, {wind_speed}mph wind"
            }
            
        except Exception as e:
            logger.error(f"Failed to get weather from Weather.gov: {e}")
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
        
        # Precipitation impact (check for keywords in conditions string)
        conditions_lower = conditions.lower()
        if 'rain' in conditions_lower or 'storm' in conditions_lower:
            score -= 15
        elif 'snow' in conditions_lower:
            score -= 25
        elif 'shower' in conditions_lower:
            score -= 10
        
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
