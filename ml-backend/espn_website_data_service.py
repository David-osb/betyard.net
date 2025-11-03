"""
ESPN Website Data Service
Provides ESPN API-powered data for website content updates
Replaces Tank01 dependencies with ESPN data sources
"""

import requests
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import time

class ESPNWebsiteDataService:
    """
    ESPN API service for website content data
    Powers news, matchups, player info, and more
    """
    
    def __init__(self):
        self.base_url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"
        self.session = requests.Session()
        self.cache = {}
        self.cache_duration = 300  # 5 minutes
    
    def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """Make cached API request to ESPN"""
        cache_key = f"{endpoint}_{str(params)}"
        
        if cache_key in self.cache:
            cached_time, data = self.cache[cache_key]
            if time.time() - cached_time < self.cache_duration:
                return data
        
        try:
            url = f"{self.base_url}/{endpoint.lstrip('/')}"
            response = self.session.get(url, params=params or {}, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            self.cache[cache_key] = (time.time(), data)
            return data
            
        except Exception as e:
            print(f"ESPN API Error for {endpoint}: {e}")
            return {}
    
    def get_latest_news(self, limit: int = 10) -> List[Dict]:
        """Get latest NFL news"""
        try:
            data = self._make_request("news", {"limit": limit})
            
            news_items = []
            for article in data.get("articles", [])[:limit]:
                news_items.append({
                    "id": article.get("id"),
                    "headline": article.get("headline", ""),
                    "description": article.get("description", ""),
                    "published": article.get("published"),
                    "images": article.get("images", []),
                    "links": article.get("links", {}),
                    "categories": [cat.get("description", "") for cat in article.get("categories", [])]
                })
            
            return news_items
            
        except Exception as e:
            print(f"Error getting news: {e}")
            return []
    
    def get_team_matchups(self, week: Optional[int] = None) -> List[Dict]:
        """Get current/specified week matchups"""
        try:
            params = {}
            if week:
                params["week"] = week
                
            data = self._make_request("scoreboard", params)
            
            matchups = []
            for game in data.get("events", []):
                competitions = game.get("competitions", [])
                if not competitions:
                    continue
                    
                competition = competitions[0]
                competitors = competition.get("competitors", [])
                
                if len(competitors) >= 2:
                    home_team = next((c for c in competitors if c.get("homeAway") == "home"), {})
                    away_team = next((c for c in competitors if c.get("homeAway") == "away"), {})
                    
                    matchups.append({
                        "game_id": game.get("id"),
                        "date": game.get("date"),
                        "status": competition.get("status", {}).get("type", {}).get("description", ""),
                        "week": game.get("week", {}).get("number"),
                        "home_team": {
                            "id": home_team.get("team", {}).get("id"),
                            "name": home_team.get("team", {}).get("displayName", ""),
                            "abbreviation": home_team.get("team", {}).get("abbreviation", ""),
                            "logo": home_team.get("team", {}).get("logo", ""),
                            "score": home_team.get("score", "0")
                        },
                        "away_team": {
                            "id": away_team.get("team", {}).get("id"),
                            "name": away_team.get("team", {}).get("displayName", ""),
                            "abbreviation": away_team.get("team", {}).get("abbreviation", ""),
                            "logo": away_team.get("team", {}).get("logo", ""),
                            "score": away_team.get("score", "0")
                        }
                    })
            
            return matchups
            
        except Exception as e:
            print(f"Error getting matchups: {e}")
            return []
    
    def search_players(self, query: str, limit: int = 10) -> List[Dict]:
        """Search for players by name"""
        try:
            # ESPN doesn't have a direct player search, so we'll get teams and extract rosters
            teams_data = self._make_request("teams")
            
            players = []
            query_lower = query.lower()
            
            for team in teams_data.get("sports", [{}])[0].get("leagues", [{}])[0].get("teams", []):
                team_id = team.get("team", {}).get("id")
                if not team_id:
                    continue
                
                # Get team roster
                roster_data = self._make_request(f"teams/{team_id}/roster")
                
                for athlete in roster_data.get("athletes", []):
                    name = athlete.get("displayName", "").lower()
                    if query_lower in name:
                        players.append({
                            "id": athlete.get("id"),
                            "name": athlete.get("displayName", ""),
                            "position": athlete.get("position", {}).get("abbreviation", ""),
                            "team": team.get("team", {}).get("displayName", ""),
                            "team_id": team_id,
                            "jersey": athlete.get("jersey", ""),
                            "headshot": athlete.get("headshot", {}).get("href", "")
                        })
                
                if len(players) >= limit:
                    break
            
            return players[:limit]
            
        except Exception as e:
            print(f"Error searching players: {e}")
            return []
    
    def get_team_info(self, team_id: str) -> Dict:
        """Get detailed team information"""
        try:
            data = self._make_request(f"teams/{team_id}")
            
            team = data.get("team", {})
            return {
                "id": team.get("id"),
                "name": team.get("displayName", ""),
                "abbreviation": team.get("abbreviation", ""),
                "location": team.get("location", ""),
                "nickname": team.get("name", ""),
                "color": team.get("color", ""),
                "alternateColor": team.get("alternateColor", ""),
                "logo": team.get("logos", [{}])[0].get("href", "") if team.get("logos") else "",
                "record": team.get("record", {}).get("items", [{}])[0].get("summary", "") if team.get("record") else "",
                "next_event": team.get("nextEvent", [{}])[0] if team.get("nextEvent") else {}
            }
            
        except Exception as e:
            print(f"Error getting team info: {e}")
            return {}
    
    def get_current_injuries(self, team_id: Optional[str] = None) -> List[Dict]:
        """Get injury reports"""
        try:
            if team_id:
                # Team-specific injuries
                data = self._make_request(f"teams/{team_id}/injuries")
            else:
                # League-wide injuries
                data = self._make_request("news", {"category": "injuries"})
            
            injuries = []
            for item in data.get("items", []):
                injuries.append({
                    "player": item.get("athlete", {}).get("displayName", ""),
                    "team": item.get("team", {}).get("displayName", ""),
                    "position": item.get("athlete", {}).get("position", {}).get("abbreviation", ""),
                    "status": item.get("status", ""),
                    "description": item.get("description", ""),
                    "date": item.get("date", "")
                })
            
            return injuries
            
        except Exception as e:
            print(f"Error getting injuries: {e}")
            return []
    
    def get_weekly_schedule(self, week: Optional[int] = None) -> Dict:
        """Get schedule for specific week"""
        try:
            params = {}
            if week:
                params["week"] = week
                
            data = self._make_request("scoreboard", params)
            
            return {
                "week": data.get("week", {}).get("number", 1),
                "season_type": data.get("season", {}).get("type", 2),
                "season_year": data.get("season", {}).get("year", 2024),
                "games": self.get_team_matchups(week)
            }
            
        except Exception as e:
            print(f"Error getting schedule: {e}")
            return {}
    
    def get_current_standings(self) -> List[Dict]:
        """Get current NFL standings"""
        try:
            data = self._make_request("standings")
            
            standings = []
            for conference in data.get("standings", []):
                for division in conference.get("entries", []):
                    team = division.get("team", {})
                    stats = division.get("stats", [])
                    
                    # Extract wins, losses, ties
                    wins = losses = ties = 0
                    for stat in stats:
                        if stat.get("name") == "wins":
                            wins = stat.get("value", 0)
                        elif stat.get("name") == "losses":
                            losses = stat.get("value", 0)
                        elif stat.get("name") == "ties":
                            ties = stat.get("value", 0)
                    
                    standings.append({
                        "team_id": team.get("id"),
                        "team_name": team.get("displayName", ""),
                        "abbreviation": team.get("abbreviation", ""),
                        "logo": team.get("logos", [{}])[0].get("href", "") if team.get("logos") else "",
                        "wins": wins,
                        "losses": losses,
                        "ties": ties,
                        "winning_percentage": division.get("stats", [{}])[0].get("value", 0),
                        "conference": conference.get("name", ""),
                        "division": division.get("note", "")
                    })
            
            return standings
            
        except Exception as e:
            print(f"Error getting standings: {e}")
            return []
    
    def get_trending_players(self, limit: int = 10) -> List[Dict]:
        """Get trending/popular players"""
        try:
            # Get recent news and extract player mentions
            news_data = self._make_request("news", {"limit": 50})
            
            player_mentions = {}
            trending = []
            
            for article in news_data.get("articles", []):
                # Extract athlete references from categories or content
                for category in article.get("categories", []):
                    if category.get("type") == "athlete":
                        athlete_id = category.get("athleteId")
                        if athlete_id:
                            player_mentions[athlete_id] = player_mentions.get(athlete_id, 0) + 1
            
            # Get top mentioned players
            sorted_players = sorted(player_mentions.items(), key=lambda x: x[1], reverse=True)
            
            for player_id, mentions in sorted_players[:limit]:
                # Get player details
                try:
                    player_data = self._make_request(f"athletes/{player_id}")
                    athlete = player_data.get("athlete", {})
                    
                    trending.append({
                        "id": athlete.get("id"),
                        "name": athlete.get("displayName", ""),
                        "position": athlete.get("position", {}).get("abbreviation", ""),
                        "team": athlete.get("team", {}).get("displayName", ""),
                        "headshot": athlete.get("headshot", {}).get("href", ""),
                        "mentions": mentions
                    })
                except:
                    continue
            
            return trending
            
        except Exception as e:
            print(f"Error getting trending players: {e}")
            return []

    # Tank01 Compatibility Layer
    def get_tank01_team_stats(self, team_id: str) -> Dict:
        """Tank01 compatibility - team stats"""
        team_info = self.get_team_info(team_id)
        return {
            "teamID": team_id,
            "teamName": team_info.get("name", ""),
            "teamAbv": team_info.get("abbreviation", ""),
            "teamCity": team_info.get("location", ""),
            "teamLogo": team_info.get("logo", ""),
            "record": team_info.get("record", "")
        }
    
    def get_tank01_player_game_logs(self, player_id: str) -> List[Dict]:
        """Tank01 compatibility - player game logs"""
        # ESPN equivalent would require multiple API calls
        # For now, return structured placeholder
        return [
            {
                "gameID": f"game_{player_id}",
                "playerID": player_id,
                "gameDate": datetime.now().strftime("%Y-%m-%d"),
                "stats": {}
            }
        ]