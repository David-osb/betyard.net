"""
Fetch NHL player statistics from ESPN API - 2024-25 Season
Fetches all players from all 32 NHL teams
Focus: Goals, Assists, Shots, Points, +/- for skaters; Saves, GAA for goalies
"""

import requests
import json
import time

# ESPN API endpoints
ESPN_BASE = "https://site.web.api.espn.com/apis"
PLAYER_GAMELOG_URL = f"{ESPN_BASE}/common/v3/sports/hockey/nhl/athletes/{{player_id}}/gamelog"
TEAM_ROSTER_URL = f"{ESPN_BASE}/site/v2/sports/hockey/nhl/teams/{{team_id}}/roster"

# NHL Team IDs for ESPN API
NHL_TEAMS = {
    'ANA': 25, 'BOS': 6, 'BUF': 7, 'CGY': 20, 'CAR': 12, 'CHI': 11,
    'COL': 21, 'CBJ': 29, 'DAL': 10, 'DET': 17, 'EDM': 22, 'FLA': 13,
    'LAK': 26, 'MIN': 30, 'MTL': 8, 'NSH': 18, 'NJD': 1, 'NYI': 2,
    'NYR': 3, 'OTT': 9, 'PHI': 4, 'PIT': 5, 'SEA': 55, 'SJS': 28,
    'STL': 19, 'TBL': 14, 'TOR': 10, 'UTA': 54, 'VAN': 23, 'VGK': 37,
    'WSH': 15, 'WPG': 52
}

def fetch_with_retry(url, retries=2, delay=1):
    """Fetch URL with retry logic"""
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=15)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
        
        if attempt < retries - 1:
            time.sleep(delay)
    
    return None

def get_team_roster(team_code):
    """Fetch all players from team roster"""
    team_id = NHL_TEAMS.get(team_code)
    if not team_id:
        return []
    
    url = TEAM_ROSTER_URL.format(team_id=team_id)
    data = fetch_with_retry(url)
    
    if not data or 'athletes' not in data:
        return []
    
    players = []
    for group in data.get('athletes', []):
        for athlete in group.get('items', []):
            try:
                position_data = athlete.get('position', {})
                position_abbr = position_data.get('abbreviation', 'F')
                
                # Include all positions (C, LW, RW, D, G)
                players.append({
                    'id': athlete.get('id'),
                    'name': athlete.get('displayName', 'Unknown'),
                    'team': team_code,
                    'position': position_abbr
                })
            except Exception as e:
                print(f"Error parsing player: {e}")
    
    return players

def get_player_gamelog(player_id, position):
    """
    Fetch player game log for 2024-25 season
    
    For Skaters: Goals, Assists, Points, Shots, +/-, Hits, Blocks
    For Goalies: Saves, Goals Against, Save %, Shots Against
    """
    url = PLAYER_GAMELOG_URL.format(player_id=player_id)
    data = fetch_with_retry(url, retries=2, delay=0.5)
    
    if not data:
        return []
    
    try:
        stat_names = data.get('names', [])
        if not stat_names:
            return []
        
        is_goalie = position == 'G'
        
        # Find indices for key stats
        goals_idx = None
        assists_idx = None
        points_idx = None
        shots_idx = None
        saves_idx = None
        ga_idx = None
        plusminus_idx = None
        
        for i, name in enumerate(stat_names):
            name_lower = name.lower()
            if not is_goalie:
                if name_lower in ['g', 'goals']:
                    goals_idx = i
                elif name_lower in ['a', 'assists']:
                    assists_idx = i
                elif name_lower in ['pts', 'points']:
                    points_idx = i
                elif name_lower in ['sog', 'shots']:
                    shots_idx = i
                elif '+/-' in name_lower or 'plus' in name_lower:
                    plusminus_idx = i
            else:
                if 'saves' in name_lower or 'sv' in name_lower:
                    saves_idx = i
                elif 'ga' in name_lower or 'goals against' in name_lower:
                    ga_idx = i
                elif 'sa' in name_lower or 'shots against' in name_lower:
                    shots_idx = i
        
        gamelog = []
        
        # Events are stored in seasonTypes -> categories -> events (same as NBA)
        season_types = data.get('seasonTypes', [])
        for season in season_types:
            if 'Regular Season' not in season.get('displayName', ''):
                continue
                
            categories = season.get('categories', [])
            for category in categories:
                events_list = category.get('events', [])
                
                for event in events_list:
                    try:
                        event_id = event.get('eventId')
                        stat_values = event.get('stats', [])
                        
                        if not stat_values:
                            continue
                        
                        if is_goalie:
                            game_stats = {
                                'game_id': event_id,
                                'saves': float(stat_values[saves_idx]) if saves_idx is not None and saves_idx < len(stat_values) else 0.0,
                                'goals_against': float(stat_values[ga_idx]) if ga_idx is not None and ga_idx < len(stat_values) else 0.0,
                                'shots_against': float(stat_values[shots_idx]) if shots_idx is not None and shots_idx < len(stat_values) else 0.0
                            }
                        else:
                            game_stats = {
                                'game_id': event_id,
                                'goals': float(stat_values[goals_idx]) if goals_idx is not None and goals_idx < len(stat_values) else 0.0,
                                'assists': float(stat_values[assists_idx]) if assists_idx is not None and assists_idx < len(stat_values) else 0.0,
                                'points': float(stat_values[points_idx]) if points_idx is not None and points_idx < len(stat_values) else 0.0,
                                'shots': float(stat_values[shots_idx]) if shots_idx is not None and shots_idx < len(stat_values) else 0.0
                            }
                        
                        gamelog.append(game_stats)
                        
                    except Exception as e:
                        print(f"Error parsing game: {e}")
                        continue
        
        return gamelog
        
    except Exception as e:
        print(f"Error processing gamelog: {e}")
        return []

def calculate_goal_probability(gamelog):
    """Calculate probability of scoring a goal (anytime goal scorer)"""
    if not gamelog:
        return {
            'games_played': 0,
            'games_with_goal': 0,
            'goal_probability': 0.0,
            'total_goals': 0.0,
            'avg_goals_per_game': 0.0
        }
    
    games_played = len(gamelog)
    goals = [g.get('goals', 0.0) for g in gamelog]
    games_with_goal = sum(1 for g in goals if g > 0)
    total_goals = sum(goals)
    
    return {
        'games_played': games_played,
        'games_with_goal': games_with_goal,
        'goal_probability': round((games_with_goal / games_played) * 100, 1) if games_played > 0 else 0.0,
        'total_goals': total_goals,
        'avg_goals_per_game': round(total_goals / games_played, 2) if games_played > 0 else 0.0
    }

def fetch_all_nhl_players():
    """Fetch all NHL players from all teams"""
    all_players = []
    
    print("Fetching NHL rosters from all 32 teams...")
    
    for team_code, team_id in NHL_TEAMS.items():
        print(f"Fetching {team_code}...")
        roster = get_team_roster(team_code)
        
        for player in roster:
            player_id = player['id']
            position = player['position']
            print(f"  {player['name']} ({position})...", end=' ')
            
            gamelog = get_player_gamelog(player_id, position)
            
            if gamelog:
                games_played = len(gamelog)
                
                if position == 'G':
                    # Goalie stats
                    saves = [g.get('saves', 0) for g in gamelog]
                    ga = [g.get('goals_against', 0) for g in gamelog]
                    
                    player_data = {
                        'id': player_id,
                        'player_name': player['name'],
                        'team': team_code,
                        'position': position,
                        'gamelog': gamelog,
                        'season_stats': {
                            'games_played': games_played,
                            'avg_saves': round(sum(saves) / games_played, 1) if games_played > 0 else 0.0,
                            'avg_goals_against': round(sum(ga) / games_played, 2) if games_played > 0 else 0.0
                        }
                    }
                else:
                    # Skater stats
                    goals = [g.get('goals', 0) for g in gamelog]
                    assists = [g.get('assists', 0) for g in gamelog]
                    shots = [g.get('shots', 0) for g in gamelog]
                    
                    goal_prob = calculate_goal_probability(gamelog)
                    
                    player_data = {
                        'id': player_id,
                        'player_name': player['name'],
                        'team': team_code,
                        'position': position,
                        'gamelog': gamelog,
                        'goal_stats': goal_prob,
                        'season_stats': {
                            'games_played': games_played,
                            'goals': sum(goals),
                            'assists': sum(assists),
                            'avg_shots': round(sum(shots) / games_played, 1) if games_played > 0 else 0.0,
                            'goal_probability': goal_prob['goal_probability']
                        }
                    }
                
                all_players.append(player_data)
                print(f"✓ {games_played} games")
            else:
                print("No data")
            
            time.sleep(0.3)  # Rate limiting
    
    return all_players

if __name__ == '__main__':
    print("=" * 60)
    print("NHL PLAYER STATS FETCHER - 2024-25 SEASON")
    print("=" * 60)
    
    players = fetch_all_nhl_players()
    
    # Save to JSON
    output_file = 'nhl_player_data.json'
    with open(output_file, 'w') as f:
        json.dump(players, f, indent=2)
    
    print(f"\n{'=' * 60}")
    print(f"COMPLETE: {len(players)} players saved to {output_file}")
    print(f"{'=' * 60}")
    
    # Print summary
    skaters = [p for p in players if p['position'] != 'G']
    goalies = [p for p in players if p['position'] == 'G']
    
    print(f"\nSummary:")
    print(f"  Total Players: {len(players)}")
    print(f"  Skaters: {len(skaters)}")
    print(f"  Goalies: {len(goalies)}")
