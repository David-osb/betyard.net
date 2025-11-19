"""
Fetch NBA player statistics from ESPN API - 2024-25 Season
Fetches all players from all 30 NBA teams
Focus: Points, Rebounds, Assists, 3PM, Steals, Blocks for prop betting
"""

import requests
import json
import time

# ESPN API endpoints
ESPN_BASE = "https://site.web.api.espn.com/apis"
PLAYER_GAMELOG_URL = f"{ESPN_BASE}/common/v3/sports/basketball/nba/athletes/{{player_id}}/gamelog"
TEAM_ROSTER_URL = f"{ESPN_BASE}/site/v2/sports/basketball/nba/teams/{{team_id}}/roster"

# NBA Team IDs for ESPN API
NBA_TEAMS = {
    'ATL': 1, 'BOS': 2, 'BKN': 17, 'CHA': 30, 'CHI': 4, 'CLE': 5,
    'DAL': 6, 'DEN': 7, 'DET': 8, 'GSW': 9, 'HOU': 10, 'IND': 11,
    'LAC': 12, 'LAL': 13, 'MEM': 29, 'MIA': 14, 'MIL': 15, 'MIN': 16,
    'NOP': 3, 'NYK': 18, 'OKC': 25, 'ORL': 19, 'PHI': 20, 'PHX': 21,
    'POR': 22, 'SAC': 23, 'SAS': 24, 'TOR': 28, 'UTA': 26, 'WAS': 27
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
    team_id = NBA_TEAMS.get(team_code)
    if not team_id:
        return []
    
    url = TEAM_ROSTER_URL.format(team_id=team_id)
    data = fetch_with_retry(url)
    
    if not data or 'athletes' not in data:
        return []
    
    players = []
    for athlete in data.get('athletes', []):
        try:
            position_data = athlete.get('position', {})
            position_abbr = position_data.get('abbreviation', 'G')
            
            # Include all positions (PG, SG, SF, PF, C, G, F)
            players.append({
                'id': athlete.get('id'),
                'name': athlete.get('displayName', 'Unknown'),
                'team': team_code,
                'position': position_abbr
            })
        except Exception as e:
            print(f"Error parsing player: {e}")
    
    return players

def get_player_gamelog(player_id):
    """
    Fetch player game log for 2024-25 season
    
    Returns: gamelog_data with stats per game
    Stats tracked: Points, Rebounds, Assists, 3PM, Steals, Blocks, Minutes
    """
    url = PLAYER_GAMELOG_URL.format(player_id=player_id)
    data = fetch_with_retry(url, retries=2, delay=0.5)
    
    if not data:
        return []
    
    try:
        stat_names = data.get('names', [])
        if not stat_names:
            return []
        
        # Find indices for key stats (check names array)
        points_idx = None
        rebounds_idx = None
        assists_idx = None
        threes_idx = None
        steals_idx = None
        blocks_idx = None
        minutes_idx = None
        
        for i, name in enumerate(stat_names):
            name_lower = name.lower()
            if 'points' in name_lower:
                points_idx = i
            elif 'rebounds' in name_lower and 'total' in name_lower:
                rebounds_idx = i
            elif 'assists' in name_lower:
                assists_idx = i
            elif 'threepointfieldgoalsmade' in name_lower.replace('-', ''):
                threes_idx = i
            elif 'steals' in name_lower:
                steals_idx = i
            elif 'blocks' in name_lower:
                blocks_idx = i
            elif 'minutes' in name_lower:
                minutes_idx = i
        
        gamelog = []
        
        # Events are stored in seasonTypes -> categories -> events
        season_types = data.get('seasonTypes', [])
        for season in season_types:
            if 'Regular Season' not in season.get('displayName', ''):
                continue
                
            categories = season.get('categories', [])
            for category in categories:
                events = category.get('events', [])
                
                for event in events:
                    try:
                        event_id = event.get('eventId')
                        stat_values = event.get('stats', [])
                        
                        if not stat_values:
                            continue
                        
                        # Parse 3PM from "made-attempted" format
                        threes_made = 0.0
                        if threes_idx is not None and threes_idx < len(stat_values):
                            threes_str = stat_values[threes_idx]
                            if isinstance(threes_str, str) and '-' in threes_str:
                                threes_made = float(threes_str.split('-')[0])
                            else:
                                threes_made = float(threes_str) if threes_str else 0.0
                        
                        game_stats = {
                            'game_id': event_id,
                            'points': float(stat_values[points_idx]) if points_idx is not None and points_idx < len(stat_values) else 0.0,
                            'rebounds': float(stat_values[rebounds_idx]) if rebounds_idx is not None and rebounds_idx < len(stat_values) else 0.0,
                            'assists': float(stat_values[assists_idx]) if assists_idx is not None and assists_idx < len(stat_values) else 0.0,
                            'threes_made': threes_made,
                            'steals': float(stat_values[steals_idx]) if steals_idx is not None and steals_idx < len(stat_values) else 0.0,
                            'blocks': float(stat_values[blocks_idx]) if blocks_idx is not None and blocks_idx < len(stat_values) else 0.0,
                            'minutes': float(stat_values[minutes_idx]) if minutes_idx is not None and minutes_idx < len(stat_values) else 0.0
                        }
                        
                        gamelog.append(game_stats)
                        
                    except Exception as e:
                        print(f"Error parsing game: {e}")
                        continue
        
        return gamelog
        
    except Exception as e:
        print(f"Error processing gamelog: {e}")
        return []

def calculate_prop_probabilities(gamelog, prop_type, threshold):
    """
    Calculate probability of player hitting a prop threshold
    
    prop_type: 'points', 'rebounds', 'assists', 'threes_made', 'steals', 'blocks'
    threshold: The O/U line (e.g., 25.5 points)
    
    Returns: dict with probability stats
    """
    if not gamelog:
        return {
            'games_played': 0,
            'times_over': 0,
            'over_probability': 0.0,
            'average': 0.0
        }
    
    games_played = len(gamelog)
    values = [game.get(prop_type, 0.0) for game in gamelog]
    times_over = sum(1 for val in values if val > threshold)
    
    return {
        'games_played': games_played,
        'times_over': times_over,
        'over_probability': round((times_over / games_played) * 100, 1) if games_played > 0 else 0.0,
        'average': round(sum(values) / games_played, 1) if games_played > 0 else 0.0,
        'last_5_avg': round(sum(values[:5]) / min(5, len(values)), 1) if values else 0.0
    }

def fetch_all_nba_players():
    """Fetch all NBA players from all teams"""
    all_players = []
    
    print("Fetching NBA rosters from all 30 teams...")
    
    for team_code, team_id in NBA_TEAMS.items():
        print(f"Fetching {team_code}...")
        roster = get_team_roster(team_code)
        
        for player in roster:
            player_id = player['id']
            print(f"  {player['name']} ({player['position']})...", end=' ')
            
            gamelog = get_player_gamelog(player_id)
            
            if gamelog:
                # Calculate season averages
                games_played = len(gamelog)
                points = [g['points'] for g in gamelog]
                rebounds = [g['rebounds'] for g in gamelog]
                assists = [g['assists'] for g in gamelog]
                threes = [g['threes_made'] for g in gamelog]
                
                player_data = {
                    'id': player_id,
                    'player_name': player['name'],
                    'team': team_code,
                    'position': player['position'],
                    'gamelog': gamelog,
                    'season_stats': {
                        'games_played': games_played,
                        'ppg': round(sum(points) / games_played, 1) if games_played > 0 else 0.0,
                        'rpg': round(sum(rebounds) / games_played, 1) if games_played > 0 else 0.0,
                        'apg': round(sum(assists) / games_played, 1) if games_played > 0 else 0.0,
                        'three_pm': round(sum(threes) / games_played, 1) if games_played > 0 else 0.0
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
    print("NBA PLAYER STATS FETCHER - 2024-25 SEASON")
    print("=" * 60)
    
    players = fetch_all_nba_players()
    
    # Save to JSON
    output_file = 'nba_player_data.json'
    with open(output_file, 'w') as f:
        json.dump(players, f, indent=2)
    
    print(f"\n{'=' * 60}")
    print(f"COMPLETE: {len(players)} players saved to {output_file}")
    print(f"{'=' * 60}")
    
    # Print summary stats
    total_games = sum(p['season_stats']['games_played'] for p in players)
    avg_ppg = sum(p['season_stats']['ppg'] for p in players) / len(players) if players else 0
    
    print(f"\nSummary:")
    print(f"  Total Players: {len(players)}")
    print(f"  Total Games Logged: {total_games}")
    print(f"  Average PPG: {avg_ppg:.1f}")
