"""
Fetch MLS player statistics from ESPN API - 2024 Season
Fetches all players from all MLS teams
Focus: Goals, Assists, Shots on Target for field players; Saves for goalkeepers
"""

import requests
import json
import time

# ESPN API endpoints
ESPN_BASE = "https://site.web.api.espn.com/apis"
PLAYER_GAMELOG_URL = f"{ESPN_BASE}/common/v3/sports/soccer/usa.1/athletes/{{player_id}}/gamelog"
TEAM_ROSTER_URL = f"{ESPN_BASE}/site/v2/sports/soccer/usa.1/teams/{{team_id}}/roster"

# MLS Team IDs for ESPN API (usa.1 is MLS league code)
MLS_TEAMS = {
    'ATL': 1, 'ATX': 27, 'CLT': 28, 'CHI': 3, 'CIN': 24, 'COL': 4,
    'CLB': 5, 'DAL': 6, 'DC': 7, 'HOU': 9, 'LAG': 11, 'LAFC': 26,
    'MIA': 29, 'MIN': 22, 'MTL': 10, 'NSH': 25, 'NE': 12, 'NY': 13,
    'NYC': 23, 'ORL': 21, 'PHI': 14, 'POR': 15, 'RSL': 16, 'SJ': 17,
    'SEA': 18, 'SKC': 20, 'STL': 30, 'TOR': 19, 'VAN': 8
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
    team_id = MLS_TEAMS.get(team_code)
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
            position_abbr = position_data.get('abbreviation', 'M')
            
            # Include all positions (F, M, D, GK)
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
    Fetch player game log for 2024 MLS season
    
    For Field Players: Goals, Assists, Shots, Shots on Target
    For Goalkeepers: Saves, Goals Against, Clean Sheets
    """
    url = PLAYER_GAMELOG_URL.format(player_id=player_id)
    data = fetch_with_retry(url, retries=2, delay=0.5)
    
    if not data:
        return []
    
    try:
        stat_names = data.get('names', [])
        if not stat_names:
            return []
        
        is_goalkeeper = position == 'GK'
        
        # Find indices for key stats
        goals_idx = None
        assists_idx = None
        shots_idx = None
        sot_idx = None
        saves_idx = None
        ga_idx = None
        
        for i, name in enumerate(stat_names):
            name_lower = name.lower()
            if not is_goalkeeper:
                if name_lower in ['g', 'goals']:
                    goals_idx = i
                elif name_lower in ['a', 'assists']:
                    assists_idx = i
                elif 'shots' in name_lower and 'target' not in name_lower:
                    shots_idx = i
                elif 'shots on target' in name_lower or 'sot' in name_lower:
                    sot_idx = i
            else:
                if 'saves' in name_lower or 'sv' in name_lower:
                    saves_idx = i
                elif 'ga' in name_lower or 'goals against' in name_lower:
                    ga_idx = i
        
        gamelog = []
        events = data.get('events', [])
        
        for event in events:
            try:
                stats = event.get('statistics', {})
                splits = stats.get('splits', {})
                categories = splits.get('categories', [])
                
                if not categories:
                    continue
                
                stat_values = []
                for cat in categories:
                    stat_values.extend(cat.get('stats', []))
                
                if is_goalkeeper:
                    game_stats = {
                        'game_id': event.get('eventId'),
                        'opponent': event.get('opponent', {}).get('abbreviation', 'UNK'),
                        'date': event.get('gameDate', ''),
                        'saves': float(stat_values[saves_idx]) if saves_idx is not None and saves_idx < len(stat_values) else 0.0,
                        'goals_against': float(stat_values[ga_idx]) if ga_idx is not None and ga_idx < len(stat_values) else 0.0
                    }
                else:
                    game_stats = {
                        'game_id': event.get('eventId'),
                        'opponent': event.get('opponent', {}).get('abbreviation', 'UNK'),
                        'date': event.get('gameDate', ''),
                        'goals': float(stat_values[goals_idx]) if goals_idx is not None and goals_idx < len(stat_values) else 0.0,
                        'assists': float(stat_values[assists_idx]) if assists_idx is not None and assists_idx < len(stat_values) else 0.0,
                        'shots': float(stat_values[shots_idx]) if shots_idx is not None and shots_idx < len(stat_values) else 0.0,
                        'shots_on_target': float(stat_values[sot_idx]) if sot_idx is not None and sot_idx < len(stat_values) else 0.0
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

def fetch_all_mls_players():
    """Fetch all MLS players from all teams"""
    all_players = []
    
    print("Fetching MLS rosters from all 29 teams...")
    print("NOTE: Using 2024 season data")
    
    for team_code, team_id in MLS_TEAMS.items():
        print(f"Fetching {team_code}...")
        roster = get_team_roster(team_code)
        
        for player in roster:
            player_id = player['id']
            position = player['position']
            print(f"  {player['name']} ({position})...", end=' ')
            
            gamelog = get_player_gamelog(player_id, position)
            
            if gamelog:
                games_played = len(gamelog)
                is_goalkeeper = position == 'GK'
                
                if is_goalkeeper:
                    # Goalkeeper stats
                    saves = [g.get('saves', 0) for g in gamelog]
                    ga = [g.get('goals_against', 0) for g in gamelog]
                    clean_sheets = sum(1 for g in ga if g == 0)
                    
                    player_data = {
                        'id': player_id,
                        'player_name': player['name'],
                        'team': team_code,
                        'position': position,
                        'gamelog': gamelog,
                        'season_stats': {
                            'games_played': games_played,
                            'total_saves': sum(saves),
                            'avg_saves': round(sum(saves) / games_played, 1) if games_played > 0 else 0.0,
                            'clean_sheets': clean_sheets,
                            'clean_sheet_percentage': round((clean_sheets / games_played) * 100, 1) if games_played > 0 else 0.0
                        }
                    }
                else:
                    # Field player stats
                    goals = [g.get('goals', 0) for g in gamelog]
                    assists = [g.get('assists', 0) for g in gamelog]
                    shots = [g.get('shots', 0) for g in gamelog]
                    sot = [g.get('shots_on_target', 0) for g in gamelog]
                    
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
                            'total_goals': sum(goals),
                            'total_assists': sum(assists),
                            'avg_shots': round(sum(shots) / games_played, 1) if games_played > 0 else 0.0,
                            'avg_shots_on_target': round(sum(sot) / games_played, 1) if games_played > 0 else 0.0,
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
    print("MLS PLAYER STATS FETCHER - 2024 SEASON")
    print("=" * 60)
    
    players = fetch_all_mls_players()
    
    # Save to JSON
    output_file = 'mls_player_data.json'
    with open(output_file, 'w') as f:
        json.dump(players, f, indent=2)
    
    print(f"\n{'=' * 60}")
    print(f"COMPLETE: {len(players)} players saved to {output_file}")
    print(f"{'=' * 60}")
    
    # Print summary
    field_players = [p for p in players if p['position'] != 'GK']
    goalkeepers = [p for p in players if p['position'] == 'GK']
    
    print(f"\nSummary:")
    print(f"  Total Players: {len(players)}")
    print(f"  Field Players: {len(field_players)}")
    print(f"  Goalkeepers: {len(goalkeepers)}")
