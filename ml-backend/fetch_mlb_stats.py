"""
Fetch MLB player statistics from ESPN API - 2024 Season (2025 season hasn't started yet)
Fetches hitters and pitchers from all 30 MLB teams
Focus: Hits, Home Runs, RBIs for hitters; Strikeouts, Hits Allowed, Earned Runs for pitchers
"""

import requests
import json
import time

# ESPN API endpoints
ESPN_BASE = "https://site.web.api.espn.com/apis"
PLAYER_GAMELOG_URL = f"{ESPN_BASE}/common/v3/sports/baseball/mlb/athletes/{{player_id}}/gamelog"
TEAM_ROSTER_URL = f"{ESPN_BASE}/site/v2/sports/baseball/mlb/teams/{{team_id}}/roster"

# MLB Team IDs for ESPN API
MLB_TEAMS = {
    'ARI': 29, 'ATL': 15, 'BAL': 1, 'BOS': 2, 'CHC': 16, 'CWS': 4,
    'CIN': 17, 'CLE': 5, 'COL': 27, 'DET': 6, 'HOU': 18, 'KC': 7,
    'LAA': 3, 'LAD': 19, 'MIA': 28, 'MIL': 8, 'MIN': 9, 'NYM': 21,
    'NYY': 10, 'OAK': 11, 'PHI': 22, 'PIT': 23, 'SD': 25, 'SF': 26,
    'SEA': 12, 'STL': 24, 'TB': 30, 'TEX': 13, 'TOR': 14, 'WSH': 20
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
    team_id = MLB_TEAMS.get(team_code)
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
                position_abbr = position_data.get('abbreviation', 'P')
                
                # Include all positions
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
    Fetch player game log for 2024 season
    
    For Hitters: Hits, Home Runs, RBIs, Strikeouts, Walks
    For Pitchers: Strikeouts, Hits Allowed, Earned Runs, Innings Pitched, Walks
    """
    url = PLAYER_GAMELOG_URL.format(player_id=player_id)
    data = fetch_with_retry(url, retries=2, delay=0.5)
    
    if not data:
        return []
    
    try:
        stat_names = data.get('names', [])
        if not stat_names:
            return []
        
        is_pitcher = position in ['P', 'SP', 'RP']
        
        # Find indices for key stats
        hits_idx = None
        hr_idx = None
        rbi_idx = None
        k_idx = None
        bb_idx = None
        er_idx = None
        ip_idx = None
        
        for i, name in enumerate(stat_names):
            name_lower = name.lower()
            if not is_pitcher:
                # Hitter stats
                if name_lower in ['h', 'hits']:
                    hits_idx = i
                elif name_lower in ['hr', 'home runs']:
                    hr_idx = i
                elif name_lower in ['rbi', 'rbis']:
                    rbi_idx = i
                elif name_lower in ['k', 'so', 'strikeouts']:
                    k_idx = i
                elif name_lower in ['bb', 'walks']:
                    bb_idx = i
            else:
                # Pitcher stats
                if name_lower in ['k', 'so', 'strikeouts']:
                    k_idx = i
                elif name_lower in ['h', 'hits']:
                    hits_idx = i
                elif name_lower in ['er', 'earned runs']:
                    er_idx = i
                elif name_lower in ['ip', 'innings pitched']:
                    ip_idx = i
                elif name_lower in ['bb', 'walks']:
                    bb_idx = i
        
        gamelog = []
        
        # Events are stored in seasonTypes -> categories -> events (same as NBA/NHL)
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
                        
                        if is_pitcher:
                            game_stats = {
                                'game_id': event_id,
                                'strikeouts': float(stat_values[k_idx]) if k_idx is not None and k_idx < len(stat_values) else 0.0,
                                'hits_allowed': float(stat_values[hits_idx]) if hits_idx is not None and hits_idx < len(stat_values) else 0.0,
                                'earned_runs': float(stat_values[er_idx]) if er_idx is not None and er_idx < len(stat_values) else 0.0,
                                'innings_pitched': float(stat_values[ip_idx]) if ip_idx is not None and ip_idx < len(stat_values) else 0.0
                            }
                        else:
                            game_stats = {
                                'game_id': event_id,
                                'hits': float(stat_values[hits_idx]) if hits_idx is not None and hits_idx < len(stat_values) else 0.0,
                                'home_runs': float(stat_values[hr_idx]) if hr_idx is not None and hr_idx < len(stat_values) else 0.0,
                                'rbis': float(stat_values[rbi_idx]) if rbi_idx is not None and rbi_idx < len(stat_values) else 0.0,
                                'strikeouts': float(stat_values[k_idx]) if k_idx is not None and k_idx < len(stat_values) else 0.0
                            }
                        
                        gamelog.append(game_stats)
                        
                    except Exception as e:
                        print(f"Error parsing game: {e}")
                        continue
        
        return gamelog
        
    except Exception as e:
        print(f"Error processing gamelog: {e}")
        return []

def calculate_hit_probability(gamelog, threshold=1.5):
    """Calculate probability of getting 2+ hits"""
    if not gamelog:
        return {
            'games_played': 0,
            'games_with_2plus_hits': 0,
            'probability_2plus_hits': 0.0,
            'total_hits': 0.0,
            'avg_hits_per_game': 0.0
        }
    
    games_played = len(gamelog)
    hits = [g.get('hits', 0.0) for g in gamelog]
    games_with_2plus = sum(1 for h in hits if h >= 2)
    total_hits = sum(hits)
    
    return {
        'games_played': games_played,
        'games_with_2plus_hits': games_with_2plus,
        'probability_2plus_hits': round((games_with_2plus / games_played) * 100, 1) if games_played > 0 else 0.0,
        'total_hits': total_hits,
        'avg_hits_per_game': round(total_hits / games_played, 2) if games_played > 0 else 0.0
    }

def fetch_all_mlb_players():
    """Fetch all MLB players from all teams"""
    all_players = []
    
    print("Fetching MLB rosters from all 30 teams...")
    print("NOTE: Using 2024 season data (2025 season hasn't started yet)")
    
    for team_code, team_id in MLB_TEAMS.items():
        print(f"Fetching {team_code}...")
        roster = get_team_roster(team_code)
        
        for player in roster:
            player_id = player['id']
            position = player['position']
            print(f"  {player['name']} ({position})...", end=' ')
            
            gamelog = get_player_gamelog(player_id, position)
            
            if gamelog:
                games_played = len(gamelog)
                is_pitcher = position in ['P', 'SP', 'RP']
                
                if is_pitcher:
                    # Pitcher stats
                    strikeouts = [g.get('strikeouts', 0) for g in gamelog]
                    hits_allowed = [g.get('hits_allowed', 0) for g in gamelog]
                    earned_runs = [g.get('earned_runs', 0) for g in gamelog]
                    
                    player_data = {
                        'id': player_id,
                        'player_name': player['name'],
                        'team': team_code,
                        'position': position,
                        'gamelog': gamelog,
                        'season_stats': {
                            'games_played': games_played,
                            'total_strikeouts': sum(strikeouts),
                            'avg_strikeouts': round(sum(strikeouts) / games_played, 1) if games_played > 0 else 0.0,
                            'avg_hits_allowed': round(sum(hits_allowed) / games_played, 1) if games_played > 0 else 0.0,
                            'avg_earned_runs': round(sum(earned_runs) / games_played, 2) if games_played > 0 else 0.0
                        }
                    }
                else:
                    # Hitter stats
                    hits = [g.get('hits', 0) for g in gamelog]
                    hrs = [g.get('home_runs', 0) for g in gamelog]
                    rbis = [g.get('rbis', 0) for g in gamelog]
                    
                    hit_prob = calculate_hit_probability(gamelog)
                    
                    player_data = {
                        'id': player_id,
                        'player_name': player['name'],
                        'team': team_code,
                        'position': position,
                        'gamelog': gamelog,
                        'hit_stats': hit_prob,
                        'season_stats': {
                            'games_played': games_played,
                            'total_hits': sum(hits),
                            'total_home_runs': sum(hrs),
                            'total_rbis': sum(rbis),
                            'avg_hits': round(sum(hits) / games_played, 2) if games_played > 0 else 0.0,
                            'probability_2plus_hits': hit_prob['probability_2plus_hits']
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
    print("MLB PLAYER STATS FETCHER - 2024 SEASON")
    print("=" * 60)
    
    players = fetch_all_mlb_players()
    
    # Save to JSON
    output_file = 'mlb_player_data.json'
    with open(output_file, 'w') as f:
        json.dump(players, f, indent=2)
    
    print(f"\n{'=' * 60}")
    print(f"COMPLETE: {len(players)} players saved to {output_file}")
    print(f"{'=' * 60}")
    
    # Print summary
    pitchers = [p for p in players if p['position'] in ['P', 'SP', 'RP']]
    hitters = [p for p in players if p['position'] not in ['P', 'SP', 'RP']]
    
    print(f"\nSummary:")
    print(f"  Total Players: {len(players)}")
    print(f"  Pitchers: {len(pitchers)}")
    print(f"  Hitters: {len(hitters)}")
