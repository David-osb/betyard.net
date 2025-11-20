"""
Train NBA player prop predictions using REAL ESPN odds
Only includes players with actual sportsbook prop lines
"""

import json
import requests
from datetime import datetime

ESPN_BASE = "https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba"

def fetch_todays_nba_games():
    """Get today's and upcoming NBA games"""
    scoreboard_url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
    response = requests.get(scoreboard_url)
    data = response.json()
    
    events = data.get('events', [])
    game_ids = [event['id'] for event in events]
    
    print(f"📅 Found {len(game_ids)} NBA games")
    return game_ids

def fetch_game_props(event_id):
    """Fetch all prop bets for a specific game"""
    try:
        # Get ESPN BET props (provider 58)
        props_url = f"{ESPN_BASE}/events/{event_id}/competitions/{event_id}/odds/58/propBets?limit=1000"
        response = requests.get(props_url, timeout=10)
        
        if response.status_code != 200:
            return []
        
        data = response.json()
        return data.get('items', [])
    except Exception as e:
        print(f"  ❌ Error fetching props for game {event_id}: {e}")
        return []

def parse_espn_props(props_items):
    """Parse ESPN props into player data with real lines"""
    player_props = {}
    
    for prop in props_items:
        athlete_ref = prop.get('athlete', {}).get('$ref', '')
        if not athlete_ref:
            continue
            
        athlete_id = athlete_ref.split('/')[-1].split('?')[0]
        prop_type = prop.get('type', {}).get('name', '')
        line = prop.get('current', {}).get('target', {}).get('value')
        
        if not line:
            continue
        
        # Map ESPN prop names to our format
        prop_mapping = {
            'Total Points': 'points',
            'Total Rebounds': 'rebounds',
            'Total Assists': 'assists',
            'Total 3-Point Field Goals': 'threes_made'
        }
        
        if prop_type in prop_mapping:
            our_prop_name = prop_mapping[prop_type]
            
            if athlete_id not in player_props:
                player_props[athlete_id] = {}
            
            # Use first line encountered for each prop type
            if our_prop_name not in player_props[athlete_id]:
                player_props[athlete_id][our_prop_name] = float(line)
    
    return player_props

def fetch_player_data(athlete_id):
    """Fetch player name and team"""
    try:
        url = f"{ESPN_BASE}/seasons/2026/athletes/{athlete_id}"
        response = requests.get(url, timeout=5)
        data = response.json()
        
        name = data.get('displayName', 'Unknown')
        
        # Get team
        team_ref = data.get('team', {}).get('$ref', '')
        if team_ref:
            team_id = team_ref.split('/')[-1].split('?')[0]
            team_resp = requests.get(f"{ESPN_BASE}/seasons/2026/teams/{team_id}")
            team_data = team_resp.json()
            team = team_data.get('abbreviation', 'UNK')
        else:
            team = 'UNK'
        
        position = data.get('position', {}).get('abbreviation', 'G')
        
        return {
            'id': athlete_id,
            'name': name,
            'team': team,
            'position': position
        }
    except:
        return {
            'id': athlete_id,
            'name': f'Player_{athlete_id}',
            'team': 'UNK',
            'position': 'G'
        }

def calculate_prop_probability(gamelog, stat_key, line):
    """Calculate probability of hitting over on a prop line"""
    if not gamelog:
        return 50.0
    
    times_over = sum(1 for game in gamelog if game.get(stat_key, 0) > line)
    return round((times_over / len(gamelog)) * 100, 1)

def load_historical_data():
    """Load our historical game log data"""
    try:
        with open('nba_player_data.json', 'r') as f:
            return json.load(f)
    except:
        print("⚠️  No historical data found, probabilities will be estimates")
        return []

def generate_real_odds_predictions(all_espn_props, historical_data):
    """Generate predictions using real ESPN odds"""
    
    # Create lookup for historical data
    hist_lookup = {p['player_name']: p for p in historical_data}
    
    predictions = []
    
    for athlete_id, espn_lines in all_espn_props.items():
        # Fetch player info
        player_info = fetch_player_data(athlete_id)
        player_name = player_info['name']
        
        # Get historical data if available
        hist_data = hist_lookup.get(player_name)
        gamelog = hist_data.get('gamelog', []) if hist_data else []
        
        # Calculate season averages from historical data
        if hist_data:
            ppg = hist_data['season_stats']['ppg']
            rpg = hist_data['season_stats']['rpg']
            apg = hist_data['season_stats']['apg']
            three_pm = hist_data['season_stats']['three_pm']
            games_played = len(gamelog)
        else:
            # Estimate from ESPN lines (rough approximation)
            ppg = espn_lines.get('points', 0) * 1.5
            rpg = espn_lines.get('rebounds', 0) * 1.5
            apg = espn_lines.get('assists', 0) * 1.5
            three_pm = espn_lines.get('threes_made', 0) * 1.5
            games_played = 10
        
        # Build prediction with REAL odds
        player_prediction = {
            'id': athlete_id,
            'player_name': player_name,
            'team': player_info['team'],
            'position': player_info['position'],
            'games_played': games_played,
            'props': {}
        }
        
        # Add each prop with real ESPN line
        for prop_key, real_line in espn_lines.items():
            # Calculate probability using historical data
            over_prob = calculate_prop_probability(gamelog, prop_key, real_line)
            
            # Determine average
            avg_map = {
                'points': ppg,
                'rebounds': rpg,
                'assists': apg,
                'threes_made': three_pm
            }
            avg = avg_map.get(prop_key, real_line * 1.3)
            
            # Calculate last 5 average
            if gamelog and len(gamelog) >= 5:
                last_5_avg = round(sum(g.get(prop_key, 0) for g in gamelog[:5]) / 5, 1)
            else:
                last_5_avg = avg
            
            # Determine recommendation
            if over_prob > 60:
                recommendation = 'OVER'
            elif over_prob < 40:
                recommendation = 'UNDER'
            else:
                recommendation = 'NO BET'
            
            player_prediction['props'][prop_key] = {
                'line': real_line,
                'average': round(avg, 1),
                'over_probability': over_prob,
                'under_probability': round(100 - over_prob, 1),
                'last_5_avg': last_5_avg,
                'recommendation': recommendation,
                'source': 'ESPN BET'
            }
        
        predictions.append(player_prediction)
    
    return predictions

def main():
    print("\n" + "="*60)
    print("NBA REAL ODDS PREDICTION MODEL")
    print("Using ESPN BET Live Prop Lines")
    print("="*60 + "\n")
    
    # Load historical data
    print("📊 Loading historical game logs...")
    historical_data = load_historical_data()
    print(f"   Loaded data for {len(historical_data)} players\n")
    
    # Fetch today's games
    game_ids = fetch_todays_nba_games()
    
    if not game_ids:
        print("❌ No games found")
        return
    
    # Fetch props for all games
    all_espn_props = {}
    
    for game_id in game_ids[:15]:  # Limit to 15 games
        print(f"  Fetching props for game {game_id}...")
        props = fetch_game_props(game_id)
        
        if props:
            game_props = parse_espn_props(props)
            all_espn_props.update(game_props)
            print(f"    ✅ Found props for {len(game_props)} players")
        else:
            print(f"    ⚠️  No props available")
    
    if not all_espn_props:
        print("\n❌ No ESPN prop data available")
        return
    
    print(f"\n{'='*60}")
    print(f"Total players with ESPN BET props: {len(all_espn_props)}")
    print(f"{'='*60}\n")
    
    # Generate predictions
    print("🤖 Generating predictions with real odds...\n")
    predictions = generate_real_odds_predictions(all_espn_props, historical_data)
    
    # Save predictions
    output_file = 'nba_prop_predictions.json'
    with open(output_file, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    print(f"{'='*60}")
    print(f"✅ COMPLETE: {len(predictions)} predictions saved")
    print(f"📁 Output: {output_file}")
    print(f"{'='*60}\n")
    
    # Show sample
    if predictions:
        sample = predictions[0]
        print(f"Sample prediction for {sample['player_name']} ({sample['team']}):")
        for prop_key, prop_data in sample['props'].items():
            print(f"  {prop_key.upper()}: {prop_data['average']} avg, "
                  f"O/U {prop_data['line']} = {prop_data['over_probability']}% OVER "
                  f"[{prop_data['recommendation']}]")
        print()

if __name__ == '__main__':
    main()
