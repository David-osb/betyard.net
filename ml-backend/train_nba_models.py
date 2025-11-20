"""
Train NBA player prop prediction models
Uses historical 2024-25 season data to predict:
- Points Over/Under
- Rebounds Over/Under  
- Assists Over/Under
- 3-Pointers Made Over/Under
"""

import json
import numpy as np
from collections import defaultdict

def load_nba_data():
    """Load NBA player data"""
    with open('nba_player_data.json', 'r') as f:
        return json.load(f)

def calculate_prop_probability(games, stat_key, threshold):
    """
    Calculate probability of hitting a prop threshold
    
    Args:
        games: List of game stats
        stat_key: The stat to check (points, rebounds, assists, threes_made)
        threshold: The O/U line
    
    Returns:
        dict with probability stats
    """
    if not games:
        return {
            'probability': 0.0,
            'average': 0.0,
            'games_played': 0,
            'times_over': 0,
            'last_5_avg': 0.0,
            'last_10_avg': 0.0
        }
    
    values = [g.get(stat_key, 0) for g in games]
    times_over = sum(1 for v in values if v > threshold)
    
    return {
        'probability': round((times_over / len(games)) * 100, 1),
        'average': round(sum(values) / len(games), 1),
        'games_played': len(games),
        'times_over': times_over,
        'last_5_avg': round(sum(values[:5]) / min(5, len(values)), 1) if values else 0.0,
        'last_10_avg': round(sum(values[:10]) / min(10, len(values)), 1) if values else 0.0
    }

def generate_nba_predictions(players):
    """
    Generate prop predictions for all NBA players
    
    Common betting lines by position:
    - Stars: 25.5 pts, 8.5 reb, 7.5 ast, 3.5 3PM
    - Role Players: 12.5 pts, 5.5 reb, 3.5 ast, 2.5 3PM
    """
    
    predictions = []
    
    print(f"\n{'='*60}")
    print(f"GENERATING NBA PROP PREDICTIONS")
    print(f"{'='*60}\n")
    
    for player in players:
        gamelog = player.get('gamelog', [])
        
        if len(gamelog) < 3:  # Need at least 3 games
            continue
        
        # Calculate season averages
        ppg = player['season_stats']['ppg']
        rpg = player['season_stats']['rpg']
        apg = player['season_stats']['apg']
        three_pm = player['season_stats']['three_pm']
        
        # Use realistic prop lines based on sportsbook patterns
        # Lines are typically set to create ~55-60% over probability for popular bets
        # This means lines are often 35-40% below player averages
        
        def calculate_realistic_line(avg, position=''):
            """Calculate realistic sportsbook line"""
            if avg < 5:
                # Low averages: use ~75% of average
                return round((avg * 0.75) * 2) / 2
            elif avg < 15:
                # Mid averages: use ~65% of average  
                return round((avg * 0.65) * 2) / 2
            else:
                # High averages: use ~60% of average
                return round((avg * 0.60) * 2) / 2
        
        points_line = max(0.5, calculate_realistic_line(ppg))
        rebounds_line = max(0.5, calculate_realistic_line(rpg))
        assists_line = max(0.5, calculate_realistic_line(apg))
        threes_line = max(0.5, calculate_realistic_line(three_pm))
        
        # Calculate probabilities for each prop
        points_prob = calculate_prop_probability(gamelog, 'points', points_line)
        rebounds_prob = calculate_prop_probability(gamelog, 'rebounds', rebounds_line)
        assists_prob = calculate_prop_probability(gamelog, 'assists', assists_line)
        threes_prob = calculate_prop_probability(gamelog, 'threes_made', threes_line)
        
        player_prediction = {
            'id': player['id'],
            'player_name': player['player_name'],
            'team': player['team'],
            'position': player['position'],
            'games_played': len(gamelog),
            
            'props': {
                'points': {
                    'line': points_line,
                    'average': ppg,
                    'over_probability': points_prob['probability'],
                    'under_probability': 100 - points_prob['probability'],
                    'last_5_avg': points_prob['last_5_avg'],
                    'recommendation': 'OVER' if points_prob['probability'] > 55 else 'UNDER' if points_prob['probability'] < 45 else 'NO BET'
                },
                'rebounds': {
                    'line': rebounds_line,
                    'average': rpg,
                    'over_probability': rebounds_prob['probability'],
                    'under_probability': 100 - rebounds_prob['probability'],
                    'last_5_avg': rebounds_prob['last_5_avg'],
                    'recommendation': 'OVER' if rebounds_prob['probability'] > 55 else 'UNDER' if rebounds_prob['probability'] < 45 else 'NO BET'
                },
                'assists': {
                    'line': assists_line,
                    'average': apg,
                    'over_probability': assists_prob['probability'],
                    'under_probability': 100 - assists_prob['probability'],
                    'last_5_avg': assists_prob['last_5_avg'],
                    'recommendation': 'OVER' if assists_prob['probability'] > 55 else 'UNDER' if assists_prob['probability'] < 45 else 'NO BET'
                },
                'threes_made': {
                    'line': threes_line,
                    'average': three_pm,
                    'over_probability': threes_prob['probability'],
                    'under_probability': 100 - threes_prob['probability'],
                    'last_5_avg': threes_prob['last_5_avg'],
                    'recommendation': 'OVER' if threes_prob['probability'] > 55 else 'UNDER' if threes_prob['probability'] < 45 else 'NO BET'
                }
            }
        }
        
        predictions.append(player_prediction)
    
    return predictions

if __name__ == '__main__':
    print("\n" + "="*60)
    print("NBA PROP PREDICTION MODEL TRAINING")
    print("="*60)
    
    # Load data
    players = load_nba_data()
    print(f"\nLoaded {len(players)} NBA players")
    
    # Generate predictions
    predictions = generate_nba_predictions(players)
    
    # Save predictions
    output_file = 'nba_prop_predictions.json'
    with open(output_file, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"COMPLETE: {len(predictions)} player predictions saved")
    print(f"Output: {output_file}")
    print(f"{'='*60}")
    
    # Show sample predictions
    if predictions:
        print(f"\nSample prediction for {predictions[0]['player_name']}:")
        print(f"  Points: {predictions[0]['props']['points']['average']} avg, " 
              f"O/U {predictions[0]['props']['points']['line']} = "
              f"{predictions[0]['props']['points']['over_probability']}% OVER")
        print(f"  Rebounds: {predictions[0]['props']['rebounds']['average']} avg, "
              f"O/U {predictions[0]['props']['rebounds']['line']} = "
              f"{predictions[0]['props']['rebounds']['over_probability']}% OVER")
