"""
Train NHL player prop prediction models
Uses historical 2024-25 season data to predict:
- Anytime Goal Scorer probability
- Assists Over/Under
- Shots Over/Under
- Goalie Saves Over/Under
"""

import json
import numpy as np

def load_nhl_data():
    """Load NHL player data"""
    with open('nhl_player_data.json', 'r') as f:
        return json.load(f)

def calculate_goal_scorer_odds(goal_probability):
    """
    Convert goal probability to American odds
    
    Typical odds:
    - 50% probability = +100
    - 33% probability = +200
    - 25% probability = +300
    - 20% probability = +400
    """
    if goal_probability == 0:
        return 10000  # Very long odds for 0% probability
    if goal_probability >= 50:
        # Favorite odds (negative)
        odds = int(-100 * (goal_probability / (100 - goal_probability)))
    else:
        # Underdog odds (positive)
        odds = int(100 * ((100 - goal_probability) / goal_probability))
    
    return odds

def generate_nhl_predictions(players):
    """
    Generate prop predictions for all NHL players
    """
    
    skater_predictions = []
    goalie_predictions = []
    
    print(f"\n{'='*60}")
    print(f"GENERATING NHL PROP PREDICTIONS")
    print(f"{'='*60}\n")
    
    for player in players:
        position = player['position']
        gamelog = player.get('gamelog', [])
        
        if len(gamelog) < 3:  # Need at least 3 games
            continue
        
        if position == 'G':  # Goalie
            # Goalie predictions
            avg_saves = player['season_stats']['avg_saves']
            avg_ga = player['season_stats']['avg_goals_against']
            
            # Calculate save probability for common lines
            saves_line = 0.5 + round(avg_saves * 2) / 2
            
            saves_over_count = sum(1 for g in gamelog if g.get('saves', 0) > saves_line)
            saves_probability = round((saves_over_count / len(gamelog)) * 100, 1)
            
            goalie_predictions.append({
                'id': player['id'],
                'player_name': player['player_name'],
                'team': player['team'],
                'position': position,
                'games_played': len(gamelog),
                
                'props': {
                    'saves': {
                        'line': saves_line,
                        'average': avg_saves,
                        'over_probability': saves_probability,
                        'under_probability': 100 - saves_probability,
                        'recommendation': 'OVER' if saves_probability > 55 else 'UNDER' if saves_probability < 45 else 'NO BET'
                    },
                    'goals_against': {
                        'average': avg_ga,
                        'under_2_5_probability': round((sum(1 for g in gamelog if g.get('goals_against', 0) < 2.5) / len(gamelog)) * 100, 1)
                    }
                }
            })
            
        else:  # Skater (C, LW, RW, D)
            # Use goal_stats from data
            goal_stats = player.get('goal_stats', {})
            season_stats = player['season_stats']
            
            goal_probability = goal_stats.get('goal_probability', 0.0)
            total_goals = season_stats.get('goals', 0)
            total_assists = season_stats.get('assists', 0)
            avg_shots = season_stats.get('avg_shots', 0.0)
            
            # Calculate anytime goal scorer odds
            goal_odds = calculate_goal_scorer_odds(goal_probability)
            
            # Assists and shots probabilities
            assists_line = 0.5
            shots_line = 0.5 + round(avg_shots * 2) / 2
            
            assists_over = sum(1 for g in gamelog if g.get('assists', 0) > assists_line)
            shots_over = sum(1 for g in gamelog if g.get('shots', 0) > shots_line)
            
            assists_prob = round((assists_over / len(gamelog)) * 100, 1) if len(gamelog) > 0 else 0
            shots_prob = round((shots_over / len(gamelog)) * 100, 1) if len(gamelog) > 0 else 0
            
            skater_predictions.append({
                'id': player['id'],
                'player_name': player['player_name'],
                'team': player['team'],
                'position': position,
                'games_played': len(gamelog),
                
                'props': {
                    'anytime_goal': {
                        'probability': goal_probability,
                        'odds': goal_odds,
                        'season_goals': total_goals,
                        'games_with_goal': goal_stats.get('games_with_goal', 0),
                        'recommendation': 'BET' if goal_probability > 25 else 'NO BET'
                    },
                    'assists': {
                        'line': assists_line,
                        'over_probability': assists_prob,
                        'under_probability': 100 - assists_prob,
                        'season_total': total_assists,
                        'recommendation': 'OVER' if assists_prob > 55 else 'UNDER' if assists_prob < 45 else 'NO BET'
                    },
                    'shots': {
                        'line': shots_line,
                        'average': avg_shots,
                        'over_probability': shots_prob,
                        'under_probability': 100 - shots_prob,
                        'recommendation': 'OVER' if shots_prob > 55 else 'UNDER' if shots_prob < 45 else 'NO BET'
                    }
                }
            })
    
    return {
        'skaters': skater_predictions,
        'goalies': goalie_predictions
    }

if __name__ == '__main__':
    print("\n" + "="*60)
    print("NHL PROP PREDICTION MODEL TRAINING")
    print("="*60)
    
    # Load data
    players = load_nhl_data()
    print(f"\nLoaded {len(players)} NHL players")
    
    # Generate predictions
    predictions = generate_nhl_predictions(players)
    
    # Save predictions
    output_file = 'nhl_prop_predictions.json'
    with open(output_file, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"COMPLETE:")
    print(f"  Skaters: {len(predictions['skaters'])} predictions")
    print(f"  Goalies: {len(predictions['goalies'])} predictions")
    print(f"Output: {output_file}")
    print(f"{'='*60}")
    
    # Show sample predictions
    if predictions['skaters']:
        sample = predictions['skaters'][0]
        print(f"\nSample skater prediction for {sample['player_name']}:")
        print(f"  Anytime Goal: {sample['props']['anytime_goal']['probability']}% "
              f"({sample['props']['anytime_goal']['odds']:+d})")
    
    if predictions['goalies']:
        sample = predictions['goalies'][0]
        print(f"\nSample goalie prediction for {sample['player_name']}:")
        print(f"  Saves O/U {sample['props']['saves']['line']}: "
              f"{sample['props']['saves']['over_probability']}% OVER")
