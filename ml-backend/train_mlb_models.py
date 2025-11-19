"""
Train MLB player prop prediction models
Uses historical 2024 season data to predict:
- Hits Over/Under (2+ hits common prop)
- Home Runs (anytime HR)
- RBIs Over/Under
- Pitcher Strikeouts Over/Under
"""

import json
import numpy as np

def load_mlb_data():
    """Load MLB player data"""
    with open('mlb_player_data.json', 'r') as f:
        return json.load(f)

def calculate_hr_odds(hr_probability):
    """Convert home run probability to American odds"""
    if hr_probability == 0:
        return 10000  # Very long odds for 0% probability
    if hr_probability >= 50:
        odds = int(-100 * (hr_probability / (100 - hr_probability)))
    else:
        odds = int(100 * ((100 - hr_probability) / hr_probability))
    return odds

def generate_mlb_predictions(players):
    """
    Generate prop predictions for all MLB players
    """
    
    hitter_predictions = []
    pitcher_predictions = []
    
    print(f"\n{'='*60}")
    print(f"GENERATING MLB PROP PREDICTIONS")
    print(f"{'='*60}\n")
    
    for player in players:
        position = player['position']
        gamelog = player.get('gamelog', [])
        
        if len(gamelog) < 10:  # Need at least 10 games for baseball
            continue
        
        if position in ['P', 'SP', 'RP']:  # Pitcher
            season_stats = player['season_stats']
            avg_k = season_stats['avg_strikeouts']
            total_k = season_stats['total_strikeouts']
            
            # Common strikeout lines: 4.5, 5.5, 6.5, 7.5
            k_line = 0.5 + round(avg_k)
            
            k_over = sum(1 for g in gamelog if g.get('strikeouts', 0) > k_line)
            k_probability = round((k_over / len(gamelog)) * 100, 1)
            
            pitcher_predictions.append({
                'id': player['id'],
                'player_name': player['player_name'],
                'team': player['team'],
                'position': position,
                'games_played': len(gamelog),
                
                'props': {
                    'strikeouts': {
                        'line': k_line,
                        'average': avg_k,
                        'over_probability': k_probability,
                        'under_probability': 100 - k_probability,
                        'season_total': total_k,
                        'recommendation': 'OVER' if k_probability > 55 else 'UNDER' if k_probability < 45 else 'NO BET'
                    },
                    'hits_allowed': {
                        'average': season_stats['avg_hits_allowed'],
                        'under_6_5_probability': round((sum(1 for g in gamelog if g.get('hits_allowed', 0) < 6.5) / len(gamelog)) * 100, 1)
                    },
                    'earned_runs': {
                        'average': season_stats['avg_earned_runs'],
                        'under_3_5_probability': round((sum(1 for g in gamelog if g.get('earned_runs', 0) < 3.5) / len(gamelog)) * 100, 1)
                    }
                }
            })
            
        else:  # Hitter
            season_stats = player['season_stats']
            hit_stats = player.get('hit_stats', {})
            
            avg_hits = season_stats['avg_hits']
            total_hits = season_stats['total_hits']
            total_hrs = season_stats['total_home_runs']
            total_rbis = season_stats['total_rbis']
            
            # 2+ hits probability
            hits_2plus_prob = hit_stats.get('probability_2plus_hits', 0.0)
            
            # Home run probability (games with HR / games played)
            hr_games = sum(1 for g in gamelog if g.get('home_runs', 0) > 0)
            hr_probability = round((hr_games / len(gamelog)) * 100, 1)
            hr_odds = calculate_hr_odds(hr_probability)
            
            # RBI probability (1+ RBI)
            rbi_games = sum(1 for g in gamelog if g.get('rbis', 0) > 0)
            rbi_probability = round((rbi_games / len(gamelog)) * 100, 1)
            
            hitter_predictions.append({
                'id': player['id'],
                'player_name': player['player_name'],
                'team': player['team'],
                'position': position,
                'games_played': len(gamelog),
                
                'props': {
                    'hits': {
                        'line': 1.5,  # 2+ hits
                        'average': avg_hits,
                        'over_probability': hits_2plus_prob,
                        'under_probability': 100 - hits_2plus_prob,
                        'season_total': total_hits,
                        'recommendation': 'OVER' if hits_2plus_prob > 55 else 'UNDER' if hits_2plus_prob < 45 else 'NO BET'
                    },
                    'home_run': {
                        'probability': hr_probability,
                        'odds': hr_odds,
                        'season_total': total_hrs,
                        'games_with_hr': hr_games,
                        'recommendation': 'BET' if hr_probability > 15 else 'NO BET'
                    },
                    'rbis': {
                        'line': 0.5,  # 1+ RBI
                        'over_probability': rbi_probability,
                        'under_probability': 100 - rbi_probability,
                        'season_total': total_rbis,
                        'recommendation': 'OVER' if rbi_probability > 55 else 'UNDER' if rbi_probability < 45 else 'NO BET'
                    }
                }
            })
    
    return {
        'hitters': hitter_predictions,
        'pitchers': pitcher_predictions
    }

if __name__ == '__main__':
    print("\n" + "="*60)
    print("MLB PROP PREDICTION MODEL TRAINING")
    print("="*60)
    
    # Load data
    players = load_mlb_data()
    print(f"\nLoaded {len(players)} MLB players")
    
    # Generate predictions
    predictions = generate_mlb_predictions(players)
    
    # Save predictions
    output_file = 'mlb_prop_predictions.json'
    with open(output_file, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"COMPLETE:")
    print(f"  Hitters: {len(predictions['hitters'])} predictions")
    print(f"  Pitchers: {len(predictions['pitchers'])} predictions")
    print(f"Output: {output_file}")
    print(f"{'='*60}")
    
    # Show sample predictions
    if predictions['hitters']:
        sample = predictions['hitters'][0]
        print(f"\nSample hitter prediction for {sample['player_name']}:")
        print(f"  2+ Hits: {sample['props']['hits']['over_probability']}%")
        print(f"  Home Run: {sample['props']['home_run']['probability']}% "
              f"({sample['props']['home_run']['odds']:+d})")
    
    if predictions['pitchers']:
        sample = predictions['pitchers'][0]
        print(f"\nSample pitcher prediction for {sample['player_name']}:")
        print(f"  Strikeouts O/U {sample['props']['strikeouts']['line']}: "
              f"{sample['props']['strikeouts']['over_probability']}% OVER")
