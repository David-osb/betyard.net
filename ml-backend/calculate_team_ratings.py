"""
Team Ratings Calculator - Analyzes ESPN player data to generate real team ratings
Calculates offensive/defensive ratings (0-100) from actual 2025 season performance
"""

import json
import os
from typing import Dict
from collections import defaultdict


def calculate_team_ratings_from_player_data():
    """
    Calculate team offensive/defensive ratings from ESPN player game logs
    Returns dict of {team_code: {'offense': rating, 'defense': rating}}
    """
    
    data_file = os.path.join(os.path.dirname(__file__), 'espn_player_data.json')
    
    if not os.path.exists(data_file):
        print("❌ espn_player_data.json not found")
        return {}
    
    with open(data_file, 'r') as f:
        players = json.load(f)
    
    # Aggregate stats by team
    team_stats = defaultdict(lambda: {
        'total_yards': 0,
        'total_tds': 0,
        'games_played': set(),
        'qb_yards': 0,
        'rush_yards': 0,
        'rec_yards': 0
    })
    
    for player in players:
        team = player.get('team')
        position = player.get('position')
        gamelog = player.get('gamelog', [])
        
        if not team or not gamelog:
            continue
        
        # Calculate player totals
        total_yards = sum(g.get('yards', 0) for g in gamelog)
        total_tds = sum(g.get('tds', 0) for g in gamelog)
        
        # Track unique games
        for game in gamelog:
            team_stats[team]['games_played'].add(game.get('game_id'))
        
        team_stats[team]['total_yards'] += total_yards
        team_stats[team]['total_tds'] += total_tds
        
        # Break down by position
        if position == 'QB':
            team_stats[team]['qb_yards'] += total_yards
        elif position == 'RB':
            team_stats[team]['rush_yards'] += total_yards
        elif position in ['WR', 'TE']:
            team_stats[team]['rec_yards'] += total_yards
    
    # Calculate ratings
    team_ratings = {}
    
    # Find league averages for normalization
    all_yards_per_game = []
    all_tds_per_game = []
    
    for team, stats in team_stats.items():
        games = len(stats['games_played'])
        if games > 0:
            yards_per_game = stats['total_yards'] / games
            tds_per_game = stats['total_tds'] / games
            all_yards_per_game.append(yards_per_game)
            all_tds_per_game.append(tds_per_game)
    
    if not all_yards_per_game:
        return {}
    
    league_avg_yards = sum(all_yards_per_game) / len(all_yards_per_game)
    league_avg_tds = sum(all_tds_per_game) / len(all_tds_per_game)
    
    print(f"\nLeague Averages:")
    print(f"  Yards/game: {league_avg_yards:.1f}")
    print(f"  TDs/game: {league_avg_tds:.2f}")
    
    # Calculate each team's rating
    for team, stats in team_stats.items():
        games = len(stats['games_played'])
        if games == 0:
            continue
        
        yards_per_game = stats['total_yards'] / games
        tds_per_game = stats['total_tds'] / games
        
        # Normalize to 0-100 scale
        # 75 = league average
        # Each 10% above/below average = ±5 points
        
        yards_ratio = yards_per_game / league_avg_yards
        tds_ratio = tds_per_game / league_avg_tds
        
        # Weighted average (60% yards, 40% TDs)
        offense_rating = 75 + ((yards_ratio - 1.0) * 25 * 0.6) + ((tds_ratio - 1.0) * 25 * 0.4)
        
        # Clamp to 0-100
        offense_rating = max(50, min(100, round(offense_rating)))
        
        # For defense, we'd need opponent stats (not available in current data)
        # Use inverse of offense as rough estimate for now
        # TODO: Calculate real defensive rating from opponent performance
        defense_rating = 75  # Placeholder
        
        team_ratings[team] = {
            'offense': offense_rating,
            'defense': defense_rating,
            'yards_per_game': round(yards_per_game, 1),
            'tds_per_game': round(tds_per_game, 2),
            'games': games
        }
    
    return team_ratings


def save_team_ratings():
    """Calculate and save team ratings to JSON file"""
    
    print("Calculating team ratings from ESPN player data...")
    print("=" * 60)
    
    ratings = calculate_team_ratings_from_player_data()
    
    if not ratings:
        print("❌ Failed to calculate ratings")
        return
    
    # Save to file
    output_file = os.path.join(os.path.dirname(__file__), 'team_ratings.json')
    with open(output_file, 'w') as f:
        json.dump(ratings, f, indent=2)
    
    print(f"\n[SUCCESS] Saved {len(ratings)} team ratings to team_ratings.json")
    
    # Display top offenses
    sorted_teams = sorted(ratings.items(), key=lambda x: x[1]['offense'], reverse=True)
    
    print("\nTop 10 Offenses (by rating):")
    print("-" * 60)
    for i, (team, data) in enumerate(sorted_teams[:10], 1):
        print(f"{i:2d}. {team:3s} - {data['offense']:3d} offense ({data['yards_per_game']:.1f} ypg, {data['tds_per_game']:.2f} td/g)")
    
    print("\nBottom 5 Offenses:")
    print("-" * 60)
    for i, (team, data) in enumerate(sorted_teams[-5:], 1):
        print(f"{i:2d}. {team:3s} - {data['offense']:3d} offense ({data['yards_per_game']:.1f} ypg, {data['tds_per_game']:.2f} td/g)")


if __name__ == '__main__':
    save_team_ratings()
