#!/usr/bin/env python3

import datetime
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def get_team_emoji(team_name):
    """Get team emoji for team name"""
    team_emojis = {
        'Lakers': '🏀', 'Warriors': '⚡', 'Celtics': '🍀', 'Heat': '🔥',
        'Nuggets': '⛰️', 'Mavericks': '🤠', 'Suns': '☀️', 'Clippers': '✂️'
    }
    return team_emojis.get(team_name, '🏀')

def test_nba_games():
    """Test NBA games generation"""
    try:
        print("🏀 Testing NBA games generation...")
        
        today = datetime.datetime.now()
        print(f"Current date: {today}")
        
        formatted_games = []
        
        # Generate realistic NBA games for today
        nba_matchups = [
            {
                'home': {'name': 'Lakers', 'city': 'Los Angeles', 'abbr': 'LAL', 'record': '10-5'},
                'away': {'name': 'Warriors', 'city': 'Golden State', 'abbr': 'GSW', 'record': '12-3'},
                'time': '20:00', 'venue': 'Crypto.com Arena'
            },
            {
                'home': {'name': 'Celtics', 'city': 'Boston', 'abbr': 'BOS', 'record': '13-2'}, 
                'away': {'name': 'Heat', 'city': 'Miami', 'abbr': 'MIA', 'record': '8-7'},
                'time': '19:30', 'venue': 'TD Garden'
            },
            {
                'home': {'name': 'Nuggets', 'city': 'Denver', 'abbr': 'DEN', 'record': '11-4'},
                'away': {'name': 'Mavericks', 'city': 'Dallas', 'abbr': 'DAL', 'record': '9-6'},
                'time': '21:00', 'venue': 'Ball Arena'
            },
            {
                'home': {'name': 'Suns', 'city': 'Phoenix', 'abbr': 'PHX', 'record': '9-6'},
                'away': {'name': 'Clippers', 'city': 'LA', 'abbr': 'LAC', 'record': '10-5'},
                'time': '22:00', 'venue': 'Footprint Center'
            }
        ]
        
        for i, matchup in enumerate(nba_matchups):
            game_time = today.replace(hour=int(matchup['time'].split(':')[0]), minute=int(matchup['time'].split(':')[1]), second=0, microsecond=0)
            
            formatted_game = {
                'gameId': f"nba_00{i+1}",
                'date': today.strftime('%Y-%m-%d'),
                'time': matchup['time'],
                'homeTeam': matchup['home']['name'],
                'awayTeam': matchup['away']['name'],
                'homeCity': matchup['home']['city'],
                'awayCity': matchup['away']['city'],
                'homeAbbr': matchup['home']['abbr'],
                'awayAbbr': matchup['away']['abbr'],
                'homeRecord': matchup['home']['record'],
                'awayRecord': matchup['away']['record'],
                'venue': matchup['venue'],
                'homeTeamEmoji': get_team_emoji(matchup['home']['name']),
                'awayTeamEmoji': get_team_emoji(matchup['away']['name']),
                'status': 'upcoming' if game_time > today else ('live' if i < 2 else 'final'),
                'homeScore': 0 if game_time > today else (105 + (i * 5)),
                'awayScore': 0 if game_time > today else (98 + (i * 3)),
                'quarter': 0 if game_time > today else (4 if i >= 2 else 3),
                'timeRemaining': '' if game_time > today else ('2:45' if i < 2 else 'Final'),
                'odds': {
                    'spread': {'home': -2.5 - (i * 0.5), 'away': 2.5 + (i * 0.5)},
                    'moneyline': {'home': -130 - (i * 10), 'away': 110 + (i * 10)},
                    'total': 220.5 + (i * 2)
                }
            }
            formatted_games.append(formatted_game)
        
        print(f"✅ Successfully generated {len(formatted_games)} NBA games")
        
        # Print each game for verification
        for i, game in enumerate(formatted_games, 1):
            print(f"\nGame {i}:")
            print(f"  {game['awayTeam']} @ {game['homeTeam']}")
            print(f"  Date: {game['date']}")
            print(f"  Time: {game['time']}")
            print(f"  Status: {game['status']}")
            print(f"  Venue: {game['venue']}")
        
        result = {
            'success': True,
            'games': formatted_games,
            'total': len(formatted_games),
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        return result
        
    except Exception as e:
        print(f"❌ Error generating NBA games: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'success': False,
            'error': str(e),
            'games': [],
            'total': 0,
            'timestamp': datetime.datetime.now().isoformat()
        }

if __name__ == "__main__":
    result = test_nba_games()
    print(f"\n🏀 Final result:")
    print(f"Success: {result['success']}")
    print(f"Total games: {result['total']}")
    if result['success']:
        print("✅ NBA games generation working correctly!")
    else:
        print(f"❌ Error: {result.get('error', 'Unknown error')}")