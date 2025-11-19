"""
Test script to verify ESPN API access for all sports
Fetches a few star players from each league to validate data structure
"""

import requests
import json

def test_nba():
    """Test NBA API with Trae Young"""
    print("\n" + "="*60)
    print("TESTING NBA API")
    print("="*60)
    
    url = "https://site.web.api.espn.com/apis/common/v3/sports/basketball/nba/athletes/3136195/gamelog"
    response = requests.get(url, timeout=15)
    data = response.json()
    
    stat_names = data.get('names', [])
    points_idx = next((i for i, n in enumerate(stat_names) if 'points' in n.lower()), None)
    rebounds_idx = next((i for i, n in enumerate(stat_names) if 'totalrebounds' in n.lower().replace(' ', '')), None)
    assists_idx = next((i for i, n in enumerate(stat_names) if 'assists' in n.lower()), None)
    
    games = []
    for season in data.get('seasonTypes', []):
        if 'Regular Season' in season.get('displayName', ''):
            for cat in season.get('categories', []):
                for event in cat.get('events', []):
                    stats = event.get('stats', [])
                    if stats:
                        games.append({
                            'pts': stats[points_idx],
                            'reb': stats[rebounds_idx],
                            'ast': stats[assists_idx]
                        })
    
    print(f"✓ Trae Young: {len(games)} games found")
    print(f"  Last 3 games: {games[:3]}")
    return len(games) > 0

def test_nhl():
    """Test NHL API with Connor McDavid"""
    print("\n" + "="*60)
    print("TESTING NHL API")
    print("="*60)
    
    url = "https://site.web.api.espn.com/apis/common/v3/sports/hockey/nhl/athletes/3895074/gamelog"
    response = requests.get(url, timeout=15)
    data = response.json()
    
    stat_names = data.get('names', [])
    goals_idx = next((i for i, n in enumerate(stat_names) if n.lower() == 'g' or 'goals' in n.lower()), None)
    assists_idx = next((i for i, n in enumerate(stat_names) if n.lower() == 'a' or 'assists' in n.lower()), None)
    
    games = []
    for season in data.get('seasonTypes', []):
        if 'Regular Season' in season.get('displayName', ''):
            for cat in season.get('categories', []):
                for event in cat.get('events', []):
                    stats = event.get('stats', [])
                    if stats:
                        games.append({
                            'goals': stats[goals_idx] if goals_idx and goals_idx < len(stats) else 0,
                            'assists': stats[assists_idx] if assists_idx and assists_idx < len(stats) else 0
                        })
    
    print(f"✓ Connor McDavid: {len(games)} games found")
    print(f"  Last 3 games: {games[:3]}")
    return len(games) > 0

def test_mlb():
    """Test MLB API with Aaron Judge"""
    print("\n" + "="*60)
    print("TESTING MLB API (2024 season)")
    print("="*60)
    
    url = "https://site.web.api.espn.com/apis/common/v3/sports/baseball/mlb/athletes/33192/gamelog"
    response = requests.get(url, timeout=15)
    data = response.json()
    
    stat_names = data.get('names', [])
    hits_idx = next((i for i, n in enumerate(stat_names) if n.lower() == 'h' or 'hits' in n.lower()), None)
    hr_idx = next((i for i, n in enumerate(stat_names) if n.lower() == 'hr' or 'home runs' in n.lower()), None)
    
    games = []
    for season in data.get('seasonTypes', []):
        if 'Regular Season' in season.get('displayName', ''):
            for cat in season.get('categories', []):
                for event in cat.get('events', []):
                    stats = event.get('stats', [])
                    if stats:
                        games.append({
                            'hits': stats[hits_idx] if hits_idx and hits_idx < len(stats) else 0,
                            'hr': stats[hr_idx] if hr_idx and hr_idx < len(stats) else 0
                        })
    
    print(f"✓ Aaron Judge: {len(games)} games found")
    print(f"  Last 3 games: {games[:3]}")
    return len(games) > 0

def test_mls():
    """Test MLS API with Lionel Messi"""
    print("\n" + "="*60)
    print("TESTING MLS API (2024 season)")
    print("="*60)
    
    url = "https://site.web.api.espn.com/apis/common/v3/sports/soccer/usa.1/athletes/45843/gamelog"
    response = requests.get(url, timeout=15)
    data = response.json()
    
    stat_names = data.get('names', [])
    goals_idx = next((i for i, n in enumerate(stat_names) if n.lower() == 'g' or 'goals' in n.lower()), None)
    assists_idx = next((i for i, n in enumerate(stat_names) if n.lower() == 'a' or 'assists' in n.lower()), None)
    
    games = []
    for season in data.get('seasonTypes', []):
        if 'Regular Season' in season.get('displayName', ''):
            for cat in season.get('categories', []):
                for event in cat.get('events', []):
                    stats = event.get('stats', [])
                    if stats:
                        games.append({
                            'goals': stats[goals_idx] if goals_idx and goals_idx < len(stats) else 0,
                            'assists': stats[assists_idx] if assists_idx and assists_idx < len(stats) else 0
                        })
    
    print(f"✓ Lionel Messi: {len(games)} games found")
    print(f"  Last 3 games: {games[:3]}")
    return len(games) > 0

if __name__ == '__main__':
    print("\n" + "="*60)
    print("ESPN API VALIDATION TEST")
    print("="*60)
    
    results = {
        'NBA': test_nba(),
        'NHL': test_nhl(),
        'MLB': test_mlb(),
        'MLS': test_mls()
    }
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for sport, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{sport}: {status}")
    
    all_pass = all(results.values())
    print(f"\nOverall: {'✓ ALL APIS WORKING' if all_pass else '✗ SOME APIS FAILED'}")
