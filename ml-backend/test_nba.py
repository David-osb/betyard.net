#!/usr/bin/env python3
"""
Simple test script for NBA API
"""

def test_nba_api():
    try:
        print("🏀 Testing nba_api import...")
        from nba_api.live.nba.endpoints import scoreboard
        print("✅ scoreboard import successful")
        
        from nba_api.stats.endpoints import leaguegamefinder
        print("✅ leaguegamefinder import successful")
        
        print("🏀 Testing scoreboard fetch...")
        games_today = scoreboard.ScoreBoard()
        print("✅ ScoreBoard object created")
        
        games_data = games_today.get_dict()
        print("✅ ScoreBoard data fetched")
        print(f"Data keys: {list(games_data.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_nba_api()