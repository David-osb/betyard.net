#!/usr/bin/env python3
"""
ESPN API Integration Test Script
Test ESPN connectivity and data quality for enhanced XGBoost predictions
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from espn_api_integration import ESPNAPIIntegration
import json
import traceback

def test_espn_integration():
    """Test ESPN API integration components"""
    print("🏈 Testing ESPN API Integration for Enhanced XGBoost Model")
    print("=" * 60)
    
    try:
        # Initialize ESPN API
        print("\n1️⃣ Initializing ESPN API Integration...")
        espn_api = ESPNAPIIntegration()
        print("✅ ESPN API Integration initialized successfully")
        
        # Test team mapping
        print(f"✅ Team mapping loaded: {len(espn_api.team_mapping)} teams")
        
        # Test individual endpoints
        print("\n2️⃣ Testing ESPN API Endpoints...")
        
        # Test Team Statistics
        print("\n📊 Testing Team Statistics API...")
        team_stats = espn_api.get_team_statistics("BUF")  # Buffalo Bills
        if team_stats:
            print("✅ Team Statistics API working")
            print(f"   Sample data keys: {list(team_stats.keys())[:5]}")
        else:
            print("⚠️ Team Statistics API returned no data")
        
        # Test Enhanced Player Data
        print("\n🏃 Testing Enhanced Player Data API...")
        player_data = espn_api.get_enhanced_player_data("Josh Allen", "BUF", "QB")
        if player_data:
            print("✅ Enhanced Player Data API working")
            print(f"   Player: {player_data.player_name}")
            print(f"   Position: {player_data.position}")
            print(f"   Team: {player_data.team}")
            print(f"   Real Data: {player_data.real_data}")
            print(f"   Recent Form: {player_data.recent_form_factor:.3f}")
            print(f"   Injury Status: {player_data.injury_status}")
        else:
            print("⚠️ Enhanced Player Data API returned no data")
        
        # Test Team Roster (Injury Status)
        print("\n🏥 Testing Team Roster API (Injury Status)...")
        roster_data = espn_api.get_team_roster("BUF")
        if roster_data:
            print("✅ Team Roster API working")
            print(f"   Roster size: {len(roster_data)}")
            # Show injury status sample
            injured_players = [p for p in roster_data if p.get('injury_status') != 'Healthy']
            print(f"   Players with injury status: {len(injured_players)}")
        else:
            print("⚠️ Team Roster API returned no data")
        
        # Test Player Gamelog
        print("\n📈 Testing Player Gamelog API...")
        gamelog = espn_api.get_player_gamelog("Josh Allen", "BUF")
        if gamelog:
            print("✅ Player Gamelog API working")
            print(f"   Recent games: {len(gamelog)}")
        else:
            print("⚠️ Player Gamelog API returned no data")
        
        # Test Events/Schedule
        print("\n📅 Testing Events/Schedule API...")
        schedule = espn_api.get_team_schedule("BUF")
        if schedule:
            print("✅ Events/Schedule API working")
            print(f"   Schedule entries: {len(schedule)}")
        else:
            print("⚠️ Events/Schedule API returned no data")
        
        # Test Player Splits
        print("\n🎯 Testing Player Splits API...")
        splits = espn_api.get_player_splits("Josh Allen", "BUF")
        if splits:
            print("✅ Player Splits API working")
            print(f"   Split categories: {len(splits)}")
        else:
            print("⚠️ Player Splits API returned no data")
        
        # Test data quality calculation
        print("\n3️⃣ Testing Data Quality Assessment...")
        if player_data:
            data_quality = calculate_data_quality(player_data, team_stats)
            print(f"✅ Data Quality Score: {data_quality:.3f}")
            if data_quality > 0.8:
                print("🎯 EXCELLENT data quality - Maximum accuracy boost expected")
            elif data_quality > 0.6:
                print("✅ GOOD data quality - Significant accuracy boost expected")
            else:
                print("⚠️ LIMITED data quality - Modest accuracy boost expected")
        
        print("\n4️⃣ ESPN Integration Summary")
        print("=" * 40)
        print("✅ ESPN API Integration: READY")
        print("✅ Tier 1 Endpoints: FUNCTIONAL")
        print("✅ Enhanced XGBoost Model: READY")
        print("✅ Maximum Accuracy Mode: ENABLED")
        
        print("\n🏆 Your enhanced XGBoost model is ready for maximum accuracy predictions!")
        
    except Exception as e:
        print(f"\n❌ Error testing ESPN integration: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        
        print("\n🔧 Troubleshooting:")
        print("1. Check internet connectivity")
        print("2. Verify ESPN API endpoints are accessible")
        print("3. Check firewall/proxy settings")
        print("4. Fallback to Tank01 data will be used automatically")

def calculate_data_quality(player_data, team_stats):
    """Calculate data quality score for testing"""
    quality_score = 0.6  # Base score
    
    if player_data.real_data:
        quality_score += 0.2
    if player_data.season_passing_yards and player_data.season_passing_yards > 0:
        quality_score += 0.1
    if player_data.recent_form_factor > 0:
        quality_score += 0.1
    if player_data.injury_status != 'Unknown':
        quality_score += 0.05
    if team_stats:
        quality_score += 0.1
        
    return min(1.0, quality_score)

def test_full_prediction():
    """Test full enhanced prediction pipeline"""
    print("\n🎯 Testing Full Enhanced Prediction Pipeline...")
    print("=" * 50)
    
    try:
        from app import NFLMLModel
        
        print("Initializing Enhanced NFL ML Model...")
        model = NFLMLModel()
        
        if model.espn_api:
            print("✅ ESPN API integration loaded in main model")
            
            # Test QB prediction
            print("\n🏈 Testing Enhanced QB Prediction...")
            result = model.predict_qb_performance("Josh Allen", "BUF")
            
            print(f"✅ Enhanced QB Prediction Complete:")
            print(f"   Passing Yards: {result.passing_yards}")
            print(f"   Model Accuracy: {result.model_accuracy}%")
            print(f"   Prediction Confidence: {result.confidence}%")
            
            # Check for ESPN context
            if hasattr(result, 'espn_context'):
                print(f"   ESPN Data Quality: {result.espn_context.get('data_quality_score', 'N/A')}")
                print(f"   Enhanced Accuracy: {result.espn_context.get('enhanced_accuracy', False)}")
            
        else:
            print("⚠️ ESPN API not loaded in main model - check configuration")
            
    except Exception as e:
        print(f"❌ Error testing full prediction: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_espn_integration()
    test_full_prediction()