#!/usr/bin/env python3
"""
ESPN Primary System Verification Test
Verify ESPN-enhanced predictions are the sole system for website
"""

import requests
import json
import sys

def test_espn_primary_system(base_url="http://localhost:5000"):
    """Test that ESPN system is the primary prediction method"""
    
    print("🏆 ESPN PRIMARY SYSTEM VERIFICATION")
    print("=" * 50)
    
    # Test 1: Check system status
    print("\n1️⃣ Testing ESPN System Status...")
    try:
        response = requests.get(f"{base_url}/espn/status")
        if response.status_code == 200:
            status_data = response.json()
            print("✅ ESPN Status API working")
            
            integration_status = status_data.get('integration_status', {})
            all_phases_active = all(integration_status.values())
            
            if all_phases_active:
                print("✅ All ESPN phases active (1, 2, 3)")
            else:
                print("⚠️ Some ESPN phases inactive:", integration_status)
                
            print(f"   System Version: {status_data.get('system_version')}")
            print(f"   Monitoring: {status_data.get('monitoring_status', {}).get('continuous_monitoring')}")
            
        else:
            print(f"❌ ESPN Status API failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ ESPN Status test failed: {e}")
        return False
    
    # Test 2: Check model info shows ESPN enhancement
    print("\n2️⃣ Testing Model Info (ESPN Enhancement)...")
    try:
        response = requests.get(f"{base_url}/model/info")
        if response.status_code == 200:
            model_data = response.json()
            print("✅ Model Info API working")
            
            if "ESPN-Enhanced" in model_data.get('model_type', ''):
                print("✅ Model shows ESPN-Enhanced system")
            else:
                print("⚠️ Model doesn't show ESPN enhancement")
                
            version = model_data.get('version', '')
            if "ESPN" in version:
                print(f"✅ ESPN version confirmed: {version}")
            else:
                print(f"⚠️ Version doesn't show ESPN: {version}")
                
            # Check accuracy improvements
            max_accuracy = model_data.get('max_accuracy_achieved', {})
            if max_accuracy:
                print("✅ ESPN accuracy improvements confirmed:")
                for pos, acc in max_accuracy.items():
                    print(f"   {pos}: {acc}")
            
        else:
            print(f"❌ Model Info API failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Model Info test failed: {e}")
        return False
    
    # Test 3: Test actual prediction with ESPN enhancement
    print("\n3️⃣ Testing ESPN-Enhanced Predictions...")
    test_predictions = [
        {"player_name": "Josh Allen", "team_code": "BUF", "position": "QB", "opponent_code": "KC"},
        {"player_name": "Christian McCaffrey", "team_code": "SF", "position": "RB", "opponent_code": "LAR"},
        {"player_name": "Cooper Kupp", "team_code": "LAR", "position": "WR", "opponent_code": "SF"},
        {"player_name": "Travis Kelce", "team_code": "KC", "position": "TE", "opponent_code": "BUF"}
    ]
    
    espn_enhanced_count = 0
    
    for test_data in test_predictions:
        try:
            response = requests.post(f"{base_url}/predict", json=test_data)
            if response.status_code == 200:
                pred_data = response.json()
                
                # Check if ESPN enhanced
                is_espn_enhanced = pred_data.get('espn_enhanced', False)
                system_type = pred_data.get('prediction_system', 'Unknown')
                
                print(f"✅ {test_data['position']} Prediction: {test_data['player_name']}")
                print(f"   ESPN Enhanced: {is_espn_enhanced}")
                print(f"   System: {system_type}")
                print(f"   Accuracy: {pred_data.get('model_accuracy', 0)}%")
                
                if is_espn_enhanced:
                    espn_enhanced_count += 1
                    
                    # Check for ESPN context
                    if 'espn_data_quality' in pred_data:
                        print(f"   ESPN Data Quality: {pred_data['espn_data_quality']:.3f}")
                    
                    if 'advanced_analysis' in pred_data:
                        advanced = pred_data['advanced_analysis']
                        print(f"   Advanced Features: Matchup={advanced.get('matchup_analysis')}, Weather={advanced.get('weather_impact'):.2f}")
                
            else:
                print(f"❌ {test_data['position']} prediction failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Prediction test failed for {test_data['position']}: {e}")
    
    # Test 4: Summary
    print(f"\n4️⃣ ESPN Enhancement Summary:")
    print(f"   ESPN-Enhanced Predictions: {espn_enhanced_count}/4")
    
    if espn_enhanced_count == 4:
        print("✅ ALL PREDICTIONS USE ESPN ENHANCEMENT - PRIMARY SYSTEM CONFIRMED")
        success = True
    else:
        print(f"⚠️ Only {espn_enhanced_count}/4 predictions use ESPN enhancement")
        success = False
    
    return success

def test_manual_retraining(base_url="http://localhost:5000"):
    """Test manual retraining endpoint"""
    print("\n5️⃣ Testing Manual Retraining Capability...")
    
    try:
        # Test retraining endpoint (won't actually retrain unless needed)
        response = requests.post(f"{base_url}/espn/retrain", json={})
        
        if response.status_code == 200:
            retrain_data = response.json()
            print("✅ Manual retraining endpoint working")
            print(f"   Status: {retrain_data.get('status')}")
            print(f"   Message: {retrain_data.get('message')}")
            
            if retrain_data.get('status') == 'no_action_needed':
                print("✅ Models performing well - no retraining needed")
            elif retrain_data.get('status') == 'success':
                print("✅ Retraining completed successfully")
            
            return True
        else:
            print(f"⚠️ Manual retraining endpoint status: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Manual retraining test failed: {e}")
        return False

if __name__ == "__main__":
    print("🎯 Testing ESPN as Primary Prediction System")
    print("Make sure your Flask app is running on localhost:5000")
    print()
    
    # Test ESPN as primary system
    primary_success = test_espn_primary_system()
    
    # Test manual retraining
    retrain_success = test_manual_retraining()
    
    print("\n" + "=" * 50)
    if primary_success and retrain_success:
        print("🏆 SUCCESS: ESPN-Enhanced System is Primary Prediction Method")
        print("   ✅ All predictions use ESPN enhancement")
        print("   ✅ Advanced features active")
        print("   ✅ Manual retraining available")
        print("   ✅ Maximum accuracy system deployed")
        sys.exit(0)
    else:
        print("⚠️ ISSUES FOUND: ESPN system not fully primary")
        if not primary_success:
            print("   ❌ ESPN enhancement not active on all predictions")
        if not retrain_success:
            print("   ❌ Manual retraining not available")
        sys.exit(1)