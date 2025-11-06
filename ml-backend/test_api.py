#!/usr/bin/env python3
"""
Test the API endpoint directly to see if enhanced QB predictions work
"""

import requests
import json
import time
import subprocess
import sys
import os

# Add the parent directory to the path so we can import app
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_qb_prediction():
    """Test QB prediction API with enhanced touchdown breakdown"""
    
    # Test payload for Lamar Jackson
    payload = {
        "player_name": "Lamar Jackson",
        "team_code": "BAL",
        "position": "QB"
    }
    
    print("🏈 Testing QB Prediction API...")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    try:
        # Make API request
        response = requests.post(
            "http://localhost:5001/predict",
            json=payload,
            timeout=10
        )
        
        print(f"\n📡 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n📊 Response Data:")
            print(json.dumps(data, indent=2))
            
            # Check if enhanced touchdown fields are present
            prediction = data.get('prediction', {})
            
            print(f"\n🎯 Touchdown Breakdown Analysis:")
            print(f"   Legacy Touchdowns: {prediction.get('touchdowns', 'NOT FOUND')}")
            print(f"   Passing Touchdowns: {prediction.get('passing_touchdowns', 'NOT FOUND')}")
            print(f"   Rushing Touchdowns: {prediction.get('rushing_touchdowns', 'NOT FOUND')}")
            
            if 'passing_touchdowns' in prediction and 'rushing_touchdowns' in prediction:
                print(f"\n✅ SUCCESS: Enhanced touchdown breakdown is working!")
                return True
            else:
                print(f"\n❌ ISSUE: Enhanced touchdown fields missing from response")
                return False
        else:
            print(f"\n❌ API Error: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"\n❌ Cannot connect to localhost:5001")
        print(f"Make sure the ML backend is running!")
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_qb_prediction()