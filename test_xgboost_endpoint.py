#!/usr/bin/env python3
"""
Quick test script for the XGBoost moneyline prediction endpoint
"""
import requests
import json

def test_xgboost_endpoint():
    url = "http://localhost:5001/api/moneyline/prediction"
    
    test_data = {
        "team1": "DAL",
        "team2": "NYG"
    }
    
    print(f"Testing XGBoost endpoint: {url}")
    print(f"Test data: {test_data}")
    print("-" * 50)
    
    try:
        response = requests.post(
            url, 
            headers={"Content-Type": "application/json"},
            json=test_data
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Success! XGBoost prediction received:")
            print(json.dumps(result, indent=2))
            
            # Print key insights
            print("\n🎯 Key Insights:")
            team1 = result['predictions']['team1']
            team2 = result['predictions']['team2']
            
            print(f"  {team1['name']}: {team1['win_probability']}% chance ({team1['american_odds']})")
            print(f"  {team2['name']}: {team2['win_probability']}% chance ({team2['american_odds']})")
            print(f"  Model Confidence: {result['confidence']}%")
            print(f"  Model Type: {result['model_info']['type']}")
            
        else:
            print("❌ Error:")
            print(f"Status: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Backend not running on http://localhost:5001")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_xgboost_endpoint()