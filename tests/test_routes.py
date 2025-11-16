#!/usr/bin/env python3
"""
Test script to check Flask routes and test our endpoint directly
"""
import requests
import json

def test_routes():
    """Test what routes are available"""
    print("🔍 Testing available endpoints...")
    
    test_endpoints = [
        "/health",
        "/model/info", 
        "/predict",
        "/api/moneyline/prediction"
    ]
    
    for endpoint in test_endpoints:
        url = f"http://localhost:5001{endpoint}"
        try:
            if endpoint == "/api/moneyline/prediction":
                response = requests.post(
                    url, 
                    json={"team1": "DAL", "team2": "NYG"},
                    timeout=5
                )
            else:
                response = requests.get(url, timeout=5)
            
            status = "✅" if response.status_code == 200 else f"❌ {response.status_code}"
            print(f"  {endpoint}: {status}")
            
        except requests.exceptions.ConnectionError:
            print(f"  {endpoint}: ❌ Connection Error")
        except Exception as e:
            print(f"  {endpoint}: ❌ {e}")

def test_xgboost_direct():
    """Test the XGBoost endpoint directly"""
    url = "http://localhost:5001/api/moneyline/prediction"
    
    test_data = {
        "team1": "DAL",
        "team2": "NYG"
    }
    
    print(f"\n🎯 Testing XGBoost endpoint: {url}")
    print(f"Data: {test_data}")
    
    try:
        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json=test_data,
            timeout=10
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ SUCCESS! Response:")
            print(json.dumps(result, indent=2))
        else:
            print("❌ ERROR Response:")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Exception: {e}")

if __name__ == "__main__":
    test_routes()
    test_xgboost_direct()