#!/usr/bin/env python3
"""
Test client for the moneyline prediction endpoint
"""
import requests
import json

def test_moneyline_endpoint():
    """Test the moneyline prediction endpoint"""
    url = "http://localhost:5001/api/moneyline/prediction"
    
    test_data = {
        "team1": "Kansas City Chiefs",
        "team2": "New York Giants"
    }
    
    try:
        print(f"Testing endpoint: {url}")
        print(f"Test data: {json.dumps(test_data, indent=2)}")
        
        response = requests.post(url, json=test_data, timeout=10)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            print("SUCCESS! Response:")
            print(json.dumps(result, indent=2))
        else:
            print(f"ERROR: {response.status_code}")
            print(f"Response text: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to server. Is it running on port 5001?")
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    test_moneyline_endpoint()