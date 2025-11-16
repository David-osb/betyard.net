#!/usr/bin/env python3
"""
Test script to verify the moneyline prediction endpoint works
"""
import threading
import time
import requests
import json
import sys
import os

# Add the ml-backend directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ml-backend'))

def test_endpoint():
    """Test the moneyline prediction endpoint"""
    try:
        # Import the app
        import app
        
        print("✅ App imported successfully")
        print(f"✅ Found {len(list(app.app.url_map.iter_rules()))} routes")
        
        # Check if our route is registered
        routes = [str(rule) for rule in app.app.url_map.iter_rules()]
        moneyline_routes = [r for r in routes if 'moneyline' in r]
        print(f"✅ Moneyline routes: {moneyline_routes}")
        
        # Start the server in a thread
        def run_server():
            app.app.run(host='127.0.0.1', port=5001, debug=False, use_reloader=False)
        
        server_thread = threading.Thread(target=run_server)
        server_thread.daemon = True
        server_thread.start()
        
        print("🚀 Server started, waiting 3 seconds...")
        time.sleep(3)
        
        # Test the endpoint
        url = "http://localhost:5001/api/moneyline/prediction"
        test_data = {
            "team1": "Kansas City Chiefs",
            "team2": "New York Giants"
        }
        
        print(f"📡 Testing {url}")
        print(f"📊 Data: {json.dumps(test_data, indent=2)}")
        
        response = requests.post(url, json=test_data, timeout=10)
        
        print(f"📈 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("🎉 SUCCESS! Response:")
            print(json.dumps(result, indent=2))
        else:
            print(f"❌ ERROR: {response.status_code}")
            print(f"📄 Response: {response.text}")
            
    except Exception as e:
        print(f"💥 ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_endpoint()