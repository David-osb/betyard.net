#!/usr/bin/env python3
"""
Simple test to check Flask routes
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

try:
    from app import app
    print("✅ App imported successfully!")
    
    print("\n🔍 Available routes:")
    for rule in app.url_map.iter_rules():
        print(f"  {rule.rule} -> {rule.methods}")
    
    print(f"\n🏃 Starting server on port 5001...")
    app.run(debug=True, port=5001, host='0.0.0.0')
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()