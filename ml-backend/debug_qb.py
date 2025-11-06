#!/usr/bin/env python3
"""
Debug QB prediction to see if the enhanced touchdown breakdown is working
"""

# Add the parent directory to the path so we can import app
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import app
import logging

# Set logging level to see debug info
logging.basicConfig(level=logging.INFO)

# Test QB prediction directly
try:
    print("🏈 Testing QB prediction with enhanced touchdown breakdown...")
    
    # Initialize the ML model (this should already be done when importing app)
    ml_model = app.ml_model
    
    # Test with Lamar Jackson (should have high mobility factor)
    print(f"\n🚀 Testing Lamar Jackson (mobile QB)...")
    prediction = ml_model.predict_qb_performance("Lamar Jackson", "BAL", "CIN")
    
    print(f"\n📊 Results for Lamar Jackson:")
    print(f"   Position: {prediction.position}")
    print(f"   Legacy Touchdowns: {prediction.touchdowns}")
    print(f"   Passing Touchdowns: {prediction.passing_touchdowns}")
    print(f"   Rushing Touchdowns: {prediction.rushing_touchdowns}")
    print(f"   Passing Yards: {prediction.passing_yards}")
    print(f"   QB Rating: {prediction.qb_rating}")
    
    # Test with a pocket passer
    print(f"\n🚀 Testing Aaron Rodgers (pocket passer)...")
    prediction2 = ml_model.predict_qb_performance("Aaron Rodgers", "NYJ", "BUF")
    
    print(f"\n📊 Results for Aaron Rodgers:")
    print(f"   Position: {prediction2.position}")
    print(f"   Legacy Touchdowns: {prediction2.touchdowns}")
    print(f"   Passing Touchdowns: {prediction2.passing_touchdowns}")
    print(f"   Rushing Touchdowns: {prediction2.rushing_touchdowns}")
    print(f"   Passing Yards: {prediction2.passing_yards}")
    print(f"   QB Rating: {prediction2.qb_rating}")
    
    print(f"\n✅ QB prediction testing complete!")
    
except Exception as e:
    print(f"❌ Error testing QB prediction: {e}")
    import traceback
    traceback.print_exc()