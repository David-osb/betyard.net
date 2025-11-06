#!/usr/bin/env python3
"""
Test the enhanced QB prediction by simulating the API response format
"""

# Add the parent directory to the path so we can import app
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import app
import json
from datetime import datetime

def simulate_api_response():
    """Simulate the API response format to see if enhanced fields work"""
    
    print("🏈 Simulating API Response for Enhanced QB Prediction...")
    
    try:
        # Get the ML model
        ml_model = app.ml_model
        
        # Test with Lamar Jackson (mobile QB)
        print(f"\n🚀 Testing Lamar Jackson (mobile QB)...")
        prediction = ml_model.predict_qb_performance("Lamar Jackson", "BAL", "CIN")
        
        # Build response with the same logic as the API endpoint
        response_prediction = {
            'position': prediction.position,
            'confidence': float(prediction.confidence),
            'model_accuracy': float(prediction.model_accuracy),
            'prediction_likelihood': float(prediction.prediction_likelihood),
            'fantasy_points': float(prediction.fantasy_points) if prediction.fantasy_points else None
        }
        
        # Add position-specific stats
        if prediction.passing_yards is not None:
            response_prediction['passing_yards'] = float(prediction.passing_yards)
            response_prediction['completions'] = float(prediction.completions)
            response_prediction['attempts'] = float(prediction.attempts)
        
        if prediction.rushing_yards is not None:
            response_prediction['rushing_yards'] = float(prediction.rushing_yards)
            response_prediction['rushing_attempts'] = float(prediction.rushing_attempts)
        
        if prediction.receiving_yards is not None:
            response_prediction['receiving_yards'] = float(prediction.receiving_yards)
        if prediction.receptions is not None:
            response_prediction['receptions'] = float(prediction.receptions)
        if prediction.targets is not None:
            response_prediction['targets'] = float(prediction.targets)
        
        if prediction.touchdowns is not None:
            response_prediction['touchdowns'] = float(prediction.touchdowns)
        
        # Enhanced touchdown breakdown for QB predictions
        print(f"\n🔍 Checking touchdown field values:")
        print(f"   prediction.passing_touchdowns: {prediction.passing_touchdowns} (type: {type(prediction.passing_touchdowns)})")
        print(f"   prediction.rushing_touchdowns: {prediction.rushing_touchdowns} (type: {type(prediction.rushing_touchdowns)})")
        print(f"   Is passing_touchdowns None? {prediction.passing_touchdowns is None}")
        print(f"   Is rushing_touchdowns None? {prediction.rushing_touchdowns is None}")
        
        if prediction.passing_touchdowns is not None:
            response_prediction['passing_touchdowns'] = float(prediction.passing_touchdowns)
            print(f"   ✅ Added passing_touchdowns = {prediction.passing_touchdowns}")
        else:
            print(f"   ❌ Skipped passing_touchdowns (is None)")
        
        if prediction.rushing_touchdowns is not None:
            response_prediction['rushing_touchdowns'] = float(prediction.rushing_touchdowns)
            print(f"   ✅ Added rushing_touchdowns = {prediction.rushing_touchdowns}")
        else:
            print(f"   ❌ Skipped rushing_touchdowns (is None)")
        
        if prediction.receiving_touchdowns is not None:
            response_prediction['receiving_touchdowns'] = float(prediction.receiving_touchdowns)
        
        if prediction.interceptions is not None:
            response_prediction['interceptions'] = float(prediction.interceptions)
        
        if prediction.qb_rating is not None:
            response_prediction['qb_rating'] = float(prediction.qb_rating)
        
        # Build full API response
        api_response = {
            'success': True,
            'prediction': response_prediction,
            'metadata': {
                'player_name': "Lamar Jackson",
                'team': "BAL",
                'opponent': "CIN",
                'position': "QB",
                'model_version': '2.0',
                'timestamp': datetime.now().isoformat()
            }
        }
        
        print(f"\n📊 Simulated API Response:")
        print(json.dumps(api_response, indent=2))
        
        # Check if enhanced fields are present
        pred = api_response['prediction']
        print(f"\n🎯 Touchdown Breakdown Check:")
        print(f"   Legacy Touchdowns: {pred.get('touchdowns', 'NOT FOUND')}")
        print(f"   Passing Touchdowns: {pred.get('passing_touchdowns', 'NOT FOUND')}")
        print(f"   Rushing Touchdowns: {pred.get('rushing_touchdowns', 'NOT FOUND')}")
        
        if 'passing_touchdowns' in pred and 'rushing_touchdowns' in pred:
            print(f"\n✅ SUCCESS: Enhanced touchdown breakdown is working!")
        else:
            print(f"\n❌ ISSUE: Enhanced touchdown fields missing")
        
    except Exception as e:
        print(f"❌ Error simulating API response: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    simulate_api_response()