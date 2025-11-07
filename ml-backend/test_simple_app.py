#!/usr/bin/env python3
"""
Minimal test to verify our Flask routes work
"""
from flask import Flask, jsonify, request
from datetime import datetime
import logging

# Create Flask app
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

@app.route('/test/moneyline', methods=['GET'])
def test_moneyline():
    """Simple test endpoint to verify the code is running"""
    return jsonify({
        'message': 'Moneyline endpoint test successful!',
        'timestamp': datetime.now().isoformat(),
        'status': 'working'
    })

@app.route('/api/moneyline/prediction', methods=['POST'])
def predict_moneyline_xgboost():
    """Get XGBoost-powered moneyline predictions for a matchup"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        team1 = data.get('team1')
        team2 = data.get('team2')
        
        if not team1 or not team2:
            return jsonify({'error': 'Both team1 and team2 are required'}), 400
            
        logger.info(f"Generating XGBoost moneyline prediction for {team1} vs {team2}")
        
        # Simple mock prediction
        prediction = {
            'matchup': f"{team1} vs {team2}",
            'predictions': {
                'team1': {
                    'name': team1,
                    'win_probability': 60.0,
                    'american_odds': -150,
                    'strength_score': 0.6
                },
                'team2': {
                    'name': team2,
                    'win_probability': 40.0,
                    'american_odds': 125,
                    'strength_score': 0.4
                }
            },
            'confidence': 75.0,
            'model_info': {
                'type': 'XGBoost Ensemble',
                'positions': ['QB', 'RB', 'WR', 'TE'],
                'training_games': 27332
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(prediction)
        
    except Exception as e:
        logger.error(f"Error in XGBoost moneyline prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logger.info("🚀 Starting Test ML Backend...")
    logger.info("Available endpoints:")
    logger.info("  GET  /health - Health check")
    logger.info("  GET  /test/moneyline - Test moneyline endpoint")  
    logger.info("  POST /api/moneyline/prediction - XGBoost moneyline predictions")
    
    app.run(host='0.0.0.0', port=5001, debug=True)