#!/usr/bin/env python3
"""
Simple test server to verify Flask and models work
"""

from flask import Flask, jsonify
from flask_cors import CORS
import pickle

app = Flask(__name__)
CORS(app)

# Load one model to test
print("Loading QB model...")
qb_model = pickle.load(open('qb_model.pkl', 'rb'))
print("Model loaded successfully!")

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'message': 'Test server running with real XGBoost model'
    })

@app.route('/test')
def test():
    return jsonify({
        'message': 'Simple test endpoint working',
        'model_type': str(type(qb_model))
    })

if __name__ == '__main__':
    print("🚀 Starting simple test server...")
    print("📡 Running on port 5001")
    app.run(host='0.0.0.0', port=5001, debug=True)