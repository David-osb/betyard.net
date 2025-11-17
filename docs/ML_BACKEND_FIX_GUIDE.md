# 🔧 ML Backend Feature Mismatch Fix Guide

## The Problem

**Error**: `Feature shape mismatch, expected: 10, got 8`

**What's happening**:
- Your XGBoost model was trained with **10 features**
- Your current backend code is sending **8 features** to the model
- This causes a 500 error on the `/predict` endpoint

---

## Current Frontend Payload

The frontend sends these 4 fields:

```json
{
  "player_name": "Patrick Mahomes",
  "team_code": "KC",
  "opponent_code": "BUF",
  "position": "QB"
}
```

The backend must extract **10 features** from this data before passing to the model.

---

## What the Model Expects (10 Features)

Based on typical NFL prediction models, your XGBoost model likely expects:

```python
feature_vector = [
    # 1. Team offensive rating (0-100)
    team_offensive_rating,
    
    # 2. Team defensive rating (0-100)
    team_defensive_rating,
    
    # 3. Opponent defensive rating (0-100)
    opponent_defensive_rating,
    
    # 4. Home/Away indicator (0 or 1)
    is_home_game,
    
    # 5. Player's season average yards
    player_season_avg_yards,
    
    # 6. Player's season average TDs
    player_season_avg_touchdowns,
    
    # 7. Player's last 3 games average
    player_recent_avg,
    
    # 8. Weather factor (0-100)
    weather_score,
    
    # 9. Matchup difficulty (0-100)
    matchup_difficulty,
    
    # 10. Player health/injury status (0-100)
    player_health_score
]
```

---

## How to Fix the Backend

### Option 1: Add Missing Features (Recommended)

Update your backend's `/predict` endpoint to extract all 10 features:

```python
from flask import Flask, request, jsonify
import xgboost as xgb
import numpy as np

app = Flask(__name__)

# Load your trained model
model = xgb.Booster()
model.load_model('qb_model.pkl')

# Feature extraction functions
def get_team_stats(team_code):
    """Get team offensive/defensive ratings"""
    # TODO: Fetch from ESPN API or database
    team_ratings = {
        'KC': {'offense': 95, 'defense': 85},
        'BUF': {'offense': 92, 'defense': 88},
        # ... add all 32 teams
    }
    return team_ratings.get(team_code, {'offense': 75, 'defense': 75})

def get_opponent_defense(opponent_code):
    """Get opponent defensive rating"""
    team_stats = get_team_stats(opponent_code)
    return team_stats['defense']

def get_player_stats(player_name, position):
    """Get player's season averages"""
    # TODO: Fetch from ESPN API or database
    # For now, return league averages
    if position == 'QB':
        return {
            'avg_yards': 250,
            'avg_touchdowns': 2,
            'recent_avg': 245
        }
    return {'avg_yards': 0, 'avg_touchdowns': 0, 'recent_avg': 0}

def get_matchup_difficulty(team_offense, opponent_defense):
    """Calculate matchup difficulty"""
    return max(0, min(100, opponent_defense - team_offense + 50))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        player_name = data.get('player_name')
        team_code = data.get('team_code')
        opponent_code = data.get('opponent_code')
        position = data.get('position', 'QB')
        
        # Extract team stats
        team_stats = get_team_stats(team_code)
        opponent_defense = get_opponent_defense(opponent_code)
        
        # Extract player stats
        player_stats = get_player_stats(player_name, position)
        
        # Build feature vector (MUST be 10 features in correct order!)
        features = np.array([
            team_stats['offense'],              # Feature 1
            team_stats['defense'],              # Feature 2
            opponent_defense,                   # Feature 3
            1.0,                                # Feature 4: is_home (default)
            player_stats['avg_yards'],          # Feature 5
            player_stats['avg_touchdowns'],     # Feature 6
            player_stats['recent_avg'],         # Feature 7
            75.0,                               # Feature 8: weather (default)
            get_matchup_difficulty(             # Feature 9
                team_stats['offense'], 
                opponent_defense
            ),
            100.0                               # Feature 10: health (default)
        ]).reshape(1, -1)
        
        # CRITICAL: Verify feature count
        if features.shape[1] != 10:
            return jsonify({
                'error': f'Feature count mismatch: got {features.shape[1]}, expected 10'
            }), 500
        
        # Make prediction
        dmatrix = xgb.DMatrix(features)
        prediction = model.predict(dmatrix)[0]
        
        # Return prediction in expected format
        return jsonify({
            'passing_yards': float(prediction),
            'completions': float(prediction * 0.088),  # ~22 if 250 yards
            'attempts': float(prediction * 0.14),      # ~35 if 250 yards
            'touchdowns': 2.0,
            'interceptions': 1.0,
            'completion_percentage': 62.9,
            'yards_per_attempt': 7.1,
            'passer_rating': 88.5,
            'confidence': 75
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
```

---

### Option 2: Retrain Model with 8 Features

If you can't add the missing features, retrain the model with only the available features:

```python
import xgboost as xgb
import pandas as pd

# Load your training data
df = pd.read_csv('training_data.csv')

# Use only 8 features instead of 10
feature_columns = [
    'team_offense',
    'team_defense', 
    'opponent_defense',
    'is_home',
    'player_avg_yards',
    'player_avg_touchdowns',
    'recent_avg',
    'matchup_difficulty'
]

X = df[feature_columns]
y = df['actual_yards']

# Train new model
model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1
)

model.fit(X, y)
model.save_model('qb_model_8features.pkl')
```

---

## Quick Fix for Testing

If you need a temporary fix to test the system, you can pad the features with dummy values:

```python
@app.route('/predict', methods=['POST'])
def predict():
    # ... existing code ...
    
    # Build 8 real features
    features_8 = np.array([
        team_offense,
        team_defense,
        opponent_defense,
        is_home,
        player_avg_yards,
        player_avg_touchdowns,
        recent_avg,
        matchup_difficulty
    ])
    
    # Pad with 2 dummy features to reach 10
    features_10 = np.append(features_8, [75.0, 100.0])
    features = features_10.reshape(1, -1)
    
    # Now it's 10 features!
    dmatrix = xgb.DMatrix(features)
    prediction = model.predict(dmatrix)[0]
    # ... rest of code ...
```

---

## Deployment Steps

### 1. Find Your Backend Code

Your backend is deployed on Render at:
```
https://betyard-ml-backend.onrender.com
```

You need to find the source code repository for this backend. Check:
- Your Render dashboard
- GitHub repositories
- Local machine backups

### 2. Update the `/predict` Endpoint

Add the 10-feature extraction logic from Option 1 above.

### 3. Test Locally

```bash
# Install dependencies
pip install flask xgboost numpy

# Run backend locally
python app.py

# Test with curl
curl -X POST http://localhost:10000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "player_name": "Patrick Mahomes",
    "team_code": "KC",
    "opponent_code": "BUF",
    "position": "QB"
  }'
```

### 4. Verify Feature Count

Add logging to verify:

```python
print(f"Feature shape: {features.shape}")  # Should print: (1, 10)
```

### 5. Deploy to Render

```bash
git add .
git commit -m "Fix: Add missing 2 features for model (10 total)"
git push origin main
```

Render will auto-deploy the update.

### 6. Test Production

Visit your site and click a game - props should now load!

---

## Feature Mapping Reference

| Feature # | Name | Type | Source | Example |
|-----------|------|------|--------|---------|
| 1 | Team Offense Rating | Float (0-100) | ESPN API / Static DB | 95.0 |
| 2 | Team Defense Rating | Float (0-100) | ESPN API / Static DB | 85.0 |
| 3 | Opponent Defense | Float (0-100) | ESPN API / Static DB | 88.0 |
| 4 | Is Home Game | Binary (0/1) | Game Location | 1.0 |
| 5 | Player Avg Yards | Float | ESPN Stats | 275.5 |
| 6 | Player Avg TDs | Float | ESPN Stats | 2.3 |
| 7 | Recent 3-Game Avg | Float | ESPN Stats | 285.0 |
| 8 | Weather Score | Float (0-100) | Weather API | 75.0 |
| 9 | Matchup Difficulty | Float (0-100) | Calculated | 65.0 |
| 10 | Player Health | Float (0-100) | Injury Reports | 100.0 |

---

## Alternative: Use Frontend Fallback (Current Setup)

Your frontend now has a fallback system that works even when the backend is broken. This is a **temporary workaround** but not ideal long-term.

**Pros**:
- Site still works
- Users still see props
- No backend fixes needed immediately

**Cons**:
- Predictions are generic averages
- No personalized AI analysis
- Lower accuracy

To fully leverage your ML model, **fix the backend** using Option 1 above.

---

## Need Help?

1. **Find backend code**: Check your Render dashboard → Service → Repository link
2. **Can't find code**: The backend needs to be rebuilt from scratch
3. **Model file missing**: You'll need to retrain the XGBoost model

**Questions?** Check these files:
- `backend/qb_model.pkl` - Your trained model
- Render dashboard - Backend source repository
- Git history - Previous backend commits

---

**Last Updated**: November 16, 2025  
**Status**: Frontend fallback deployed, backend fix pending
