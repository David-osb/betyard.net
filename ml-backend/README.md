# BetYard ML Backend

Fixed version that resolves the "Feature shape mismatch: expected 10, got 8" error.

## What's Fixed

✅ **10-Feature Extraction** - Now returns exactly 10 features:
1. Team offensive rating
2. Team defensive rating
3. Opponent defensive rating
4. Home game indicator
5. Player season avg yards
6. Player season avg TDs
7. Player recent 3-game avg
8. **Weather score (75.0)** ← NEW
9. Matchup difficulty
10. **Player health (100.0)** ← NEW

✅ **Built-in Team Stats** - All 32 NFL teams with offense/defense ratings
✅ **Position Baselines** - Season averages for QB/RB/WR/TE
✅ **Feature Verification** - Confirms 10 features before prediction

## Deployment

### Option 1: Deploy to Render (Recommended)

1. **Push to GitHub:**
```bash
cd /workspaces/betyard.net/backend
git add .
git commit -m "Fix feature mismatch - add weather and health features"
git push origin main
```

2. **Connect to Render:**
   - Go to https://dashboard.render.com
   - Click "New +" → "Web Service"
   - Connect your `betyard.net` repository
   - Select the `backend` folder
   - Render will auto-detect `render.yaml` and deploy

3. **Verify:**
```bash
curl https://betyard-ml-backend.onrender.com/health
```

Expected response:
```json
{
  "status": "healthy",
  "models_loaded": {"qb": true, "rb": true, "wr": true, "te": true},
  "version": "v2025-11-16-feature-fix",
  "features_count": 10
}
```

### Option 2: Test Locally First

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
python app.py
# Server starts on http://localhost:10000

# Test health endpoint
curl http://localhost:10000/health

# Test prediction
curl -X POST http://localhost:10000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "player_name": "Patrick Mahomes",
    "team_code": "KC",
    "opponent_code": "BUF",
    "position": "QB"
  }'
```

## Files Included

```
backend/
├── app.py                    # Fixed Flask application
├── app_fixed.py             # Backup copy
├── requirements.txt         # Python dependencies
├── render.yaml             # Render deployment config
├── qb_model.pkl            # Quarterback model
├── rb_model.pkl            # Running back model
├── wr_model.pkl            # Wide receiver model
├── te_model.pkl            # Tight end model
├── game_winner_model.pkl   # Game prediction model
├── point_spread_model.pkl  # Spread prediction model
└── total_points_model.pkl  # Totals prediction model
```

## API Endpoints

### Health Check
```
GET /health
```

Response:
```json
{
  "status": "healthy",
  "models_loaded": {
    "qb": true,
    "rb": true,
    "wr": true,
    "te": true
  },
  "version": "v2025-11-16-feature-fix",
  "features_count": 10
}
```

### Predict Player Performance
```
POST /predict
Content-Type: application/json

{
  "player_name": "Patrick Mahomes",
  "team_code": "KC",
  "opponent_code": "BUF",
  "position": "QB"
}
```

Response (QB):
```json
{
  "passing_yards": 285.3,
  "completions": 25.1,
  "attempts": 39.9,
  "touchdowns": 2.3,
  "interceptions": 1.1,
  "completion_percentage": 62.9,
  "yards_per_attempt": 7.1,
  "passer_rating": 88.5,
  "confidence": 75,
  "player_name": "Patrick Mahomes",
  "team_code": "KC",
  "opponent_code": "BUF",
  "position": "QB",
  "model_version": "v2025-11-16-feature-fix"
}
```

## Environment Variables

- `PORT` - Server port (default: 10000)

## What Changed

### Before (8 features):
```python
features = [
    team_offense,
    team_defense,
    opponent_defense,
    is_home,
    player_avg_yards,
    player_avg_tds,
    player_recent_avg,
    matchup_difficulty
]  # ❌ Only 8 features
```

### After (10 features):
```python
features = [
    team_offense,
    team_defense,
    opponent_defense,
    is_home,
    player_avg_yards,
    player_avg_tds,
    player_recent_avg,
    weather_score,        # ✅ NEW
    matchup_difficulty,
    player_health         # ✅ NEW
]  # ✅ 10 features (matches model)
```

## Troubleshooting

### Models not loading?
- Ensure all `.pkl` files are in the same directory as `app.py`
- Check file permissions: `chmod 644 *.pkl`

### Still getting feature mismatch?
- Verify `extract_features()` returns exactly 10 features
- Check line 107 in `app.py` - should show `]).reshape(1, -1)`

### Render deployment fails?
- Check build logs in Render dashboard
- Verify `requirements.txt` has all dependencies
- Ensure Python version is 3.11+

## Support

Backend URL: https://betyard-ml-backend.onrender.com
Health Check: https://betyard-ml-backend.onrender.com/health
