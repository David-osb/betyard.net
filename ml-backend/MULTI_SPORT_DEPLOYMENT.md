# Multi-Sport ML Model Deployment Summary

## ✅ COMPLETED: NBA, NHL, MLB Prediction Models & API Endpoints

### Data Collection Results
- **NBA**: 503 players → 472 predictions (1.5MB data)
- **NHL**: 752 players → 650 skaters + 67 goalies (1.9MB data)
- **MLB**: 1,024 players → 441 hitters + 422 pitchers (8.2MB data)
- **Total**: 2,279 players with 2024-25 historical data

---

## Training Scripts Created

### 1. NBA Model (`train_nba_models.py`)
**Predictions Generated**: 472 players with ≥3 games played

**Props**:
- **Points O/U**: Dynamic line based on average (nearest 0.5)
- **Rebounds O/U**: Dynamic line
- **Assists O/U**: Dynamic line
- **3-Pointers Made O/U**: Dynamic line

**Recommendation Logic**:
- **OVER**: >55% probability
- **UNDER**: <45% probability
- **NO BET**: 45-55% (coin flip)

**Output**: `nba_prop_predictions.json`

---

### 2. NHL Model (`train_nhl_models.py`)
**Predictions Generated**: 650 skaters + 67 goalies

**Skater Props**:
- **Anytime Goal**: Probability + American odds (+150, +200, etc.)
  - Recommendation: BET if >25%
- **Assists O/U 0.5**: Over/under percentages
- **Shots O/U**: Dynamic line

**Goalie Props**:
- **Saves O/U**: Dynamic line based on average
- **Goals Against Under 2.5**: Probability percentage

**Output**: `nhl_prop_predictions.json` (separate arrays for skaters/goalies)

---

### 3. MLB Model (`train_mlb_models.py`)
**Predictions Generated**: 441 hitters + 422 pitchers

**Hitter Props**:
- **Hits O/U 1.5** (2+ hits): Over/under probabilities
- **Anytime Home Run**: Probability + odds
  - Recommendation: BET if >15%
- **RBIs O/U 0.5** (1+ RBI): Over/under probabilities

**Pitcher Props**:
- **Strikeouts O/U**: Dynamic line (4.5, 5.5, 6.5, 7.5)
- **Hits Allowed Under 6.5**: Probability
- **Earned Runs Under 3.5**: Probability

**Output**: `mlb_prop_predictions.json` (separate arrays for hitters/pitchers)

---

## Backend API Endpoints

### Flask Server Running
**URL**: `http://127.0.0.1:10000`

### Endpoints Added to `app.py`

#### 1. NBA Endpoint
```
GET /players/nba/team/<team_code>
```

**Example**: `/players/nba/team/LAL` (Lakers)

**Response**:
```json
{
  "success": true,
  "sport": "nba",
  "team": "LAL",
  "players": [...],  // Top 10 scorers
  "count": 14
}
```

**Player Object**:
```json
{
  "player_name": "Anthony Davis",
  "position": "F-C",
  "team": "LAL",
  "games_played": 11,
  "props": {
    "points": {
      "average": 28.3,
      "line": 29.0,
      "over_probability": 45.5,
      "recommendation": "NO BET"
    },
    "rebounds": {...},
    "assists": {...},
    "threes_made": {...}
  }
}
```

---

#### 2. NHL Endpoint
```
GET /players/nhl/team/<team_code>
```

**Example**: `/players/nhl/team/TOR` (Maple Leafs)

**Response**:
```json
{
  "success": true,
  "sport": "nhl",
  "team": "TOR",
  "skaters": [...],   // Top 10 goal scorers
  "goalies": [...],   // Top 2 goalies
  "total_skaters": 18,
  "total_goalies": 2
}
```

**Skater Object**:
```json
{
  "player_name": "Auston Matthews",
  "position": "C",
  "props": {
    "anytime_goal": {
      "probability": 42.1,
      "odds": 137,
      "recommendation": "BET"
    },
    "assists": {...},
    "shots": {...}
  }
}
```

**Goalie Object**:
```json
{
  "player_name": "Joseph Woll",
  "position": "G",
  "props": {
    "saves": {
      "average": 26.5,
      "line": 27.0,
      "over_probability": 55.6,
      "recommendation": "OVER"
    },
    "goals_against": {
      "under_2_5_probability": 33.3
    }
  }
}
```

---

#### 3. MLB Endpoint
```
GET /players/mlb/team/<team_code>
```

**Example**: `/players/mlb/team/NYY` (Yankees)

**Response**:
```json
{
  "success": true,
  "sport": "mlb",
  "team": "NYY",
  "hitters": [...],    // Top 10 hitters
  "pitchers": [...],   // Top 5 pitchers
  "total_hitters": 12,
  "total_pitchers": 8
}
```

**Hitter Object**:
```json
{
  "player_name": "Aaron Judge",
  "position": "RF",
  "props": {
    "hits": {
      "average": 1.17,
      "line": 1.5,
      "over_probability": 33.3,
      "recommendation": "UNDER"
    },
    "home_run": {
      "probability": 28.5,
      "odds": 251,
      "recommendation": "BET"
    },
    "rbis": {...}
  }
}
```

**Pitcher Object**:
```json
{
  "player_name": "Gerrit Cole",
  "position": "SP",
  "props": {
    "strikeouts": {
      "average": 7.2,
      "line": 7.5,
      "over_probability": 48.3,
      "recommendation": "NO BET"
    },
    "hits_allowed": {...},
    "earned_runs": {...}
  }
}
```

---

## Testing Results

### ✅ NBA Endpoint Test
```bash
curl http://127.0.0.1:10000/players/nba/team/LAL
```
**Result**: 14 Lakers players with points/rebounds/assists/3PM props

### ✅ NHL Endpoint Test
```bash
curl http://127.0.0.1:10000/players/nhl/team/TOR
```
**Result**: 18 skaters + 2 goalies with goal/assist/save props

### ✅ MLB Endpoint Test
```bash
curl http://127.0.0.1:10000/players/mlb/team/NYY
```
**Result**: 12 hitters + 8 pitchers with hits/HR/K props

---

## Files Created/Modified

### New Training Scripts
1. `/ml-backend/train_nba_models.py` (157 lines)
2. `/ml-backend/train_nhl_models.py` (177 lines)
3. `/ml-backend/train_mlb_models.py` (152 lines)

### Prediction Files Generated
1. `/ml-backend/nba_prop_predictions.json` (472 players)
2. `/ml-backend/nhl_prop_predictions.json` (717 players)
3. `/ml-backend/mlb_prop_predictions.json` (863 players)

### Backend Updates
- `/ml-backend/app.py` - Added 3 new endpoints (lines 360-530)

---

## Key Technical Details

### Model Approach
- **NFL**: XGBoost with synthetic data (existing)
- **NBA/NHL/MLB**: Probability-based using actual historical game logs
- **No ML training needed**: Direct statistical analysis of past performance

### Probability Calculations
```python
# Generic prop calculation
over_count = sum(1 for game in gamelog if game[stat] > line)
probability = (over_count / total_games) * 100
```

### American Odds Conversion (NHL/MLB)
```python
if probability >= 50:
    odds = -100 * (probability / (100 - probability))  # Favorite
else:
    odds = 100 * ((100 - probability) / probability)   # Underdog
```

### Dynamic Line Calculation
```python
# Round to nearest 0.5 for points/rebounds/assists
line = round(average * 2) / 2
```

---

## Next Steps (Optional Future Work)

### Frontend Integration
1. Create sport selector UI (NFL/NBA/NHL/MLB tabs)
2. Add team selection per sport
3. Display prop cards with recommendations
4. Color-code by confidence (OVER=green, UNDER=red, NO BET=gray)

### Enhanced Features
1. **Live odds comparison**: Integrate FanDuel/DraftKings API
2. **Expected value (EV) calculation**: Compare our probabilities to sportsbook odds
3. **Player injury status**: Filter out injured players
4. **Home/away splits**: Adjust probabilities based on game location
5. **Opponent analysis**: Factor in defensive rankings

### MLS Alternative
- ESPN MLS API is restricted
- Consider alternative data sources:
  - SoccerStats API
  - The-Odds-API (limited soccer coverage)
  - Manual web scraping (BeautifulSoup)

---

## Dependencies Installed
```bash
pip3 install flask flask-cors xgboost scikit-learn numpy
```

## Running the Backend
```bash
cd /workspaces/betyard.net/ml-backend
python3 app.py
# Server runs on http://127.0.0.1:10000
```

---

## Summary
✅ **3 sports fully operational** (NBA, NHL, MLB)  
✅ **2,279 players** with historical data  
✅ **2,052 predictions** generated  
✅ **3 API endpoints** tested and working  
✅ **Ready for frontend integration**
