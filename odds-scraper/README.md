# NFL Odds Scraper - DraftKings

Live odds scraper for NFL player props using DraftKings public API.

## Features

- ✅ **DraftKings**: Fully automated, unlimited free requests
- Real-time player props (passing yards, rushing yards, TDs, etc.)
- No authentication required
- Clean JSON responses
- Production-ready

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```python
from adaptive_odds_scraper import AdaptiveOddsScraper

scraper = AdaptiveOddsScraper(use_smart_requester=False)

# Get player props
odds = scraper.get_player_props('Josh Allen', 'passing_yards')
print(odds)
# {'book': 'draftkings', 'line': 249.5, 'over': -110, 'under': -110}
```

## Supported Stats
- `passing_yards`
- `passing_tds`
- `passing_completions`
- `passing_attempts`
- `rushing_yards`
- `rushing_attempts`
- `receiving_yards`
- `receptions`
- `interceptions`

## Player IDs

See `player_ids.py` for the full list of supported players (40+ NFL stars).

## How It Works

Hits DraftKings public API:
```
https://sportsbook-nash.draftkings.com/api/team/markets/dkusoh/v2/player/{player_id}/markets
```

No authentication, no HTML scraping, just clean JSON responses.

## Integration

This scraper is integrated into the main Flask backend at `ml-backend/app.py`:
- Endpoint: `/api/live-odds/<player_name>`
- Production: `betyard-ml-backend.onrender.com`
- Auto-refresh: Every 60 seconds in frontend

## Why DraftKings Only?

- **Free & unlimited** - No API costs or request limits
- **No authentication** - Public API, no login required
- **Reliable** - No geolocation blocks or CAPTCHAs
- **Real-time** - Always up-to-date odds
- **Production tested** - Running live since November 2025

## Example Response

```json
{
  "player": "Josh Allen",
  "stat": "passing_yards",
  "books": {
    "draftkings": {
      "line": 249.5,
      "over": -110,
      "under": -110,
      "timestamp": "2025-11-23T18:30:00Z"
    }
  }
}
```
