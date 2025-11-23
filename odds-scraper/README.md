# Sports Odds Scraper

Automated scrapers for DraftKings and FanDuel sportsbook odds.

## Features
- ✅ **DraftKings** - 100% automated, no API key required
- ⚠️ **FanDuel** - API endpoints identified but requires browser session
- Real-time odds from public APIs
- Supports NFL player props (passing yards, rushing yards, TDs, etc.)
- Multi-book comparison for line shopping

## Quick Start - DraftKings (WORKING)

```python
from adaptive_odds_scraper import AdaptiveOddsScraper

# Initialize scraper
scraper = AdaptiveOddsScraper(use_smart_requester=False)

# Get Josh Allen passing yards odds
props = scraper.get_player_props('Josh Allen', 'passing_yards')

print(f"Line: {props['books']['draftkings']['line']}")
print(f"Over: {props['books']['draftkings']['over']}")
print(f"Under: {props['books']['draftkings']['under']}")
```

## FanDuel Integration (IN PROGRESS)

FanDuel uses these API endpoints:
- `/api/content-managed-page` - Get NFL events/games
- `/api/livedata` - Real-time game data
- `/api/getMarketPrices` - Current odds (key endpoint)

**Status**: Endpoints require browser cookies/session tokens. Working on authentication bypass.

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

## Files
- `adaptive_odds_scraper.py` - Main scraper class
- `player_ids.py` - DraftKings player ID database (40+ NFL players)
- `requirements.txt` - Python dependencies

## How It Works
Hits DraftKings public API at:
```
https://sportsbook-nash.draftkings.com/api/team/markets/dkusoh/v2/player/{player_id}/markets
```

No authentication, no scraping HTML, just clean JSON responses.

## Integration
This scraper is used by the BetYard ML backend to provide live odds in the app.
