# DraftKings Odds Scraper

Automated scraper for DraftKings sportsbook odds. No API key required.

## Features
- 100% automated - no manual work needed
- Real-time odds from DraftKings public API
- Supports NFL player props (passing yards, rushing yards, TDs, etc.)
- No captchas or authentication required

## Quick Start

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
