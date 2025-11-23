# Sports Odds Scraper

Automated scrapers for DraftKings and FanDuel sportsbook odds.

## Features
- ✅ **DraftKings** - 100% automated, no API key required
- ✅ **FanDuel** - Cookie-based authentication (one-time 2-min setup)
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

## FanDuel Setup (2-Minute One-Time Setup)

### 1. Extract cookies (run once):
```bash
python extract_fanduel_cookies.py
```
- Opens Chrome, you log in to FanDuel
- Saves your cookies automatically
- Cookies last 30-90 days

### 2. Use FanDuel scraper:
```python
from fanduel_scraper import FanDuelScraper

scraper = FanDuelScraper(use_cookies=True)
props = scraper.get_player_props('Josh Allen', 'passing_yards')
```

**Full guide:** See [FANDUEL_SETUP.md](FANDUEL_SETUP.md)

## Multi-Book Comparison

Get best odds across both books:

```python
from multibook_scraper import MultiBookScraper

scraper = MultiBookScraper()
odds = scraper.get_best_odds('Josh Allen', 'passing_yards')

print(f"Best Over: {odds['best_over']['book']} @ {odds['best_over']['odds']}")
print(f"Best Under: {odds['best_under']['book']} @ {odds['best_under']['odds']}")
print(f"Edge: {odds['edge']}")
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
