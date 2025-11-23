# FanDuel Cookie Authentication Setup

## Step 1: Extract Cookies (One-Time Setup)

Run the cookie extractor script:

```bash
python extract_fanduel_cookies.py
```

**What it does:**
1. Opens Chrome to FanDuel
2. You log in manually (with your username/password)
3. Script saves your session cookies
4. Tests that cookies work

**Time required:** 2 minutes (one time)

## Step 2: Use FanDuel Scraper

Once cookies are saved, the scraper works automatically:

```python
from fanduel_scraper import FanDuelScraper

# Automatically loads saved cookies
scraper = FanDuelScraper(use_cookies=True)

# Get Josh Allen passing yards
props = scraper.get_player_props('Josh Allen', 'passing_yards')
print(props)
```

## How Long Do Cookies Last?

- **Typical:** 30-90 days
- **When they expire:** Just run `extract_fanduel_cookies.py` again
- **How you know:** Scraper will return empty data or errors

## Files Created

- `fanduel_cookies.pkl` - Binary cookie file (used by scraper)
- `fanduel_cookies.json` - Readable JSON (for inspection)

**Important:** Don't share these files - they're your login session!

## Add to .gitignore

```
fanduel_cookies.pkl
fanduel_cookies.json
```

## Security Notes

✅ **Safe to use** - You're using YOUR account  
✅ **Won't get banned** - Just reading public data  
✅ **Cookies are local** - Never sent to third parties  
⚠️ **Keep private** - Don't commit cookies to git  

## Troubleshooting

**"No saved cookies found"**
- Run `extract_fanduel_cookies.py` first

**"API returned 403"**
- Cookies expired - run extractor again

**"No events found"**
- Check if you're logged in to FanDuel
- Refresh cookies

## Refresh Schedule

Run `extract_fanduel_cookies.py` when:
- First time setup
- After 30-60 days
- If scraper stops working
- After changing FanDuel password
