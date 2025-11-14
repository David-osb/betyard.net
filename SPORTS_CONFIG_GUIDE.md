# Universal Sports Configuration System
**Last Updated:** November 13, 2025

## Overview
This system provides centralized configuration for all sports with their respective APIs, features, and display settings. Users can seamlessly switch between different sports and the website automatically pulls data from the appropriate endpoints.

## Supported Sports

### 🏈 NFL (Primary - Fully Functional)
- **APIs:** ESPN Scoreboard, News, Standings, Teams
- **Features:** ML Predictions ✅, Live Scores ✅, Player Stats ✅, News ✅
- **Refresh:** Every 30 seconds
- **Status:** Fully operational with existing ML backend

### 🏀 NBA (Fully Functional)
- **APIs:** ESPN Scoreboard, News, Standings, Teams  
- **Features:** ML Predictions ✅, Live Scores ✅, Player Stats ✅, News ✅
- **Refresh:** Every 15 seconds (faster pace)
- **Status:** Complete with professional UI and ESPN integration

### ⚾ MLB (Ready for Season)
- **APIs:** ESPN + MLB Stats API fallback
- **Features:** Live Scores ✅, Player Stats ✅, News ✅, ML Predictions (Coming Soon)
- **Refresh:** Every 60 seconds (slower pace)
- **Status:** Configured, ready for activation

### 🏒 NHL (Ready)
- **APIs:** ESPN + NHL Stats API fallback
- **Features:** Live Scores ✅, Player Stats ✅, News ✅, ML Predictions (Coming Soon)
- **Refresh:** Every 20 seconds
- **Status:** Configured, ready for activation

### ⚽ MLS (Ready)
- **APIs:** ESPN Soccer API
- **Features:** Live Scores ✅, Player Stats ✅, News ✅
- **Refresh:** Every 30 seconds
- **Status:** Configured, ready for activation

## Key Files

### `/assets/js/sports-config.js`
**Main configuration file containing:**
- `SportsConfig` object with all sport settings
- `SportDataFetcher` class for API handling
- `UniversalSportsManager` class for coordination
- Standardized data parsing for all sports
- Auto-refresh systems per sport
- Fallback API handling

### Updated in `index.html`
- Sport selector buttons now use `universalSportsManager.switchSport()`
- Fallback system to old emergency functions
- Automatic initialization on page load

## How It Works

1. **Sport Selection:** User clicks any sport icon
2. **Manager Switch:** `UniversalSportsManager.switchSport()` is called
3. **API Fetching:** Appropriate `SportDataFetcher` instance created
4. **Data Display:** Games and news displayed with sport-specific styling
5. **Auto-Refresh:** Background updates at sport-appropriate intervals

## API Endpoints

### ESPN APIs (Primary)
All sports use ESPN as the primary data source:
- Scoreboard: `https://site.api.espn.com/apis/site/v2/sports/{sport}/{league}/scoreboard`
- News: `https://site.api.espn.com/apis/site/v2/sports/{sport}/{league}/news`
- Standings: `https://site.api.espn.com/apis/site/v2/sports/{sport}/{league}/standings`
- Teams: `https://site.api.espn.com/apis/site/v2/sports/{sport}/{league}/teams`

### Fallback APIs
Each sport has backup APIs for redundancy:
- **NBA:** Ball Don't Lie API
- **MLB:** MLB Stats API  
- **NHL:** NHL Stats API
- **NFL:** Tank01 API (existing)

## Standardized Data Format

All sports return data in this format:
```javascript
{
    id: "game_id",
    date: "ISO_date_string",
    status: "game_status",
    homeTeam: { name, logo, score, record },
    awayTeam: { name, logo, score, record },
    venue: { name, city, state },
    broadcasts: ["TV", "channels"],
    sport: "sport_key",
    sportName: "display_name"
}
```

## Usage Examples

### Switch to NBA
```javascript
await universalSportsManager.switchSport('basketball');
```

### Add New Sport
```javascript
// Add to SportsConfig object in sports-config.js
newSport: {
    name: 'Display Name',
    apis: { espn: { scoreboard: 'url' } },
    features: { hasMLPredictions: false },
    displaySettings: { refreshInterval: 30000 }
}
```

### Manual API Call
```javascript
const fetcher = new SportDataFetcher('basketball');
const games = await fetcher.fetchGames();
const news = await fetcher.fetchNews();
```

## Benefits

1. **Centralized Configuration:** One file controls all sports
2. **Consistent APIs:** ESPN provides standardized data across all sports
3. **Automatic Fallbacks:** Backup APIs prevent downtime
4. **Sport-Specific Settings:** Each sport has optimized refresh rates and features
5. **Easy Expansion:** Adding new sports requires just configuration updates
6. **Standardized UI:** All sports use the same professional card layouts
7. **Live Updates:** Automatic refresh systems keep data current

## Development Notes

- The system gracefully degrades to the old emergency functions if needed
- All emoji characters removed from JavaScript to prevent parse errors
- Each sport can have different display settings (refresh rates, UI elements)
- ML prediction integration ready for all sports
- Professional styling matches existing NFL system

## Next Steps

1. **Test all sports** on live site: https://david-osb.github.io/betyard.net/
2. **Enable MLB/NHL/MLS** by removing "Coming Soon" badges
3. **Connect ML predictions** to new sports when ready
4. **Add more sports** by expanding the SportsConfig object

This system provides the foundation for a truly multi-sport betting platform with consistent user experience across all leagues.