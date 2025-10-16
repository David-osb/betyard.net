# 🏈 NFL Quarterback Roster API Setup Instructions

## Overview
Your NFL QB Predictor now includes integration with RapidAPI's NFL API Data service to fetch real-time quarterback and backup quarterback roster information for all NFL teams.

## Setup Steps

### 1. Get Your RapidAPI Key
1. Go to [RapidAPI NFL API Data](https://rapidapi.com/Creativesdev/api/nfl-api-data)
2. Sign up for a free RapidAPI account (if you don't have one)
3. Subscribe to the NFL API Data service (free tier available)
4. Copy your API key from the dashboard

### 2. Configure the API Key
1. Open `UI.roughdraft2.html` in your code editor
2. Find line with: `const RAPIDAPI_KEY = 'YOUR_RAPIDAPI_KEY_HERE';`
3. Replace `YOUR_RAPIDAPI_KEY_HERE` with your actual RapidAPI key
4. Save the file

### 3. Features Available

#### Automatic Data Loading
- On page load, the system will automatically try to fetch live QB roster data
- Priority order: RapidAPI → ScrapeOwl → Emergency cached data

#### Manual Data Refresh
- Click the **🏈 Fetch QB Roster Data** button to manually refresh quarterback data
- This will fetch the latest roster information for all NFL teams

#### Data Includes
- **Starting quarterbacks** for all 32 NFL teams
- **Backup quarterbacks** where available
- Player information: name, position, jersey number, height, weight, age, college
- Current season stats projections
- Team and roster metadata

### 4. API Endpoints Used
The system uses the following RapidAPI endpoints:
- **Team Roster**: `GET /nfl-team-roster/{teamId}` - Gets complete roster for specific team
- **Rate Limiting**: 500ms delays between requests to respect API limits

### 5. Error Handling
If RapidAPI fails:
1. System falls back to ScrapeOwl web scraping
2. If that fails, uses cached emergency data
3. Status messages show which data source is active

### 6. Teams Covered
Currently fetches quarterback data for these priority teams:
- Kansas City Chiefs (KC)
- Buffalo Bills (BUF)
- Dallas Cowboys (DAL)
- Philadelphia Eagles (PHI)
- Baltimore Ravens (BAL)
- San Francisco 49ers (SF)
- Green Bay Packers (GB)
- Miami Dolphins (MIA)
- Cincinnati Bengals (CIN)
- New York Jets (NYJ)

*Can be expanded to all 32 teams by modifying the `nflTeams` array in the code.*

### 7. Data Structure
Each quarterback object contains:
```javascript
{
    name: "Player Name",
    position: "QB",
    jersey: "12",
    height: "6-4",
    weight: "225",
    age: "29",
    experience: "7",
    college: "University Name",
    status: "Starter" | "Backup",
    depth: 1,
    stats: {
        yards: 3200,
        completions: 275,
        attempts: 425,
        touchdowns: 28,
        interceptions: 8,
        rating: 102,
        completionPercentage: "64.7"
    },
    dataSource: "rapidapi_nfl_roster",
    lastUpdate: "2025-10-15T...",
    teamId: "KC",
    teamName: "Kansas City Chiefs"
}
```

### 8. Troubleshooting

#### "API Key Missing" Error
- Ensure you've replaced `YOUR_RAPIDAPI_KEY_HERE` with your actual key
- Check that the key doesn't have extra spaces or quotes

#### "Rate Limit Exceeded" Error
- The free tier has daily request limits
- Consider upgrading your RapidAPI plan for higher limits

#### "No Data Returned" Error
- Check your internet connection
- Verify the API service is operational
- The system will automatically fall back to cached data

### 9. Cost Information
- **Free Tier**: Usually includes 500-1000 requests per month
- **Paid Tiers**: Available for higher volume usage
- Check current pricing at [RapidAPI NFL API Data](https://rapidapi.com/Creativesdev/api/nfl-api-data)

## Support
If you encounter issues:
1. Check the browser console for error messages
2. Verify your API key configuration
3. Test with the manual fetch button first
4. The system provides detailed logging for troubleshooting

---
*Last updated: October 15, 2025*