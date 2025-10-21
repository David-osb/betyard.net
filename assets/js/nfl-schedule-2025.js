// NFL 2025 Schedule Data
// Parsed from team schedules to create matchups
// Format: @ indicates away team

const NFL_2025_SCHEDULE = {
    // Week 8 schedule (October 21-27, 2025)
    8: [
        { away: 'BAL', home: 'CHI' },
        { away: 'ATL', home: 'MIA' },
        { away: 'CAR', home: 'BUF' },
        { away: 'ARI', home: 'GB', status: 'BYE for ARI' },
        { away: 'IND', home: 'LAC' },
        { away: 'JAX', home: 'LAR' },
        { away: 'KC', home: 'LV' },
        { away: 'CLE', home: 'MIA' },
        { away: 'NE', home: 'TEN' },
        { away: 'NO', home: 'CHI' },
        { away: 'NYG', home: 'DEN' },
        { away: 'PHI', home: 'NYG' },
        { away: 'WAS', home: 'DAL' },
        { away: 'HOU', home: 'SEA' },
        { away: 'TB', home: 'DET' },
        { away: 'DAL', home: 'DEN' },
        { away: 'SF', home: 'HOU' },
        { away: 'KC', home: 'WAS' },
        { away: 'IND', home: 'TEN' },
        { away: 'CIN', home: 'NYJ' },
        { away: 'MIN', home: 'LAC' }
    ],
    
    // Week 9 schedule (October 28 - November 3, 2025)
    9: [
        { away: 'DAL', home: 'ARI' },
        { away: 'ATL', home: 'NE' },
        { away: 'BAL', home: 'MIA' },
        { away: 'BUF', home: 'KC' },
        { away: 'CAR', home: 'GB' },
        { away: 'CHI', home: 'CIN' },
        { away: 'DEN', home: 'HOU' },
        { away: 'MIN', home: 'DET' },
        { away: 'LAR', home: 'NO' },
        { away: 'LV', home: 'JAX' },
        { away: 'NE', home: 'ATL' },
        { away: 'SEA', home: 'WAS' },
        { away: 'SF', home: 'NYG' }
    ],
    
    // Week 10 schedule (November 4-10, 2025)
    10: [
        { away: 'ARI', home: 'SEA' },
        { away: 'ATL', home: 'IND' },
        { away: 'BAL', home: 'MIN' },
        { away: 'BUF', home: 'MIA' },
        { away: 'CAR', home: 'NO' },
        { away: 'CHI', home: 'NYG' },
        { away: 'CLE', home: 'NYJ' },
        { away: 'DEN', home: 'LV' },
        { away: 'DET', home: 'WAS' },
        { away: 'HOU', home: 'JAX' },
        { away: 'LAC', home: 'PIT' },
        { away: 'NE', home: 'TB' }
    ],
    
    // Week 11 schedule (November 11-17, 2025)
    11: [
        { away: 'ARI', home: 'SF' },
        { away: 'ATL', home: 'CAR' },
        { away: 'BAL', home: 'CLE' },
        { away: 'BUF', home: 'TB' },
        { away: 'CHI', home: 'MIN' },
        { away: 'CIN', home: 'PIT' },
        { away: 'DEN', home: 'KC' },
        { away: 'DET', home: 'PHI' },
        { away: 'GB', home: 'NYG' },
        { away: 'LAC', home: 'JAX' },
        { away: 'MIA', home: 'WAS' },
        { away: 'TEN', home: 'HOU' }
    ],
    
    // Week 12 schedule (November 18-24, 2025)
    12: [
        { away: 'ARI', home: 'JAX' },
        { away: 'ATL', home: 'NO' },
        { away: 'BAL', home: 'NYJ' },
        { away: 'BUF', home: 'HOU' },
        { away: 'CAR', home: 'SF' },
        { away: 'CHI', home: 'PIT' },
        { away: 'CIN', home: 'NE' },
        { away: 'CLE', home: 'LV' },
        { away: 'DAL', home: 'PHI' },
        { away: 'GB', home: 'MIN' },
        { away: 'IND', home: 'KC' },
        { away: 'LAR', home: 'TB' },
        { away: 'LAR', home: 'SEA' },
        { away: 'NE', home: 'NYJ' },
        { away: 'NYG', home: 'DET' },
        { away: 'SEA', home: 'TEN' }
    ],
    
    // Week 13 schedule (November 25 - December 1, 2025)
    13: [
        { away: 'ARI', home: 'TB' },
        { away: 'ATL', home: 'NYJ' },
        { away: 'BAL', home: 'CIN' },
        { away: 'BUF', home: 'PIT' },
        { away: 'CHI', home: 'PHI' },
        { away: 'CLE', home: 'SF' },
        { away: 'DEN', home: 'WAS' },
        { away: 'HOU', home: 'IND' },
        { away: 'JAX', home: 'TEN' },
        { away: 'KC', home: 'DAL' },
        { away: 'LAR', home: 'CAR' },
        { away: 'MIN', home: 'SEA' },
        { away: 'MIA', home: 'NO' },
        { away: 'NE', home: 'NYG' }
    ],
    
    // Week 14 schedule (December 2-8, 2025)
    14: [
        { away: 'ARI', home: 'LAR' },
        { away: 'ATL', home: 'SEA' },
        { away: 'BAL', home: 'PIT' },
        { away: 'BUF', home: 'CIN' },
        { away: 'CHI', home: 'GB' },
        { away: 'DAL', home: 'DET' },
        { away: 'DEN', home: 'LV' },
        { away: 'GB', home: 'DET' },
        { away: 'HOU', home: 'KC' },
        { away: 'IND', home: 'JAX' },
        { away: 'MIA', home: 'NYJ' },
        { away: 'MIN', home: 'DAL' },
        { away: 'NO', home: 'TB' },
        { away: 'CLE', home: 'TEN' }
    ],
    
    // Week 15 schedule (December 9-15, 2025)
    15: [
        { away: 'ATL', home: 'TB' },
        { away: 'BAL', home: 'NE' },
        { away: 'BUF', home: 'CLE' },
        { away: 'CHI', home: 'CLE' },
        { away: 'CIN', home: 'BAL' },
        { away: 'DEN', home: 'GB' },
        { away: 'HOU', home: 'ARI' },
        { away: 'IND', home: 'SEA' },
        { away: 'LAC', home: 'PHI' },
        { away: 'LAR', home: 'DET' },
        { away: 'LV', home: 'PHI' },
        { away: 'MIA', home: 'PIT' },
        { away: 'NO', home: 'CAR' },
        { away: 'SF', home: 'TEN' }
    ],
    
    // Week 16 schedule (December 16-22, 2025)
    16: [
        { away: 'ARI', home: 'ATL' },
        { away: 'BUF', home: 'PHI' },
        { away: 'CAR', home: 'TB' },
        { away: 'CHI', home: 'GB' },
        { away: 'DAL', home: 'LAC' },
        { away: 'DEN', home: 'JAX' },
        { away: 'DET', home: 'PIT' },
        { away: 'HOU', home: 'LV' },
        { away: 'IND', home: 'SF' },
        { away: 'KC', home: 'LAC' },
        { away: 'LAR', home: 'SEA' },
        { away: 'MIA', home: 'CIN' },
        { away: 'MIN', home: 'NYG' },
        { away: 'NO', home: 'NYJ' },
        { away: 'PHI', home: 'WAS' },
        { away: 'TEN', home: 'KC' }
    ],
    
    // Week 17 schedule (December 23-29, 2025)
    17: [
        { away: 'ARI', home: 'CIN' },
        { away: 'ATL', home: 'LAR' },
        { away: 'BAL', home: 'GB' },
        { away: 'CHI', home: 'SF' },
        { away: 'CLE', home: 'PIT' },
        { away: 'DAL', home: 'WAS' },
        { away: 'DEN', home: 'KC' },
        { away: 'HOU', home: 'LAC' },
        { away: 'IND', home: 'JAX' },
        { away: 'LV', home: 'NYG' },
        { away: 'MIN', home: 'DET' },
        { away: 'NE', home: 'NYJ' },
        { away: 'SEA', home: 'CAR' },
        { away: 'TB', home: 'MIA' },
        { away: 'TEN', home: 'NO' }
    ],
    
    // Week 18 schedule (December 30 - January 5, 2026)
    18: [
        { away: 'ARI', home: 'LAR' },
        { away: 'ATL', home: 'NO' },
        { away: 'BAL', home: 'PIT' },
        { away: 'BUF', home: 'NYJ' },
        { away: 'CAR', home: 'TB' },
        { away: 'CHI', home: 'DET' },
        { away: 'CIN', home: 'CLE' },
        { away: 'DAL', home: 'NYG' },
        { away: 'DEN', home: 'LAC' },
        { away: 'HOU', home: 'IND' },
        { away: 'JAX', home: 'TEN' },
        { away: 'KC', home: 'LV' },
        { away: 'MIA', home: 'NE' },
        { away: 'MIN', home: 'GB' },
        { away: 'PHI', home: 'WAS' },
        { away: 'SEA', home: 'SF' }
    ]
};

// Team name mappings
const TEAM_NAMES = {
    'ARI': 'Cardinals',
    'ATL': 'Falcons',
    'BAL': 'Ravens',
    'BUF': 'Bills',
    'CAR': 'Panthers',
    'CHI': 'Bears',
    'CIN': 'Bengals',
    'CLE': 'Browns',
    'DAL': 'Cowboys',
    'DEN': 'Broncos',
    'DET': 'Lions',
    'GB': 'Packers',
    'HOU': 'Texans',
    'IND': 'Colts',
    'JAX': 'Jaguars',
    'KC': 'Chiefs',
    'LAC': 'Chargers',
    'LAR': 'Rams',
    'LV': 'Raiders',
    'MIA': 'Dolphins',
    'MIN': 'Vikings',
    'NE': 'Patriots',
    'NO': 'Saints',
    'NYG': 'Giants',
    'NYJ': 'Jets',
    'PHI': 'Eagles',
    'PIT': 'Steelers',
    'SEA': 'Seahawks',
    'SF': '49ers',
    'TB': 'Buccaneers',
    'TEN': 'Titans',
    'WAS': 'Commanders'
};

// Get games for a specific week
function getWeekSchedule(week) {
    const games = NFL_2025_SCHEDULE[week];
    if (!games) {
        console.warn(`⚠️ No schedule data for Week ${week}`);
        return [];
    }
    
    return games.map(game => ({
        away: game.away,
        home: game.home,
        awayTeam: {
            code: game.away,
            name: TEAM_NAMES[game.away] || game.away,
            score: 0
        },
        homeTeam: {
            code: game.home,
            name: TEAM_NAMES[game.home] || game.home,
            score: 0
        },
        status: 'SCHEDULED',
        time: 'TBD',
        week: week,
        gameDate: '',
        awayScore: 0,
        homeScore: 0,
        source: 'static_schedule'
    }));
}

// Export for use in other modules
window.NFLStaticSchedule = {
    getWeekSchedule,
    TEAM_NAMES
};

console.log('📅 NFL 2025 Static Schedule loaded!');
