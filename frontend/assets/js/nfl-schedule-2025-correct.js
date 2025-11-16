/**
 * NFL 2025 Schedule - CORRECT VERSION
 * Author: User-provided correct data
 * Week 8 includes Browns @ Patriots as specified
 */

const NFL_2025_SCHEDULE = {
    seasonStart: new Date('2025-09-04'),
    weeks: [
        { week: 1, start: '2025-09-04', end: '2025-09-10', title: 'Week 1 - September 2025' },
        { week: 2, start: '2025-09-11', end: '2025-09-17', title: 'Week 2 - September 2025' },
        { week: 3, start: '2025-09-18', end: '2025-09-24', title: 'Week 3 - September 2025' },
        { week: 4, start: '2025-09-25', end: '2025-10-01', title: 'Week 4 - September/October 2025' },
        { week: 5, start: '2025-10-02', end: '2025-10-08', title: 'Week 5 - October 2025' },
        { week: 6, start: '2025-10-09', end: '2025-10-15', title: 'Week 6 - October 2025' },
        { week: 7, start: '2025-10-16', end: '2025-10-22', title: 'Week 7 - October 2025' },
        { week: 8, start: '2025-10-23', end: '2025-10-29', title: 'Week 8 - October 2025' },
        { week: 9, start: '2025-10-30', end: '2025-11-05', title: 'Week 9 - November 2025' },
        { week: 10, start: '2025-11-06', end: '2025-11-12', title: 'Week 10 - November 2025' },
        { week: 11, start: '2025-11-13', end: '2025-11-19', title: 'Week 11 - November 2025' },
        { week: 12, start: '2025-11-20', end: '2025-11-26', title: 'Week 12 - November 2025' },
        { week: 13, start: '2025-11-27', end: '2025-12-03', title: 'Week 13 - December 2025' },
        { week: 14, start: '2025-12-04', end: '2025-12-10', title: 'Week 14 - December 2025' },
        { week: 15, start: '2025-12-11', end: '2025-12-17', title: 'Week 15 - December 2025' },
        { week: 16, start: '2025-12-18', end: '2025-12-24', title: 'Week 16 - December 2025' },
        { week: 17, start: '2025-12-25', end: '2025-12-31', title: 'Week 17 - December 2025' },
        { week: 18, start: '2026-01-01', end: '2026-01-07', title: 'Week 18 - January 2026' }
    ],
    games: {
        // WEEK 8 - CORRECT DATA FROM CSV: October 23-27, 2025
        week8: [
            { away: 'MIN', home: 'LAC', time: 'Thursday 8:15 PM ET', gameDate: '2025-10-23', network: 'Prime Video' },
            { away: 'MIA', home: 'ATL', time: 'Sunday 1:00 PM ET', gameDate: '2025-10-26', network: 'CBS' },
            { away: 'CHI', home: 'BAL', time: 'Sunday 1:00 PM ET', gameDate: '2025-10-26', network: 'CBS' },
            { away: 'BUF', home: 'CAR', time: 'Sunday 1:00 PM ET', gameDate: '2025-10-26', network: 'FOX' },
            { away: 'NYJ', home: 'CIN', time: 'Sunday 1:00 PM ET', gameDate: '2025-10-26', network: 'CBS' },
            { away: 'SF', home: 'HOU', time: 'Sunday 1:00 PM ET', gameDate: '2025-10-26', network: 'FOX' },
            { away: 'CLE', home: 'NE', time: 'Sunday 1:00 PM ET', gameDate: '2025-10-26', network: 'FOX' }, // BROWNS @ PATRIOTS ✅
            { away: 'NYG', home: 'PHI', time: 'Sunday 1:00 PM ET', gameDate: '2025-10-26', network: 'FOX' },
            { away: 'TB', home: 'NO', time: 'Sunday 4:05 PM ET', gameDate: '2025-10-26', network: 'FOX' },
            { away: 'DAL', home: 'DEN', time: 'Sunday 4:25 PM ET', gameDate: '2025-10-26', network: 'CBS' },
            { away: 'TEN', home: 'IND', time: 'Sunday 4:25 PM ET', gameDate: '2025-10-26', network: 'CBS' },
            { away: 'GB', home: 'PIT', time: 'Sunday 8:20 PM ET', gameDate: '2025-10-26', network: 'NBC' },
            { away: 'WAS', home: 'KC', time: 'Monday 8:15 PM ET', gameDate: '2025-10-27', network: 'ESPN' }
        ],
        
        // WEEK 7 - Add correct data here
        week7: [
            // Please provide correct Week 7 matchups
        ],
        
        // WEEK 9 - Add correct data here  
        week9: [
            // Please provide correct Week 9 matchups
        ]
        
        // Continue for all weeks...
    }
};

/**
 * Get current NFL week based on today's date
 */
function getCurrentNFLWeek() {
    const today = new Date();
    
    // Check 2025-2026 season
    const season2025Start = new Date('2025-09-04');
    const season2025End = new Date('2026-01-07');
    
    if (today >= season2025Start && today <= season2025End) {
        for (const weekData of NFL_2025_SCHEDULE.weeks) {
            const weekStart = new Date(weekData.start);
            const weekEnd = new Date(weekData.end);
            
            if (today >= weekStart && today <= weekEnd) {
                return {
                    week: weekData.week,
                    title: weekData.title,
                    status: 'regular_season',
                    season: '2025'
                };
            }
        }
    }
    
    // Default to Week 9 (current week as of Nov 3, 2025)
    return {
        week: 9,
        title: 'Week 9 - November 2025',
        status: 'regular_season', 
        season: '2025'
    };
}

/**
 * Find a team's game for a specific week
 */
function findTeamGame(teamCode, week, season = '2025') {
    const schedule = NFL_2025_SCHEDULE.games;
    const gameWeek = schedule[`week${week}`];
    
    if (!gameWeek) return null;
    
    for (const game of gameWeek) {
        if (game.home === teamCode || game.away === teamCode) {
            return {
                matchup: `${game.away} @ ${game.home}`,
                time: game.time,
                home: game.home,
                away: game.away,
                isHome: game.home === teamCode,
                gameDate: game.gameDate
            };
        }
    }
    return null; // Bye week
}

/**
 * Get games for a specific week
 */
function getWeekGames(week) {
    const schedule = NFL_2025_SCHEDULE.games;
    return schedule[`week${week}`] || [];
}

// Export functions for global use
window.NFLSchedule = {
    getCurrentNFLWeek,
    findTeamGame,
    getWeekGames,
    NFL_2025_SCHEDULE
};

// Make getCurrentWeek available globally for compatibility
window.getCurrentWeek = function() {
    return getCurrentNFLWeek().week;
};

console.log('🏈 NFL 2025 Schedule (CORRECT VERSION) loaded');
console.log(`📅 Current NFL Week: ${getCurrentNFLWeek().week} - Browns @ Patriots confirmed for Week 8`);