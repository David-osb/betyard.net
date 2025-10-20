/**
 * 
 * NFL Schedule and Week Calculation Module
 * Provides dynamic NFL week information based on current date
 */

// NFL 2025-2026 Season Schedule Data
const NFL_2025_SCHEDULE = {
    seasonStart: new Date('2025-09-04'), // Week 1 start
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
        week7: [
            { away: 'DEN', home: 'NO', time: 'Thursday 8:15 PM ET' },
            { away: 'HOU', home: 'GB', time: 'Sunday 1:00 PM ET' },
            { away: 'SEA', home: 'ATL', time: 'Sunday 1:00 PM ET' },
            { away: 'TEN', home: 'BUF', time: 'Sunday 1:00 PM ET' },
            { away: 'MIA', home: 'IND', time: 'Sunday 1:00 PM ET' },
            { away: 'CLE', home: 'CIN', time: 'Sunday 1:00 PM ET' },
            { away: 'NE', home: 'JAX', time: 'Sunday 9:30 AM ET (London)' },
            { away: 'LV', home: 'LAR', time: 'Sunday 4:05 PM ET' },
            { away: 'PHI', home: 'NYG', time: 'Sunday 1:00 PM ET' },
            { away: 'KC', home: 'SF', time: 'Sunday 4:25 PM ET' },
            { away: 'DET', home: 'MIN', time: 'Sunday 1:00 PM ET' },
            { away: 'CAR', home: 'WSH', time: 'Sunday 4:05 PM ET' },
            { away: 'NYJ', home: 'PIT', time: 'Sunday 8:20 PM ET' },
            { away: 'LAC', home: 'ARI', time: 'Monday 9:00 PM ET' }
        ],
        week8: [
            { away: 'NYJ', home: 'NE', time: 'Thursday 8:15 PM ET' },
            { away: 'BAL', home: 'CLE', time: 'Sunday 1:00 PM ET' },
            { away: 'ARI', home: 'MIA', time: 'Sunday 1:00 PM ET' },
            { away: 'ATL', home: 'TB', time: 'Sunday 1:00 PM ET' },
            { away: 'GB', home: 'JAX', time: 'Sunday 1:00 PM ET' },
            { away: 'IND', home: 'HOU', time: 'Sunday 1:00 PM ET' },
            { away: 'TEN', home: 'DET', time: 'Sunday 1:00 PM ET' },
            { away: 'PHI', home: 'CIN', time: 'Sunday 1:00 PM ET' },
            { away: 'CHI', home: 'WSH', time: 'Sunday 1:00 PM ET' },
            { away: 'BUF', home: 'SEA', time: 'Sunday 4:05 PM ET' },
            { away: 'LAR', home: 'MIN', time: 'Sunday 4:25 PM ET' },
            { away: 'SF', home: 'DAL', time: 'Sunday 8:20 PM ET' },
            { away: 'LV', home: 'KC', time: 'Monday 8:15 PM ET' }
        ]
    }
};

// NFL Stadium Information
const NFL_STADIUMS = {
    'ARI': { name: 'State Farm Stadium', city: 'Glendale, AZ', surface: 'Artificial Turf', capacity: '63,400' },
    'ATL': { name: 'Mercedes-Benz Stadium', city: 'Atlanta, GA', surface: 'Artificial Turf', capacity: '71,000' },
    'BAL': { name: 'M&T Bank Stadium', city: 'Baltimore, MD', surface: 'Artificial Turf', capacity: '71,008' },
    'BUF': { name: 'Highmark Stadium', city: 'Orchard Park, NY', surface: 'Artificial Turf', capacity: '71,608' },
    'CAR': { name: 'Bank of America Stadium', city: 'Charlotte, NC', surface: 'Artificial Turf', capacity: '75,523' },
    'CHI': { name: 'Soldier Field', city: 'Chicago, IL', surface: 'Natural Grass', capacity: '61,500' },
    'CIN': { name: 'Paycor Stadium', city: 'Cincinnati, OH', surface: 'Artificial Turf', capacity: '65,515' },
    'CLE': { name: 'Cleveland Browns Stadium', city: 'Cleveland, OH', surface: 'Natural Grass', capacity: '67,431' },
    'DAL': { name: 'AT&T Stadium', city: 'Arlington, TX', surface: 'Artificial Turf', capacity: '80,000' },
    'DEN': { name: 'Empower Field at Mile High', city: 'Denver, CO', surface: 'Natural Grass', capacity: '76,125' },
    'DET': { name: 'Ford Field', city: 'Detroit, MI', surface: 'Artificial Turf', capacity: '65,000' },
    'GB': { name: 'Lambeau Field', city: 'Green Bay, WI', surface: 'Natural Grass', capacity: '81,441' },
    'HOU': { name: 'NRG Stadium', city: 'Houston, TX', surface: 'Artificial Turf', capacity: '72,220' },
    'IND': { name: 'Lucas Oil Stadium', city: 'Indianapolis, IN', surface: 'Artificial Turf', capacity: '67,000' },
    'JAX': { name: 'TIAA Bank Field', city: 'Jacksonville, FL', surface: 'Natural Grass', capacity: '67,838' },
    'KC': { name: 'Arrowhead Stadium', city: 'Kansas City, MO', surface: 'Natural Grass', capacity: '76,416' },
    'LV': { name: 'Allegiant Stadium', city: 'Las Vegas, NV', surface: 'Artificial Turf', capacity: '65,000' },
    'LAC': { name: 'SoFi Stadium', city: 'Los Angeles, CA', surface: 'Artificial Turf', capacity: '70,240' },
    'LAR': { name: 'SoFi Stadium', city: 'Los Angeles, CA', surface: 'Artificial Turf', capacity: '70,240' },
    'MIA': { name: 'Hard Rock Stadium', city: 'Miami Gardens, FL', surface: 'Natural Grass', capacity: '65,326' },
    'MIN': { name: 'U.S. Bank Stadium', city: 'Minneapolis, MN', surface: 'Artificial Turf', capacity: '66,860' },
    'NE': { name: 'Gillette Stadium', city: 'Foxborough, MA', surface: 'Artificial Turf', capacity: '65,878' },
    'NO': { name: 'Caesars Superdome', city: 'New Orleans, LA', surface: 'Artificial Turf', capacity: '73,208' },
    'NYG': { name: 'MetLife Stadium', city: 'East Rutherford, NJ', surface: 'Artificial Turf', capacity: '82,500' },
    'NYJ': { name: 'MetLife Stadium', city: 'East Rutherford, NJ', surface: 'Artificial Turf', capacity: '82,500' },
    'PIT': { name: 'Acrisure Stadium', city: 'Pittsburgh, PA', surface: 'Natural Grass', capacity: '68,400' },
    'PHI': { name: 'Lincoln Financial Field', city: 'Philadelphia, PA', surface: 'Natural Grass', capacity: '69,596' },
    'SF': { name: "Levi's Stadium", city: 'Santa Clara, CA', surface: 'Natural Grass', capacity: '68,500' },
    'SEA': { name: 'Lumen Field', city: 'Seattle, WA', surface: 'Artificial Turf', capacity: '69,000' },
    'TB': { name: 'Raymond James Stadium', city: 'Tampa, FL', surface: 'Natural Grass', capacity: '65,890' },
    'TEN': { name: 'Nissan Stadium', city: 'Nashville, TN', surface: 'Natural Grass', capacity: '69,143' },
    'WSH': { name: 'FedExField', city: 'Landover, MD', surface: 'Natural Grass', capacity: '82,000' }
};

/**
 * Get current NFL week based on today's date
 */
function getCurrentNFLWeek() {
    const today = new Date();
    
    // Check 2025-2026 season first
    const season2025Start = new Date('2025-09-04');
    const season2025End = new Date('2026-01-07');
    
    if (today >= season2025Start && today <= season2025End) {
        // We're in the 2025-2026 season
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
    
    // Handle preseason period (before season starts)
    if (today < season2025Start) {
        return {
            week: 0,
            title: 'Preseason - August 2025',
            status: 'preseason',
            season: '2025'
        };
    }
    
    // Handle offseason period (between seasons)
    if (today < season2025Start) {
        return {
            week: 0,
            title: 'Offseason - Summer 2025',
            status: 'offseason',
            season: '2025'
        };
    }
    
    if (today > season2025End) {
        return {
            week: 19,
            title: 'Playoffs - January 2026',
            status: 'playoffs',
            season: '2025'
        };
    }
    
    // Default fallback - assume current 2025 season week 7
    return {
        week: 7,
        title: 'Week 7 - October 2025',
        status: 'regular_season',
        season: '2025'
    };
}

/**
 * Get next game day for the current week
 */
function getNextGameDay() {
    const today = new Date();
    const dayOfWeek = today.getDay(); // 0 = Sunday, 1 = Monday, etc.
    
    // Most NFL games are on Sunday (0), some on Monday (1), Thursday (4)
    let nextGameDay = 'Sunday';
    let kickoffTime = '1:00 PM ET';
    
    if (dayOfWeek === 0) { // Today is Sunday
        nextGameDay = 'Today';
        kickoffTime = '1:00 PM ET';
    } else if (dayOfWeek === 1) { // Today is Monday
        nextGameDay = 'Tonight';
        kickoffTime = '8:15 PM ET';
    } else if (dayOfWeek === 4) { // Today is Thursday
        nextGameDay = 'Tonight';
        kickoffTime = '8:15 PM ET';
    } else if (dayOfWeek < 4) { // Tuesday, Wednesday
        nextGameDay = 'Thursday';
        kickoffTime = '8:15 PM ET';
    } else { // Friday, Saturday
        nextGameDay = 'Sunday';
        kickoffTime = '1:00 PM ET';
    }
    
    return `${nextGameDay} ${kickoffTime}`;
}

/**
 * Get stadium information for a team
 */
function getStadiumInfo(teamCode) {
    const stadium = NFL_STADIUMS[teamCode];
    if (!stadium) {
        return {
            name: 'NFL Stadium',
            city: 'NFL City',
            surface: 'Natural Grass',
            capacity: '65,000'
        };
    }
    return stadium;
}

/**
 * Find a team's game for a specific week
 */
function findTeamGame(teamCode, week, season = '2025') {
    let schedule;
    
    if (season === '2025') {
        schedule = NFL_2025_SCHEDULE.games;
    } else {
        schedule = NFL_2025_SCHEDULE.games;
    }
    
    const gameWeek = schedule[`week${week}`];
    if (!gameWeek) return null;
    
    for (const game of gameWeek) {
        if (game.home === teamCode || game.away === teamCode) {
            return {
                matchup: `${game.away} @ ${game.home}`,
                time: game.time,
                home: game.home,
                away: game.away,
                isHome: game.home === teamCode
            };
        }
    }
    return null; // Bye week
}

/**
 * Update the current week display
 */
function updateCurrentWeekDisplay(selectedTeam = 'Team') {
    const currentWeek = getCurrentNFLWeek();
    const teamGame = findTeamGame(selectedTeam, currentWeek.week, currentWeek.season);
    
    const currentWeekInfo = document.getElementById('current-week-info');
    if (currentWeekInfo) {
        if (teamGame) {
            // Team has a game this week
            const stadium = getStadiumInfo(teamGame.home);
            currentWeekInfo.innerHTML = `
                <p style="color: #0369a1; margin-bottom: 8px;"><strong>Week ${currentWeek.week}</strong> - ${currentWeek.title.split(' - ')[1]}</p>
                <p style="color: #0369a1; font-size: 14px;">Game: ${teamGame.matchup}</p>
                <p style="color: #0369a1; font-size: 14px;">Kickoff: ${teamGame.time}</p>
                <p style="color: #0369a1; font-size: 14px;">Stadium: ${stadium.name}</p>
            `;
        } else if (currentWeek.status === 'offseason') {
            // Offseason period
            currentWeekInfo.innerHTML = `
                <p style="color: #0369a1; margin-bottom: 8px;"><strong>Offseason</strong> - Spring 2025</p>
                <p style="color: #6b7280; font-size: 14px;">No games scheduled</p>
                <p style="color: #6b7280; font-size: 14px;">2025 season starts September 4th</p>
            `;
        } else {
            // Bye week or no game data
            currentWeekInfo.innerHTML = `
                <p style="color: #0369a1; margin-bottom: 8px;"><strong>Week ${currentWeek.week}</strong> - ${currentWeek.title.split(' - ')[1]}</p>
                <p style="color: #ff6b35; font-size: 14px; font-weight: bold;">BYE WEEK</p>
                <p style="color: #0369a1; font-size: 14px;">No game scheduled</p>
            `;
        }
    }
}

/**
 * Update stadium information display
 */
function updateStadiumInfo(teamCode) {
    const stadium = getStadiumInfo(teamCode);
    
    const stadiumInfoElement = document.querySelector('.info-sections .grid-item:first-child p');
    if (stadiumInfoElement) {
        stadiumInfoElement.innerHTML = `
            <strong>${stadium.name}</strong><br>
            ${stadium.city}<br>
            Surface: ${stadium.surface}<br>
            Capacity: ${stadium.capacity}
        `;
    }
}

/**
 * Get mock weather conditions (to be replaced with real API)
 */
function getMockWeatherConditions() {
    const conditions = [
        { temp: '72°F', condition: 'Clear', wind: '5 mph SW' },
        { temp: '45°F', condition: 'Cloudy', wind: '12 mph NW' },
        { temp: '28°F', condition: 'Snow', wind: '18 mph N' },
        { temp: '85°F', condition: 'Sunny', wind: '3 mph SE' },
        { temp: '55°F', condition: 'Light Rain', wind: '8 mph W' }
    ];
    
    return conditions[Math.floor(Math.random() * conditions.length)];
}

/**
 * Update game conditions display
 */
function updateGameConditions() {
    const weather = getMockWeatherConditions();
    
    const gameConditionsElement = document.querySelector('.info-sections .grid-item:nth-child(2) p');
    if (gameConditionsElement) {
        gameConditionsElement.innerHTML = `
            Temperature: ${weather.temp}<br>
            Conditions: ${weather.condition}<br>
            Wind: ${weather.wind}<br>
            Field: Professional turf
        `;
    }
}

/**
 * Initialize NFL schedule module
 */
function initializeNFLSchedule() {
    console.log('🏈 NFL Schedule module loaded');
    console.log(`📅 Current NFL Week: ${getCurrentNFLWeek().week}`);
    
    // Update displays on load
    updateCurrentWeekDisplay();
    updateGameConditions();
}

// Export functions for global use
window.NFLSchedule = {
    getCurrentNFLWeek,
    getNextGameDay,
    getStadiumInfo,
    findTeamGame,
    updateCurrentWeekDisplay,
    updateStadiumInfo,
    updateGameConditions,
    initializeNFLSchedule
};

// Auto-initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeNFLSchedule);
} else {
    initializeNFLSchedule();
}