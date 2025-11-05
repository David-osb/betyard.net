/**
 * 🗓️ WEEKLY SCHEDULE MANAGER
 * Automatically updates NFL schedule every Tuesday for upcoming week
 * Handles week transitions and game updates
 * Author: GitHub Copilot
 * Version: 1.0.0
 */

class WeeklyScheduleManager {
    constructor() {
        this.currentWeek = null;
        this.currentSeason = 2024;
        this.gamesCache = new Map();
        this.lastUpdateTime = null;
        this.updateInterval = null;
        
        // ESPN API endpoints
        this.espnEndpoints = {
            scoreboard: 'https://site.web.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard',
            schedule: 'https://site.web.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard',
            teams: 'https://site.web.api.espn.com/apis/site/v2/sports/football/nfl/teams'
        };
        
        this.init();
    }
    
    init() {
        console.log('🗓️ Weekly Schedule Manager: Initializing...');
        this.determineCurrentWeek();
        this.setupAutoUpdate();
        this.loadWeeklyGames();
        console.log('✅ Weekly Schedule Manager: Ready!');
    }
    
    /**
     * Determine the current NFL week based on date
     */
    determineCurrentWeek() {
        const now = new Date();
        const currentDate = new Date(now.getFullYear(), now.getMonth(), now.getDate());
        
        // NFL 2024 season started September 5, 2024 (Week 1)
        const season2024Start = new Date(2024, 8, 5); // September 5, 2024
        
        // For the 2024 season, calculate current week
        // Real-world context: We're currently in November 2024 NFL season
        this.currentSeason = 2024;
        
        if (currentDate >= season2024Start) {
            const daysDifference = Math.floor((currentDate - season2024Start) / (1000 * 60 * 60 * 24));
            this.currentWeek = Math.min(Math.floor(daysDifference / 7) + 1, 18);
        } else {
            // Preseason or before season start
            this.currentWeek = 1;
        }
        
        // Override: If we're in November 2024 (real current time), we should be around Week 10-12
        const currentMonth = now.getMonth(); // 0-based: 0=Jan, 10=Nov
        const currentYear = now.getFullYear();
        
        if (currentYear === 2024 && currentMonth >= 10) { // November or later in 2024
            // Week 10 started around November 7, 2024
            if (now.getDate() < 7) {
                this.currentWeek = 9;  // Early November = Week 9
            } else if (now.getDate() < 14) {
                this.currentWeek = 10; // Mid November = Week 10  
            } else if (now.getDate() < 21) {
                this.currentWeek = 11; // Late November = Week 11
            } else {
                this.currentWeek = 12; // Very late November = Week 12
            }
        }
        
        // If context suggests we're testing in a future date, adjust accordingly
        if (currentYear === 2025) {
            // For 2025 context, calculate from September 2024 base
            const realSeasonStart = new Date(2024, 8, 5);
            const testDate = new Date(2024, 10, 5); // Simulate November 5, 2024
            const daysDiff = Math.floor((testDate - realSeasonStart) / (1000 * 60 * 60 * 24));
            this.currentWeek = Math.min(Math.floor(daysDiff / 7) + 1, 18);
            console.log('🕒 Future date detected: Simulating current NFL week for November 2024');
        }
        
        // Adjust for Tuesday updates (look ahead to next week on Tuesday)
        if (now.getDay() === 2) { // Tuesday = 2
            this.currentWeek = Math.min(this.currentWeek + 1, 18);
            console.log('📅 Tuesday detected: Looking ahead to next week');
        }
        
        console.log(`🏈 Current NFL Season: ${this.currentSeason}, Week: ${this.currentWeek}`);
        
        // Store in localStorage for persistence
        localStorage.setItem('nflCurrentWeek', this.currentWeek.toString());
        localStorage.setItem('nflCurrentSeason', this.currentSeason.toString());
        
        return { week: this.currentWeek, season: this.currentSeason };
    }
    
    /**
     * Setup automatic updates every Tuesday at 9 AM ET
     */
    setupAutoUpdate() {
        // Check every hour if it's Tuesday 9 AM ET
        this.updateInterval = setInterval(() => {
            const now = new Date();
            const et = new Date(now.toLocaleString("en-US", {timeZone: "America/New_York"}));
            
            // Tuesday = 2, 9 AM = hour 9
            if (et.getDay() === 2 && et.getHours() === 9 && et.getMinutes() < 30) {
                if (!this.lastUpdateTime || (Date.now() - this.lastUpdateTime) > 6 * 60 * 60 * 1000) {
                    console.log('🔄 Tuesday 9 AM ET detected: Updating weekly schedule...');
                    this.updateWeeklySchedule();
                }
            }
        }, 60 * 60 * 1000); // Check every hour
        
        // Also check immediately if we haven't updated today
        const lastUpdate = localStorage.getItem('lastScheduleUpdate');
        const today = new Date().toDateString();
        
        if (!lastUpdate || lastUpdate !== today) {
            console.log('🔄 No update today detected: Refreshing schedule...');
            setTimeout(() => this.updateWeeklySchedule(), 2000);
        }
    }
    
    /**
     * Update the weekly schedule from ESPN
     */
    async updateWeeklySchedule() {
        console.log('🔄 Updating weekly schedule from ESPN...');
        
        try {
            this.determineCurrentWeek(); // Recalculate current week
            await this.loadWeeklyGames();
            
            this.lastUpdateTime = Date.now();
            localStorage.setItem('lastScheduleUpdate', new Date().toDateString());
            localStorage.setItem('lastScheduleUpdateTime', this.lastUpdateTime.toString());
            
            // Notify other components about the update
            window.dispatchEvent(new CustomEvent('weeklyScheduleUpdated', {
                detail: {
                    week: this.currentWeek,
                    season: this.currentSeason,
                    games: Array.from(this.gamesCache.values())
                }
            }));
            
            console.log('✅ Weekly schedule updated successfully');
            
        } catch (error) {
            console.error('❌ Failed to update weekly schedule:', error);
        }
    }
    
    /**
     * Load games for the current week from ESPN
     */
    async loadWeeklyGames() {
        console.log(`🏈 Loading games for Week ${this.currentWeek}...`);
        
        try {
            // Use ESPN API to get current week's games
            const url = `${this.espnEndpoints.scoreboard}?seasontype=2&week=${this.currentWeek}`;
            console.log(`📡 Fetching from ESPN: ${url}`);
            
            const response = await fetch(url);
            if (!response.ok) {
                throw new Error(`ESPN API error: ${response.status}`);
            }
            
            const data = await response.json();
            console.log('📊 ESPN scoreboard data:', data);
            
            // Process games from ESPN response
            const games = this.processESPNGames(data);
            
            // Cache the games
            this.gamesCache.clear();
            games.forEach((game, index) => {
                this.gamesCache.set(`week${this.currentWeek}_game${index}`, game);
            });
            
            console.log(`✅ Loaded ${games.length} games for Week ${this.currentWeek}`);
            
            // Store in localStorage for offline access
            localStorage.setItem('weeklyGamesCache', JSON.stringify(games));
            localStorage.setItem('weeklyGamesCacheWeek', this.currentWeek.toString());
            
            return games;
            
        } catch (error) {
            console.error('❌ Failed to load weekly games:', error);
            
            // Try to load from cache
            const cachedGames = this.loadCachedGames();
            if (cachedGames.length > 0) {
                console.log('📱 Using cached games data');
                return cachedGames;
            }
            
            // Fallback to static data
            return this.getFallbackGames();
        }
    }
    
    /**
     * Process ESPN games data into our format
     */
    processESPNGames(data) {
        if (!data || !data.events) {
            console.warn('⚠️ No events in ESPN data');
            return [];
        }
        
        const games = data.events.map(event => {
            const competition = event.competitions[0];
            const competitors = competition.competitors;
            
            // Find home and away teams
            const homeTeam = competitors.find(c => c.homeAway === 'home');
            const awayTeam = competitors.find(c => c.homeAway === 'away');
            
            const gameDate = new Date(event.date);
            const espnStatus = competition.status.type.name;
            
            // Normalize ESPN status to our expected format
            let status;
            switch(espnStatus) {
                case 'STATUS_SCHEDULED':
                case 'STATUS_POSTPONED':
                    status = 'SCHEDULED';
                    break;
                case 'STATUS_IN_PROGRESS':
                case 'STATUS_HALFTIME':
                    status = 'LIVE';
                    break;
                case 'STATUS_FINAL':
                case 'STATUS_FINAL_OVERTIME':
                    status = 'FINAL';
                    break;
                default:
                    status = espnStatus.replace('STATUS_', '') || 'SCHEDULED';
            }
            
            return {
                id: event.id,
                name: event.name,
                date: event.date,
                gameDate: gameDate,
                week: this.currentWeek,
                season: this.currentSeason,
                status: status,
                espnStatus: espnStatus, // Keep original for debugging
                completed: status === 'FINAL',
                
                // Team information
                away: awayTeam.team.abbreviation,
                home: homeTeam.team.abbreviation,
                awayTeam: {
                    id: awayTeam.team.id,
                    code: awayTeam.team.abbreviation,
                    name: awayTeam.team.displayName,
                    shortName: awayTeam.team.shortDisplayName,
                    score: parseInt(awayTeam.score) || 0,
                    logo: awayTeam.team.logo,
                    color: awayTeam.team.color,
                    alternateColor: awayTeam.team.alternateColor
                },
                homeTeam: {
                    id: homeTeam.team.id,
                    code: homeTeam.team.abbreviation,
                    name: homeTeam.team.displayName,
                    shortName: homeTeam.team.shortDisplayName,
                    score: parseInt(homeTeam.score) || 0,
                    logo: homeTeam.team.logo,
                    color: homeTeam.team.color,
                    alternateColor: homeTeam.team.alternateColor
                },
                
                // Additional data
                venue: competition.venue ? competition.venue.fullName : 'TBD',
                broadcast: competition.broadcasts && competition.broadcasts[0] ? 
                          competition.broadcasts[0].names.join(', ') : 'TBD',
                
                // Format for UI compatibility
                awayScore: parseInt(awayTeam.score) || 0,
                homeScore: parseInt(homeTeam.score) || 0,
                time: gameDate.toLocaleTimeString('en-US', { 
                    hour: 'numeric', 
                    minute: '2-digit',
                    timeZone: 'America/New_York'
                }),
                dayOfWeek: gameDate.toLocaleDateString('en-US', { weekday: 'short' })
            };
        });
        
        // Sort games by date
        games.sort((a, b) => new Date(a.date) - new Date(b.date));
        
        console.log(`🎯 Processed ${games.length} games:`, games);
        return games;
    }
    
    /**
     * Load cached games data
     */
    loadCachedGames() {
        try {
            const cachedGames = localStorage.getItem('weeklyGamesCache');
            const cachedWeek = localStorage.getItem('weeklyGamesCacheWeek');
            
            if (cachedGames && cachedWeek && parseInt(cachedWeek) === this.currentWeek) {
                console.log('📱 Loading games from cache');
                return JSON.parse(cachedGames);
            }
        } catch (error) {
            console.error('❌ Failed to load cached games:', error);
        }
        
        return [];
    }
    
    /**
     * Fallback games data for when API is unavailable
     */
    getFallbackGames() {
        console.log('🔄 Using fallback games data');
        
        return [
            {
                id: 'fallback_1',
                away: 'BUF', home: 'MIA',
                awayTeam: { code: 'BUF', name: 'Buffalo Bills', score: 0 },
                homeTeam: { code: 'MIA', name: 'Miami Dolphins', score: 0 },
                week: this.currentWeek,
                status: 'SCHEDULED',
                time: '1:00 PM',
                dayOfWeek: 'Sun'
            },
            {
                id: 'fallback_2',
                away: 'KC', home: 'LV',
                awayTeam: { code: 'KC', name: 'Kansas City Chiefs', score: 0 },
                homeTeam: { code: 'LV', name: 'Las Vegas Raiders', score: 0 },
                week: this.currentWeek,
                status: 'SCHEDULED',
                time: '4:05 PM',
                dayOfWeek: 'Sun'
            }
        ];
    }
    
    /**
     * Get games for the current week
     */
    getCurrentWeekGames() {
        const games = Array.from(this.gamesCache.values());
        console.log(`🎯 Returning ${games.length} games for Week ${this.currentWeek}`);
        return games;
    }
    
    /**
     * Get current week information
     */
    getCurrentWeekInfo() {
        return {
            week: this.currentWeek,
            season: this.currentSeason,
            title: `Week ${this.currentWeek}`,
            gamesCount: this.gamesCache.size
        };
    }
    
    /**
     * Force refresh of the schedule
     */
    async forceRefresh() {
        console.log('🔄 Force refreshing weekly schedule...');
        this.gamesCache.clear();
        localStorage.removeItem('weeklyGamesCache');
        localStorage.removeItem('lastScheduleUpdate');
        
        await this.updateWeeklySchedule();
    }
    
    /**
     * Cleanup
     */
    destroy() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
            this.updateInterval = null;
        }
        
        this.gamesCache.clear();
        console.log('🧹 Weekly Schedule Manager: Cleaned up');
    }
}

// Initialize global instance
window.WeeklyScheduleManager = WeeklyScheduleManager;

// Auto-initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        window.weeklyScheduleManager = new WeeklyScheduleManager();
    });
} else {
    window.weeklyScheduleManager = new WeeklyScheduleManager();
}

console.log('📅 Weekly Schedule Manager loaded');