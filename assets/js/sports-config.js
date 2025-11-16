/**
 * Sports Configuration System
 * Centralized configuration for all sports with their respective APIs and display settings
 * Last Updated: November 13, 2025
 */

const SportsConfig = {
    // NFL Configuration
    football: {
        name: 'NFL',
        fullName: 'National Football League',
        icon: '🏈',
        logo: 'https://a.espncdn.com/combiner/i?img=/i/teamlogos/leagues/500/nfl.png&h=200&w=200',
        color: '#013369', // NFL Blue
        secondaryColor: '#D50A0A', // NFL Red
        
        apis: {
            // Primary ESPN API
            espn: {
                scoreboard: 'https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard',
                news: 'https://site.api.espn.com/apis/site/v2/sports/football/nfl/news',
                standings: 'https://site.api.espn.com/apis/site/v2/sports/football/nfl/standings',
                teams: 'https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams'
            },
            // Alternative APIs for redundancy
            fallback: {
                tank01: 'https://tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com/getNFLGamesForWeek',
                sportradar: null // Add when available
            }
        },
        
        features: {
            hasMLPredictions: true,
            hasPlayerStats: true,
            hasLiveScores: true,
            hasNews: true,
            hasSchedule: true,
            hasStandings: true
        },
        
        displaySettings: {
            maxGamesPerPage: 16,
            refreshInterval: 30000, // 30 seconds
            showVenue: true,
            showBroadcast: true,
            showWeather: true,
            showSpread: true
        }
    },

    // NBA Configuration
    basketball: {
        name: 'NBA',
        fullName: 'National Basketball Association',
        icon: '🏀',
        logo: 'https://a.espncdn.com/combiner/i?img=/i/teamlogos/leagues/500/nba.png&h=200&w=200',
        color: '#C8102E', // NBA Red
        secondaryColor: '#1D428A', // NBA Blue
        
        apis: {
            espn: {
                scoreboard: 'https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard',
                news: 'https://site.api.espn.com/apis/site/v2/sports/basketball/nba/news',
                standings: 'https://site.api.espn.com/apis/site/v2/sports/basketball/nba/standings',
                teams: 'https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams'
            },
            fallback: {
                balldontlie: 'https://www.balldontlie.io/api/v1/games',
                sportsdata: null // Add when available
            }
        },
        
        features: {
            hasMLPredictions: true,
            hasPlayerStats: true,
            hasLiveScores: true,
            hasNews: true,
            hasSchedule: true,
            hasStandings: true
        },
        
        displaySettings: {
            maxGamesPerPage: 12,
            refreshInterval: 15000, // 15 seconds for faster NBA pace
            showVenue: true,
            showBroadcast: true,
            showWeather: false,
            showSpread: true
        }
    },

    // MLB Configuration
    baseball: {
        name: 'MLB',
        fullName: 'Major League Baseball',
        icon: '⚾',
        logo: 'https://a.espncdn.com/combiner/i?img=/i/teamlogos/leagues/500/mlb.png&h=200&w=200',
        color: '#041E42', // MLB Navy
        secondaryColor: '#BD3039', // MLB Red
        
        apis: {
            espn: {
                scoreboard: 'https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard',
                news: 'https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/news',
                standings: 'https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/standings',
                teams: 'https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/teams'
            },
            fallback: {
                mlbstats: 'https://statsapi.mlb.com/api/v1/schedule',
                sportsdata: null
            }
        },
        
        features: {
            hasMLPredictions: false, // Coming soon
            hasPlayerStats: true,
            hasLiveScores: true,
            hasNews: true,
            hasSchedule: true,
            hasStandings: true
        },
        
        displaySettings: {
            maxGamesPerPage: 15,
            refreshInterval: 60000, // 1 minute for slower MLB pace
            showVenue: true,
            showBroadcast: true,
            showWeather: true,
            showSpread: false
        }
    },

    // NHL Configuration
    hockey: {
        name: 'NHL',
        fullName: 'National Hockey League',
        icon: '🏒',
        logo: 'https://a.espncdn.com/combiner/i?img=/i/teamlogos/leagues/500/nhl.png&h=200&w=200',
        color: '#000000', // NHL Black
        secondaryColor: '#C8102E', // NHL Red
        
        apis: {
            espn: {
                scoreboard: 'https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard',
                news: 'https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/news',
                standings: 'https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/standings',
                teams: 'https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/teams'
            },
            fallback: {
                nhlapi: 'https://statsapi.web.nhl.com/api/v1/schedule',
                sportsdata: null
            }
        },
        
        features: {
            hasMLPredictions: false, // Coming soon
            hasPlayerStats: true,
            hasLiveScores: true,
            hasNews: true,
            hasSchedule: true,
            hasStandings: true
        },
        
        displaySettings: {
            maxGamesPerPage: 12,
            refreshInterval: 20000, // 20 seconds
            showVenue: true,
            showBroadcast: true,
            showWeather: false,
            showSpread: false
        }
    },

    // MLS Configuration
    soccer: {
        name: 'MLS',
        fullName: 'Major League Soccer',
        icon: '⚽',
        logo: 'https://a.espncdn.com/combiner/i?img=/i/teamlogos/leagues/500/mls.png&h=200&w=200',
        color: '#004F9F', // MLS Blue
        secondaryColor: '#8DC63F', // MLS Green
        
        apis: {
            espn: {
                scoreboard: 'https://site.api.espn.com/apis/site/v2/sports/soccer/usa.1/scoreboard',
                news: 'https://site.api.espn.com/apis/site/v2/sports/soccer/usa.1/news',
                standings: 'https://site.api.espn.com/apis/site/v2/sports/soccer/usa.1/standings',
                teams: 'https://site.api.espn.com/apis/site/v2/sports/soccer/usa.1/teams'
            },
            fallback: {
                mlsapi: null,
                sportsdata: null
            }
        },
        
        features: {
            hasMLPredictions: false, // Coming soon
            hasPlayerStats: true,
            hasLiveScores: true,
            hasNews: true,
            hasSchedule: true,
            hasStandings: true
        },
        
        displaySettings: {
            maxGamesPerPage: 10,
            refreshInterval: 30000, // 30 seconds
            showVenue: true,
            showBroadcast: true,
            showWeather: false,
            showSpread: false
        }
    },

    // Tennis Configuration (for future expansion)
    tennis: {
        name: 'Tennis',
        fullName: 'Professional Tennis',
        icon: '🎾',
        logo: 'https://a.espncdn.com/combiner/i?img=/i/teamlogos/leagues/500/atp.png&h=200&w=200',
        color: '#228B22', // Forest Green
        secondaryColor: '#FFD700', // Gold
        
        apis: {
            espn: {
                scoreboard: 'https://site.api.espn.com/apis/site/v2/sports/tennis/atp/scoreboard',
                news: 'https://site.api.espn.com/apis/site/v2/sports/tennis/news'
            },
            fallback: {
                atpapi: null,
                wtaapi: null
            }
        },
        
        features: {
            hasMLPredictions: false,
            hasPlayerStats: true,
            hasLiveScores: true,
            hasNews: true,
            hasSchedule: true,
            hasStandings: false
        },
        
        displaySettings: {
            maxGamesPerPage: 8,
            refreshInterval: 45000,
            showVenue: true,
            showBroadcast: false,
            showWeather: false,
            showSpread: false
        }
    }
};

/**
 * Sport Data Fetcher Class
 * Handles API calls and data formatting for all sports
 */
class SportDataFetcher {
    constructor(sportKey) {
        this.sport = SportsConfig[sportKey];
        this.sportKey = sportKey;
        
        if (!this.sport) {
            throw new Error(`Sport configuration not found for: ${sportKey}`);
        }
        
        console.log(`🏆 SportDataFetcher initialized for ${this.sport.fullName}`);
    }

    /**
     * Get today's date in YYYYMMDD format for ESPN APIs
     */
    getTodaysDate() {
        const today = new Date();
        const year = today.getFullYear();
        const month = String(today.getMonth() + 1).padStart(2, '0');
        const day = String(today.getDate()).padStart(2, '0');
        return `${year}${month}${day}`;
    }

    /**
     * Get date range for current week/season context
     */
    getDateRange() {
        const today = new Date();
        const todayStr = this.getTodaysDate();
        
        // Calculate week start (Monday) and end (Sunday) for broader context
        const dayOfWeek = today.getDay();
        const daysToMonday = dayOfWeek === 0 ? 6 : dayOfWeek - 1;
        
        const weekStart = new Date(today);
        weekStart.setDate(today.getDate() - daysToMonday);
        
        const weekEnd = new Date(weekStart);
        weekEnd.setDate(weekStart.getDate() + 6);
        
        return {
            today: todayStr,
            weekStart: this.formatDate(weekStart),
            weekEnd: this.formatDate(weekEnd)
        };
    }

    /**
     * Format date for ESPN API
     */
    formatDate(date) {
        const year = date.getFullYear();
        const month = String(date.getMonth() + 1).padStart(2, '0');
        const day = String(date.getDate()).padStart(2, '0');
        return `${year}${month}${day}`;
    }

    /**
     * Fetch games/scores for the sport with current date
     */
    async fetchGames() {
        const dateInfo = this.getDateRange();
        console.log(`🎮 Fetching ${this.sport.name} games for ${dateInfo.today}...`);
        
        try {
            // Build date-aware ESPN API URL
            let apiUrl = this.sport.apis.espn.scoreboard;
            
            // For most sports, add date parameter to get today's games
            if (apiUrl.includes('?')) {
                apiUrl += `&dates=${dateInfo.today}`;
            } else {
                apiUrl += `?dates=${dateInfo.today}`;
            }
            
            console.log(`📅 ${this.sport.name}: Fetching games for ${dateInfo.today} from:`, apiUrl);
            
            // Primary ESPN API call with today's date
            const response = await fetch(apiUrl);
            
            if (!response.ok) {
                throw new Error(`ESPN API error: ${response.status}`);
            }
            
            const data = await response.json();
            const gamesCount = data.events?.length || 0;
            console.log(`✅ ${this.sport.name}: Found ${gamesCount} games for today (${dateInfo.today})`);
            
            // If no games today, try the week range for context
            if (gamesCount === 0) {
                console.log(`📅 ${this.sport.name}: No games today, checking week range...`);
                return this.fetchWeekGames(dateInfo);
            }
            
            return this.parseGames(data.events || []);
            
        } catch (error) {
            console.error(`❌ ${this.sport.name}: Primary API failed:`, error);
            return this.fetchFallbackGames();
        }
    }

    /**
     * Fetch games for the current week when no games today
     */
    async fetchWeekGames(dateInfo) {
        try {
            let apiUrl = this.sport.apis.espn.scoreboard;
            
            // Remove any existing date params and add week range
            apiUrl = apiUrl.split('?')[0];
            apiUrl += `?limit=50&dates=${dateInfo.weekStart}-${dateInfo.weekEnd}`;
            
            console.log(`🗓️ ${this.sport.name}: Fetching week games from:`, apiUrl);
            
            const response = await fetch(apiUrl);
            if (!response.ok) {
                throw new Error(`ESPN Week API error: ${response.status}`);
            }
            
            const data = await response.json();
            const weekGames = data.events || [];
            console.log(`✅ ${this.sport.name}: Found ${weekGames.length} games this week`);
            
            return this.parseGames(weekGames);
            
        } catch (error) {
            console.error(`❌ ${this.sport.name}: Week API failed:`, error);
            return this.fetchFallbackGames();
        }
    }

    /**
     * Fetch news for the sport
     */
    async fetchNews() {
        console.log(`📰 Fetching ${this.sport.name} news...`);
        
        try {
            const response = await fetch(this.sport.apis.espn.news);
            
            if (!response.ok) {
                throw new Error(`ESPN News API error: ${response.status}`);
            }
            
            const data = await response.json();
            console.log(`✅ ${this.sport.name}: Successfully fetched ${data.articles?.length || 0} articles`);
            
            return this.parseNews(data.articles || []);
            
        } catch (error) {
            console.error(`❌ ${this.sport.name}: News fetch failed:`, error);
            return [];
        }
    }

    /**
     * Parse games data into standardized format
     */
    parseGames(events) {
        return events.map(event => {
            try {
                const competition = event.competitions[0];
                const homeTeam = competition.competitors.find(c => c.homeAway === 'home');
                const awayTeam = competition.competitors.find(c => c.homeAway === 'away');
                
                return {
                    id: event.id,
                    date: event.date,
                    status: event.status.type.description,
                    statusDetail: event.status.displayClock || '',
                    
                    homeTeam: {
                        id: homeTeam.team.id,
                        name: homeTeam.team.displayName,
                        shortName: homeTeam.team.abbreviation,
                        logo: homeTeam.team.logo,
                        score: homeTeam.score,
                        record: homeTeam.records?.[0]?.summary
                    },
                    
                    awayTeam: {
                        id: awayTeam.team.id,
                        name: awayTeam.team.displayName,
                        shortName: awayTeam.team.abbreviation,
                        logo: awayTeam.team.logo,
                        score: awayTeam.score,
                        record: awayTeam.records?.[0]?.summary
                    },
                    
                    venue: {
                        name: competition.venue?.fullName || 'TBD',
                        city: competition.venue?.address?.city || '',
                        state: competition.venue?.address?.state || ''
                    },
                    
                    broadcasts: competition.broadcasts?.map(b => b.names?.[0]).filter(Boolean) || [],
                    odds: competition.odds?.[0] || null,
                    
                    // Sport-specific data
                    sport: this.sportKey,
                    sportName: this.sport.name
                };
            } catch (error) {
                console.error(`❌ Error parsing game data:`, error);
                return null;
            }
        }).filter(Boolean);
    }

    /**
     * Parse news data into standardized format
     */
    parseNews(articles) {
        return articles.slice(0, 5).map(article => ({
            id: article.id,
            headline: article.headline,
            description: article.description,
            published: article.published,
            link: article.links?.web?.href,
            sport: this.sportKey,
            sportName: this.sport.name
        }));
    }

    /**
     * Fallback API calls for when ESPN fails
     */
    async fetchFallbackGames() {
        console.log(`🔄 ${this.sport.name}: Attempting fallback API...`);
        
        // Implement fallback logic based on sport
        // For now, return empty array
        return [];
    }

    /**
     * Get sport configuration
     */
    getConfig() {
        return this.sport;
    }
}

/**
 * Universal Sports Manager
 * Manages all sports switching and data coordination
 */
class UniversalSportsManager {
    constructor() {
        this.currentSport = 'football'; // Default to NFL
        this.activeFetcher = null;
        this.refreshInterval = null;
        
        console.log('🌟 Universal Sports Manager initialized');
    }

    /**
     * Switch to a new sport
     */
    async switchSport(sportKey) {
        console.log(`🔄 Switching sport to: ${sportKey}`);
        
        // Validate sport exists
        if (!SportsConfig[sportKey]) {
            console.error(`❌ Invalid sport: ${sportKey}`);
            return false;
        }
        
        // Cleanup previous sport
        this.cleanup();
        
        // Set new sport
        this.currentSport = sportKey;
        this.activeFetcher = new SportDataFetcher(sportKey);
        
        // Update UI immediately
        this.updateSportUI(sportKey);
        
        // Load sport data
        await this.loadSportData();
        
        // Setup auto-refresh
        this.setupAutoRefresh();
        
        return true;
    }

    /**
     * Update UI for sport change
     */
    updateSportUI(sportKey) {
        const config = SportsConfig[sportKey];
        
        // Update sport icons
        document.querySelectorAll('.sport-icon').forEach(icon => {
            icon.classList.remove('active');
        });
        
        const activeIcon = document.querySelector(`[data-sport="${sportKey}"]`);
        if (activeIcon) {
            activeIcon.classList.add('active');
        }
        
        // Show loading state with logo
        const container = document.getElementById('game-centric-container');
        if (container) {
            container.innerHTML = `
                <div style="text-align: center; padding: 40px; background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%); border-radius: 16px; box-shadow: 0 8px 32px rgba(0,0,0,0.1);">
                    <div style="margin-bottom: 20px;">
                        <img src="${config.logo}" alt="${config.name}" style="width: 80px; height: 80px; object-fit: contain; animation: bounce 1s infinite;" onerror="this.style.display='none'; this.nextElementSibling.style.display='block';">
                        <div style="font-size: 64px; display: none; animation: bounce 1s infinite;">${config.icon}</div>
                    </div>
                    <h2 style="color: #1e293b; margin-bottom: 12px; font-weight: 600;">Loading ${config.fullName}...</h2>
                    <p style="color: #64748b; margin-bottom: 20px;">Getting live scores, stats, and news from ESPN</p>
                    <div class="loading-spinner" style="width: 40px; height: 40px; margin: 0 auto; border: 4px solid #f3f3f3; border-top: 4px solid ${config.color}; border-radius: 50%; animation: spin 1s linear infinite;"></div>
                </div>
                <style>
                    @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
                    @keyframes bounce { 0%, 20%, 50%, 80%, 100% { transform: translateY(0); } 40% { transform: translateY(-10px); } 60% { transform: translateY(-5px); } }
                </style>
            `;
        }
    }

    /**
     * Load data for current sport
     */
    async loadSportData() {
        if (!this.activeFetcher) return;
        
        try {
            // Load games and news in parallel
            const [games, news] = await Promise.all([
                this.activeFetcher.fetchGames(),
                this.activeFetcher.fetchNews()
            ]);
            
            // Display data
            this.displayGames(games);
            this.displayNews(news);
            
        } catch (error) {
            console.error(`❌ Error loading ${this.currentSport} data:`, error);
            this.showErrorState();
        }
    }

    /**
     * Display games in UI with date awareness
     */
    displayGames(games) {
        const container = document.getElementById('game-centric-container');
        const config = this.activeFetcher.getConfig();
        
        if (!container) return;
        
        if (!games || games.length === 0) {
            container.innerHTML = this.getNoGamesHTML(config);
            return;
        }

        // Get today's date information
        const today = new Date();
        const todayStr = today.toLocaleDateString('en-US', { 
            weekday: 'long', 
            year: 'numeric', 
            month: 'long', 
            day: 'numeric' 
        });

        // Separate today's games from other games
        const todaysGames = [];
        const otherGames = [];
        const todayDate = today.toDateString();

        games.forEach(game => {
            const gameDate = new Date(game.date).toDateString();
            if (gameDate === todayDate) {
                todaysGames.push(game);
            } else {
                otherGames.push(game);
            }
        });

        // Enhanced header with date context
        let html = this.getGamesHeaderHTML(config, games.length, todayStr, todaysGames.length);
        
        // Display today's games first
        if (todaysGames.length > 0) {
            html += `
                <h3 style="color: #1e293b; margin: 24px 0 16px 0; font-weight: 600; display: flex; align-items: center;">
                    <span style="background: ${config.color}; color: white; padding: 4px 8px; border-radius: 12px; font-size: 12px; margin-right: 8px;">TODAY</span>
                    ${todaysGames.length} game${todaysGames.length !== 1 ? 's' : ''} for ${todayStr}
                </h3>
            `;
            html += '<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; margin-bottom: 24px;">';
            todaysGames.forEach(game => {
                html += this.getGameCardHTML(game, config);
            });
            html += '</div>';
        }

        // Display other games if any
        if (otherGames.length > 0) {
            const sectionTitle = todaysGames.length > 0 ? 'Recent & Upcoming Games' : 'Games This Week';
            html += `
                <h3 style="color: #64748b; margin: 24px 0 16px 0; font-weight: 600; display: flex; align-items: center;">
                    <span style="background: #64748b; color: white; padding: 4px 8px; border-radius: 12px; font-size: 12px; margin-right: 8px;">
                        ${todaysGames.length > 0 ? 'OTHER' : 'WEEK'}
                    </span>
                    ${sectionTitle}
                </h3>
            `;
            html += '<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px;">';
            otherGames.slice(0, 6).forEach(game => {
                html += this.getGameCardHTML(game, config);
            });
            html += '</div>';
        }

        container.innerHTML = html;
    }

    /**
     * Display news in UI
     */
    displayNews(articles) {
        const newsContainer = document.getElementById('news-container') || document.getElementById('qb-news-section');
        const config = this.activeFetcher.getConfig();
        
        if (!newsContainer || !articles || articles.length === 0) return;
        
        let html = `
            <div style="text-align: center; margin-bottom: 16px;">
                <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 8px;">
                    <img src="${config.logo}" alt="${config.name}" style="width: 24px; height: 24px; object-fit: contain; margin-right: 8px;" onerror="this.style.display='none'; this.nextElementSibling.style.display='inline';">
                    <span style="display: none; margin-right: 8px;">${config.icon}</span>
                    <h3 style="color: #1e293b; margin: 0; font-size: 20px; font-weight: 600;">Latest ${config.name} News</h3>
                </div>
                <p style="color: #64748b; margin: 4px 0; font-size: 14px;">Breaking news and updates from ESPN</p>
            </div>
        `;
        
        articles.forEach(article => {
            const publishedDate = new Date(article.published).toLocaleDateString('en-US', {
                month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit'
            });
            
            html += `
                <div style="padding: 16px; background: #f8fafc; border-radius: 8px; border-left: 4px solid ${config.color}; margin-bottom: 12px;">
                    <h4 style="margin: 0 0 8px 0; font-size: 16px; font-weight: 600; color: #1e293b;">
                        <a href="${article.link || '#'}" target="_blank" style="text-decoration: none; color: inherit;">
                            ${article.headline}
                        </a>
                    </h4>
                    <p style="margin: 0 0 8px 0; color: #64748b; font-size: 14px;">${article.description || 'No description available'}</p>
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <small style="color: #94a3b8;">${publishedDate}</small>
                        <small style="background: ${config.color}; color: white; padding: 2px 6px; border-radius: 8px; font-size: 10px;">${config.name}</small>
                    </div>
                </div>
            `;
        });
        
        newsContainer.innerHTML = html;
    }

    /**
     * Generate game card HTML
     */
    getGameCardHTML(game, config) {
        const gameTime = new Date(game.date).toLocaleTimeString('en-US', { 
            hour: 'numeric', minute: '2-digit' 
        });
        const statusBadge = this.getStatusBadge(game.status, gameTime, config.color);
        
        return `
            <div class="game-card" data-game-id="${game.id}" data-home-team="${game.homeTeam.name}" data-away-team="${game.awayTeam.name}" style="
                background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%) !important;
                border-radius: 16px !important; padding: 20px !important; box-shadow: 0 8px 32px rgba(0,0,0,0.1) !important;
                border: 1px solid #e2e8f0 !important; transition: all 0.3s ease !important; cursor: pointer !important;
                position: relative !important; overflow: hidden !important; margin: 8px 0 !important;
                display: block !important; width: auto !important; min-height: auto !important; max-height: none !important;
            " onclick="universalSportsManager.selectGame('${game.id}', '${game.homeTeam.name}', '${game.awayTeam.name}')"
               onmouseover="this.style.transform='translateY(-8px)'; this.style.boxShadow='0 20px 60px rgba(0,0,0,0.15)'; this.style.borderColor='${config.color}'"
               onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 8px 32px rgba(0,0,0,0.1)'; this.style.borderColor='#e2e8f0'">
                
                <!-- Selection indicator -->
                <div class="selection-indicator" style="
                    position: absolute !important; top: 0 !important; left: 0 !important; width: 4px !important; height: 100% !important;
                    background: ${config.color} !important; transform: scaleY(0) !important; transition: all 0.3s ease !important;
                    transform-origin: top !important;
                "></div>
                
                <div style="display: flex !important; justify-content: space-between !important; align-items: center !important; margin-bottom: 16px !important;">
                    <div style="font-size: 12px !important; color: #64748b !important;">${new Date(game.date).toLocaleDateString()}</div>
                    ${statusBadge}
                </div>
                
                <div style="display: flex !important; justify-content: space-between !important; align-items: center !important; margin-bottom: 12px !important;">
                    <div style="display: flex !important; align-items: center !important; flex: 1 !important;">
                        <img src="${game.awayTeam.logo}" alt="${game.awayTeam.name}" style="width: 32px !important; height: 32px !important; margin-right: 12px !important;" onerror="this.style.display='none'">
                        <div style="font-weight: 700 !important; color: #1e293b !important; font-size: 16px !important;">${game.awayTeam.name}</div>
                    </div>
                    <div style="font-size: 24px !important; font-weight: 800 !important; color: #1e293b !important;">${game.awayTeam.score}</div>
                </div>
                
                <div style="display: flex !important; justify-content: space-between !important; align-items: center !important; margin-bottom: 16px !important;">
                    <div style="display: flex !important; align-items: center !important; flex: 1 !important;">
                        <img src="${game.homeTeam.logo}" alt="${game.homeTeam.name}" style="width: 32px !important; height: 32px !important; margin-right: 12px !important;" onerror="this.style.display='none'">
                        <div style="font-weight: 700 !important; color: #1e293b !important; font-size: 16px !important;">${game.homeTeam.name}</div>
                    </div>
                    <div style="font-size: 24px !important; font-weight: 800 !important; color: #1e293b !important;">${game.homeTeam.score}</div>
                </div>
                
                ${config.displaySettings.showVenue ? `
                    <div style="text-align: center !important; padding-top: 12px !important; border-top: 1px solid #e2e8f0 !important; font-size: 12px !important; color: #64748b !important;">
                        📍 ${game.venue.name}
                    </div>
                ` : ''}
                
                <!-- Click hint -->
                <div style="
                    position: absolute !important; bottom: 8px !important; right: 12px !important; 
                    background: rgba(0,0,0,0.05) !important; color: #64748b !important; 
                    padding: 4px 8px !important; border-radius: 8px !important; font-size: 11px !important;
                    opacity: 0.7 !important; transition: all 0.3s ease !important;
                ">
                    Click for team news
                </div>
            </div>
        `;
    }

    /**
     * Generate status badge
     */
    getStatusBadge(status, gameTime, color) {
        if (status.includes('Final')) {
            return `<span style="background: #10b981; color: white; padding: 2px 8px; border-radius: 12px; font-size: 11px;">FINAL</span>`;
        } else if (status.includes('Live') || status.includes('In Progress')) {
            return `<span style="background: #ef4444; color: white; padding: 2px 8px; border-radius: 12px; font-size: 11px; animation: pulse 2s infinite;">LIVE</span>`;
        } else {
            return `<span style="background: ${color}; color: white; padding: 2px 8px; border-radius: 12px; font-size: 11px;">${gameTime}</span>`;
        }
    }

    /**
     * Generate header HTML for games section with date awareness
     */
    getGamesHeaderHTML(config, gameCount, todayStr, todaysGameCount) {
        const today = todayStr || new Date().toLocaleDateString('en-US', {
            weekday: 'long', month: 'long', day: 'numeric'
        });

        // Create status message based on today's games
        let statusMessage;
        if (todaysGameCount > 0) {
            statusMessage = `${today} • ${todaysGameCount} game${todaysGameCount !== 1 ? 's' : ''} today • ${gameCount} total`;
        } else {
            statusMessage = `${today} • No games today • ${gameCount} recent/upcoming`;
        }

        return `
            <div style="text-align: center; margin-bottom: 24px; background: linear-gradient(135deg, ${config.color} 0%, ${config.secondaryColor} 100%); padding: 24px; border-radius: 16px; color: white; position: relative; overflow: hidden;">
                <div style="position: absolute; top: 16px; right: 16px; opacity: 0.2;">
                    <img src="${config.logo}" alt="${config.name}" style="width: 60px; height: 60px; object-fit: contain;" onerror="this.style.display='none';">
                </div>
                <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 8px; position: relative; z-index: 2;">
                    <img src="${config.logo}" alt="${config.name}" style="width: 48px; height: 48px; object-fit: contain; margin-right: 12px;" onerror="this.style.display='none'; this.nextElementSibling.style.display='inline';">
                    <span style="display: none; font-size: 32px; margin-right: 12px;">${config.icon}</span>
                    <h2 style="color: white; margin: 0; font-size: 28px; font-weight: 700;">${config.fullName}</h2>
                </div>
                <p style="color: rgba(255,255,255,0.9); margin: 0; font-size: 16px; position: relative; z-index: 2;">${statusMessage}</p>
                ${todaysGameCount > 0 ? 
                    `<div style="margin-top: 12px; position: relative; z-index: 2;">
                        <span style="background: rgba(255,255,255,0.2); color: white; padding: 6px 12px; border-radius: 20px; font-size: 14px; font-weight: 600;">
                            🔴 LIVE • Updated from ESPN
                        </span>
                    </div>` : 
                    `<div style="margin-top: 12px; position: relative; z-index: 2;">
                        <span style="background: rgba(255,255,255,0.15); color: white; padding: 6px 12px; border-radius: 20px; font-size: 14px; font-weight: 600;">
                            📅 Recent & Upcoming Games
                        </span>
                    </div>`
                }
            </div>
        `;
    }

    /**
     * Generate no games HTML with current date
     */
    getNoGamesHTML(config) {
        const today = new Date().toLocaleDateString('en-US', { 
            weekday: 'long', 
            year: 'numeric', 
            month: 'long', 
            day: 'numeric' 
        });

        return `
            <div style="text-align: center; padding: 60px 40px; background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); border-radius: 16px;">
                <div style="margin-bottom: 24px;">
                    <img src="${config.logo}" alt="${config.name}" style="width: 120px; height: 120px; object-fit: contain; opacity: 0.8;" onerror="this.style.display='none'; this.nextElementSibling.style.display='block';">
                    <div style="font-size: 80px; display: none;">${config.icon}</div>
                </div>
                <h3 style="color: #1e293b; margin-bottom: 12px;">No ${config.name} Games Today</h3>
                <p style="color: #64748b; margin-bottom: 8px;">${today}</p>
                <p style="color: #64748b;">Check back later for upcoming games!</p>
                <div style="margin-top: 20px; padding: 12px 20px; background: rgba(59, 130, 246, 0.1); border-radius: 8px; display: inline-block;">
                    <span style="color: #2563eb; font-weight: 500;">📅 Games refresh automatically throughout the day</span>
                </div>
            </div>
        `;
    }

    /**
     * Setup auto-refresh for live data with date awareness
     */
    setupAutoRefresh() {
        if (this.refreshInterval) {
            clearInterval(this.refreshInterval);
        }
        
        const config = this.activeFetcher.getConfig();
        
        // Refresh every 15 minutes to check for new games
        this.refreshInterval = setInterval(() => {
            console.log(`🔄 Auto-refreshing ${config.name} data for current date...`);
            this.loadSportData();
            this.showRefreshIndicator();
        }, 15 * 60 * 1000); // 15 minutes for live updates
        
        console.log(`⏰ Auto-refresh configured for ${config.name}: Every 15 minutes`);
    }

    /**
     * Show brief refresh indicator
     */
    showRefreshIndicator() {
        const indicator = document.createElement('div');
        indicator.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: #22c55e;
            color: white;
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: 600;
            z-index: 10000;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            transform: translateY(-100%);
            transition: transform 0.3s ease;
        `;
        indicator.innerHTML = '✅ Games Updated';
        
        document.body.appendChild(indicator);
        
        // Animate in
        setTimeout(() => {
            indicator.style.transform = 'translateY(0)';
        }, 100);
        
        // Remove after 2 seconds
        setTimeout(() => {
            indicator.style.transform = 'translateY(-100%)';
            setTimeout(() => {
                if (indicator.parentNode) {
                    indicator.parentNode.removeChild(indicator);
                }
            }, 300);
        }, 2000);
    }

    /**
     * Select a specific game and load team news
     */
    async selectGame(gameId, homeTeam, awayTeam) {
        console.log('Game selected:', gameId, homeTeam, 'vs', awayTeam);
        
        // Find and store the selected game data
        const gameData = await this.activeFetcher.fetchGames();
        const selectedGame = gameData.find(g => g.id === gameId);
        this.selectedGame = selectedGame; // Store for later use
        
        // Update visual selection
        this.updateGameSelection(gameId);
        
        // Load team-specific news
        await this.loadTeamNews([homeTeam, awayTeam]);
        
        // Hide game container and show predictions panel
        const gameContainer = document.getElementById('game-centric-container');
        if (gameContainer) {
            gameContainer.style.display = 'none'; // Hide the game cards
        }
        
        const predictionsPanel = document.getElementById('predictions-panel');
        if (predictionsPanel) {
            predictionsPanel.style.display = 'block';
            predictionsPanel.classList.add('active'); // Add active class to make it visible
            
            // Scroll to predictions panel smoothly
            predictionsPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
            
            // Add a navigation hint at the top of predictions panel
            const existingHint = predictionsPanel.querySelector('.matchup-nav-hint');
            if (!existingHint) {
                const navHint = document.createElement('div');
                navHint.className = 'matchup-nav-hint';
                navHint.style.cssText = 'background: linear-gradient(135deg, #6366f1, #8b5cf6); color: white; padding: 16px; border-radius: 8px; margin-bottom: 20px; text-align: center; box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);';
                navHint.innerHTML = `
                    <button onclick="universalSportsManager.backToGames()" style="position: absolute; left: 16px; top: 50%; transform: translateY(-50%); background: rgba(255,255,255,0.2); border: none; color: white; padding: 8px 16px; border-radius: 6px; cursor: pointer; font-size: 14px; display: flex; align-items: center; gap: 4px;">
                        <i class="material-icons" style="font-size: 18px;">arrow_back</i> Back to Games
                    </button>
                    <div style="font-size: 18px; font-weight: 600; margin-bottom: 8px;">
                        📊 ${homeTeam} vs ${awayTeam} - Model Projections
                    </div>
                    <div style="font-size: 14px; opacity: 0.9;">
                        View AI predictions below, then <a href="#news-container" style="color: #fbbf24; text-decoration: underline; cursor: pointer;" onclick="document.getElementById('news-container').scrollIntoView({behavior: 'smooth'})">scroll down for news</a>
                    </div>
                `;
                navHint.style.position = 'relative'; // For absolute positioning of back button
                predictionsPanel.insertBefore(navHint, predictionsPanel.firstChild);
            }
        }
        
        // Also trigger model projections/betting insights update
        if (typeof updateBettingInsights === 'function') {
            updateBettingInsights();
        }
        
        // Show prediction for this specific game
        if (selectedGame) {
            showPrediction(selectedGame);
        }
    }

    /**
     * Update visual selection of game cards
     */
    updateGameSelection(gameId) {
        // Remove previous selections
        document.querySelectorAll('.game-card').forEach(card => {
            card.style.borderColor = '#e2e8f0';
            const indicator = card.querySelector('.selection-indicator');
            if (indicator) {
                indicator.style.transform = 'scaleY(0)';
            }
        });
        
        // Highlight selected card
        const selectedCard = document.querySelector(`[data-game-id="${gameId}"]`);
        if (selectedCard) {
            const config = this.activeFetcher.getConfig();
            selectedCard.style.borderColor = config.color;
            selectedCard.style.borderWidth = '2px';
            
            const indicator = selectedCard.querySelector('.selection-indicator');
            if (indicator) {
                indicator.style.transform = 'scaleY(1)';
            }
        }
    }

    /**
     * Return to games view from prediction panel
     */
    backToGames() {
        // Hide predictions panel
        const predictionsPanel = document.getElementById('predictions-panel');
        if (predictionsPanel) {
            predictionsPanel.style.display = 'none';
            predictionsPanel.classList.remove('active');
            
            // Remove the navigation hint
            const navHint = predictionsPanel.querySelector('.matchup-nav-hint');
            if (navHint) {
                navHint.remove();
            }
        }
        
        // Show game container
        const gameContainer = document.getElementById('game-centric-container');
        if (gameContainer) {
            gameContainer.style.display = 'block';
            
            // Scroll to game container
            gameContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
        
        // Clear selection
        this.updateGameSelection(null);
    }

    /**
     * Load news for specific teams
     */
    async loadTeamNews(teams) {
        if (!this.activeFetcher) return;
        
        console.log('Loading news for teams:', teams);
        
        try {
            // For now, we'll show sport news filtered for the teams
            // In the future, this could be enhanced with team-specific API calls
            const allNews = await this.activeFetcher.fetchNews();
            const config = this.activeFetcher.getConfig();
            
            // Filter news that mentions the team names
            const teamNews = allNews.filter(article => {
                const content = `${article.headline} ${article.description}`.toLowerCase();
                return teams.some(team => content.includes(team.toLowerCase()));
            });
            
            this.displayTeamNews(teamNews.length > 0 ? teamNews : allNews.slice(0, 3), teams, config);
            
        } catch (error) {
            console.error('Error loading team news:', error);
            this.showNewsError(teams);
        }
    }

    /**
     * Display team-specific news
     */
    displayTeamNews(articles, teams, config) {
        const newsContainer = document.getElementById('news-container') || document.getElementById('qb-news-section');
        
        if (!newsContainer) return;
        
        let html = `
            <div style="text-align: center; margin-bottom: 20px; background: linear-gradient(135deg, ${config.color}20, ${config.secondaryColor}20); padding: 20px; border-radius: 12px; border: 2px solid ${config.color}40;">
                <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 8px;">
                    <img src="${config.logo}" alt="${config.name}" style="width: 24px; height: 24px; object-fit: contain; margin-right: 8px;" onerror="this.style.display='none'; this.nextElementSibling.style.display='inline';">
                    <span style="display: none; margin-right: 8px;">${config.icon}</span>
                    <h3 style="color: #1e293b; margin: 0; font-size: 20px; font-weight: 600;">Team News: ${teams.join(' vs ')}</h3>
                </div>
                <p style="color: #64748b; margin: 4px 0; font-size: 14px;">Latest news and updates for your selected matchup</p>
            </div>
        `;
        
        if (articles.length === 0) {
            html += `
                <div style="text-align: center; padding: 30px; background: #f8fafc; border-radius: 12px; border: 2px dashed #d1d5db;">
                    <div style="font-size: 32px; margin-bottom: 12px;">📰</div>
                    <h4 style="color: #374151; margin-bottom: 8px;">No Specific Team News Found</h4>
                    <p style="color: #6b7280; font-size: 14px;">Check back later for updates on ${teams.join(' and ')}</p>
                </div>
            `;
        } else {
            articles.forEach((article, index) => {
                const publishedDate = new Date(article.published).toLocaleDateString('en-US', {
                    month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit'
                });
                
                // Highlight team names in headlines
                let highlightedHeadline = article.headline;
                teams.forEach(team => {
                    const regex = new RegExp(`\\b${team}\\b`, 'gi');
                    highlightedHeadline = highlightedHeadline.replace(regex, `<span style="background: ${config.color}20; padding: 2px 4px; border-radius: 4px; font-weight: 700;">$&</span>`);
                });
                
                html += `
                    <div style="padding: 18px; background: #ffffff; border-radius: 12px; border: 1px solid #e5e7eb; margin-bottom: 12px; transition: all 0.2s ease; box-shadow: 0 2px 8px rgba(0,0,0,0.05);" onmouseover="this.style.boxShadow='0 8px 32px rgba(0,0,0,0.1)'; this.style.transform='translateY(-2px)'" onmouseout="this.style.boxShadow='0 2px 8px rgba(0,0,0,0.05)'; this.style.transform='translateY(0)'">
                        <div style="display: flex; align-items: center; margin-bottom: 8px;">
                            <span style="background: ${config.color}; color: white; padding: 2px 8px; border-radius: 12px; font-size: 10px; font-weight: 600; margin-right: 8px;">${config.name}</span>
                            ${index === 0 ? '<span style="background: #ef4444; color: white; padding: 2px 8px; border-radius: 12px; font-size: 10px; font-weight: 600;">FEATURED</span>' : ''}
                        </div>
                        <h4 style="margin: 0 0 10px 0; font-size: 16px; font-weight: 600; color: #1e293b; line-height: 1.4;">
                            <a href="${article.link || '#'}" target="_blank" style="text-decoration: none; color: inherit;" onmouseover="this.style.color='${config.color}'" onmouseout="this.style.color='#1e293b'">
                                ${highlightedHeadline}
                            </a>
                        </h4>
                        <p style="margin: 0 0 10px 0; color: #64748b; font-size: 14px; line-height: 1.5;">
                            ${article.description || 'No description available'}
                        </p>
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <small style="color: #94a3b8; font-weight: 500;">📅 ${publishedDate}</small>
                            <div style="display: flex; gap: 8px;">
                                ${teams.map(team => {
                                    const isRelevant = (article.headline + article.description).toLowerCase().includes(team.toLowerCase());
                                    return `<span style="background: ${isRelevant ? config.color : '#f3f4f6'}; color: ${isRelevant ? 'white' : '#6b7280'}; padding: 1px 6px; border-radius: 8px; font-size: 10px; font-weight: 500;">${team}</span>`;
                                }).join('')}
                            </div>
                        </div>
                    </div>
                `;
            });
        }
        
        newsContainer.innerHTML = html;
    }

    /**
     * Show news error
     */
    showNewsError(teams) {
        const newsContainer = document.getElementById('news-container') || document.getElementById('qb-news-section');
        if (newsContainer) {
            newsContainer.innerHTML = `
                <div style="text-align: center; padding: 30px; background: #fef2f2; border-radius: 12px; border: 1px solid #fecaca;">
                    <div style="font-size: 32px; margin-bottom: 12px;">⚠️</div>
                    <h4 style="color: #dc2626; margin-bottom: 8px;">Unable to load team news</h4>
                    <p style="color: #991b1b; font-size: 14px;">Please try again later for ${teams.join(' vs ')} updates</p>
                </div>
            `;
        }
    }

    /**
     * Cleanup when switching sports
     */
    cleanup() {
        if (this.refreshInterval) {
            clearInterval(this.refreshInterval);
            this.refreshInterval = null;
        }
        
        this.activeFetcher = null;
    }

    /**
     * Show error state
     */
    showErrorState() {
        const container = document.getElementById('game-centric-container');
        const config = this.activeFetcher?.getConfig();
        
        if (container && config) {
            container.innerHTML = `
                <div style="text-align: center; padding: 40px; background: #fef2f2; border-radius: 16px; border: 1px solid #fecaca;">
                    <div style="margin-bottom: 16px;">
                        <img src="${config.logo}" alt="${config.name}" style="width: 64px; height: 64px; object-fit: contain; opacity: 0.6;" onerror="this.style.display='none'; this.nextElementSibling.style.display='block';">
                        <div style="font-size: 48px; display: none;">⚠️</div>
                    </div>
                    <h3 style="color: #dc2626; margin-bottom: 12px;">Unable to load ${config.name} data</h3>
                    <p style="color: #991b1b; margin-bottom: 16px;">Please try again later</p>
                    <button onclick="universalSportsManager.loadSportData()" style="padding: 8px 16px; background: #dc2626; color: white; border: none; border-radius: 6px; cursor: pointer;">
                        Retry
                    </button>
                </div>
            `;
        }
    }
}

// Global instances
window.SportsConfig = SportsConfig;
window.SportDataFetcher = SportDataFetcher;
window.universalSportsManager = new UniversalSportsManager();

// Global helper function for ML predictions
function showPrediction(game) {
    console.log('🎯 Showing prediction for game:', game.id);
    
    // Get predictions panel
    const predictionsPanel = document.getElementById('predictions-panel');
    if (!predictionsPanel) {
        console.warn('Predictions panel not found');
        return;
    }
    
    // Make sure the panel is visible
    predictionsPanel.style.display = 'block';
    
    if (!game) {
        console.warn('Game data not provided');
        return;
    }
    
    // Trigger ESPN model data fetch if available
    if (typeof enhancedBettingInsights !== 'undefined' && enhancedBettingInsights.updateBettingInsightsUI) {
        console.log('🤖 Loading ESPN model projections...');
        
        // Extract team IDs from game data
        const homeTeamId = game.homeTeam?.id;
        const awayTeamId = game.awayTeam?.id;
        
        if (!homeTeamId || !awayTeamId) {
            console.warn('Could not extract team IDs from game data');
            console.log('Game structure:', game);
            return;
        }
        
        // Fetch and display predictions for the matchup
        // Using homeTeamId as primary team, awayTeamId as opponent
        enhancedBettingInsights.getBettingInsights(
            homeTeamId,     // teamId (home team)
            'TEAM',         // position (team-level prediction)
            'game_winner',  // predictionType
            homeTeamId,     // teamId again
            awayTeamId      // opponentId
        ).then(insights => {
            if (insights) {
                enhancedBettingInsights.updateBettingInsightsUI(insights);
                console.log('✅ ESPN model projections loaded');
            }
        }).catch(error => {
            console.error('Error loading ESPN projections:', error);
        });
    } else {
        console.log('ℹ️ Enhanced betting insights not available, showing static predictions');
    }
}

console.log('🏆 Universal Sports Configuration System loaded successfully');