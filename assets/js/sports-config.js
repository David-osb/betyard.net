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
     * Fetch games/scores for the sport
     */
    async fetchGames() {
        console.log(`🎮 Fetching ${this.sport.name} games...`);
        
        try {
            // Primary ESPN API call
            const response = await fetch(this.sport.apis.espn.scoreboard);
            
            if (!response.ok) {
                throw new Error(`ESPN API error: ${response.status}`);
            }
            
            const data = await response.json();
            console.log(`✅ ${this.sport.name}: Successfully fetched ${data.events?.length || 0} games`);
            
            return this.parseGames(data.events || []);
            
        } catch (error) {
            console.error(`❌ ${this.sport.name}: Primary API failed:`, error);
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
        
        // Show loading state
        const container = document.getElementById('game-centric-container');
        if (container) {
            container.innerHTML = `
                <div style="text-align: center; padding: 40px; background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%); border-radius: 16px; box-shadow: 0 8px 32px rgba(0,0,0,0.1);">
                    <div style="font-size: 64px; margin-bottom: 20px; animation: bounce 1s infinite;">${config.icon}</div>
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
     * Display games in UI
     */
    displayGames(games) {
        const container = document.getElementById('game-centric-container');
        const config = this.activeFetcher.getConfig();
        
        if (!container) return;
        
        if (!games || games.length === 0) {
            container.innerHTML = this.getNoGamesHTML(config);
            return;
        }
        
        let html = this.getGamesHeaderHTML(config, games.length);
        html += '<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; margin-top: 20px;">';
        
        games.slice(0, config.displaySettings.maxGamesPerPage).forEach(game => {
            html += this.getGameCardHTML(game, config);
        });
        
        html += '</div>';
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
                <h3 style="color: #1e293b; margin: 0; font-size: 20px; font-weight: 600;">${config.icon} Latest ${config.name} News</h3>
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
            <div style="
                background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
                border-radius: 16px; padding: 20px; box-shadow: 0 8px 32px rgba(0,0,0,0.1);
                border: 1px solid #e2e8f0; transition: all 0.3s ease;
            " onmouseover="this.style.transform='translateY(-4px)'" onmouseout="this.style.transform='translateY(0)'">
                
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
                    <div style="font-size: 12px; color: #64748b;">${new Date(game.date).toLocaleDateString()}</div>
                    ${statusBadge}
                </div>
                
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <div style="display: flex; align-items: center; flex: 1;">
                        <img src="${game.awayTeam.logo}" alt="${game.awayTeam.name}" style="width: 32px; height: 32px; margin-right: 12px;" onerror="this.style.display='none'">
                        <div style="font-weight: 700; color: #1e293b;">${game.awayTeam.name}</div>
                    </div>
                    <div style="font-size: 24px; font-weight: 800; color: #1e293b;">${game.awayTeam.score}</div>
                </div>
                
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
                    <div style="display: flex; align-items: center; flex: 1;">
                        <img src="${game.homeTeam.logo}" alt="${game.homeTeam.name}" style="width: 32px; height: 32px; margin-right: 12px;" onerror="this.style.display='none'">
                        <div style="font-weight: 700; color: #1e293b;">${game.homeTeam.name}</div>
                    </div>
                    <div style="font-size: 24px; font-weight: 800; color: #1e293b;">${game.homeTeam.score}</div>
                </div>
                
                ${config.displaySettings.showVenue ? `
                    <div style="text-align: center; padding-top: 12px; border-top: 1px solid #e2e8f0; font-size: 12px; color: #64748b;">
                        📍 ${game.venue.name}
                    </div>
                ` : ''}
                
                ${config.features.hasMLPredictions ? `
                    <div style="text-align: center; margin-top: 12px;">
                        <button onclick="showPrediction('${game.id}')" style="
                            background: ${config.color}; color: white; border: none; 
                            padding: 6px 12px; border-radius: 6px; font-size: 11px; cursor: pointer;
                        ">🎯 ML Prediction</button>
                    </div>
                ` : ''}
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
     * Generate header HTML for games section
     */
    getGamesHeaderHTML(config, gameCount) {
        const today = new Date().toLocaleDateString('en-US', {
            weekday: 'long', month: 'long', day: 'numeric'
        });

        return `
            <div style="text-align: center; margin-bottom: 24px; background: linear-gradient(135deg, ${config.color} 0%, ${config.secondaryColor} 100%); padding: 24px; border-radius: 16px; color: white;">
                <h2 style="color: white; margin: 0 0 8px 0; font-size: 28px; font-weight: 700;">${config.icon} ${config.fullName}</h2>
                <p style="color: rgba(255,255,255,0.9); margin: 0; font-size: 16px;">${today} • ${gameCount} Games • Live ESPN Data</p>
            </div>
        `;
    }

    /**
     * Generate no games HTML
     */
    getNoGamesHTML(config) {
        return `
            <div style="text-align: center; padding: 60px 40px; background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); border-radius: 16px;">
                <div style="font-size: 80px; margin-bottom: 24px;">${config.icon}</div>
                <h3 style="color: #1e293b; margin-bottom: 12px;">No ${config.name} Games Today</h3>
                <p style="color: #64748b;">Check back tomorrow for exciting matchups!</p>
            </div>
        `;
    }

    /**
     * Setup auto-refresh for live data
     */
    setupAutoRefresh() {
        if (this.refreshInterval) {
            clearInterval(this.refreshInterval);
        }
        
        const config = this.activeFetcher.getConfig();
        this.refreshInterval = setInterval(() => {
            console.log(`🔄 Auto-refreshing ${config.name} data...`);
            this.loadSportData();
        }, config.displaySettings.refreshInterval);
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
                    <div style="font-size: 48px; margin-bottom: 16px;">⚠️</div>
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
function showPrediction(gameId) {
    console.log('🎯 Showing prediction for game:', gameId);
    // This will connect to your ML backend
}

console.log('🏆 Universal Sports Configuration System loaded successfully');