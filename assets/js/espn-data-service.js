// ESPN Data Service - Replaces Tank01 with ESPN API
// Provides news, matchups, player info, and more via your ML backend

class ESPNDataService {
    constructor() {
        // Use the same ML backend URL configuration
        this.baseURL = this.getMLBackendURL();
        this.cache = new Map();
        this.cacheTimeout = 5 * 60 * 1000; // 5 minutes
        
        console.log('🏈 ESPN Data Service initialized:', this.baseURL);
    }
    
    getMLBackendURL() {
        // Use existing ML_CONFIG if available
        if (window.ML_CONFIG) {
            const activeProvider = window.ML_CONFIG.ACTIVE;
            return window.ML_CONFIG[activeProvider];
        }
        
        // Fallback detection
        const isLocal = window.location.hostname === 'localhost' || 
                       window.location.hostname === '127.0.0.1';
        return isLocal ? 'http://localhost:5001' : 'https://betyard-ml-backend.onrender.com';
    }
    
    // Helper method for cached API calls
    async cachedFetch(url, options = {}) {
        const cacheKey = url + JSON.stringify(options);
        const cached = this.cache.get(cacheKey);
        
        if (cached && Date.now() - cached.timestamp < this.cacheTimeout) {
            console.log('📦 Cache hit:', url);
            return cached.data;
        }
        
        try {
            const response = await fetch(url, {
                timeout: 10000,
                ...options
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();
            
            // Cache successful responses
            this.cache.set(cacheKey, {
                data,
                timestamp: Date.now()
            });
            
            return data;
        } catch (error) {
            console.error('ESPN API Error:', error);
            throw error;
        }
    }
    
    // Get latest NFL news
    async getLatestNews(limit = 10) {
        const url = `${this.baseURL}/api/news/latest?limit=${limit}`;
        return await this.cachedFetch(url);
    }
    
    // Get current week matchups
    async getTeamMatchups(week = null) {
        const url = week 
            ? `${this.baseURL}/api/teams/matchups?week=${week}`
            : `${this.baseURL}/api/teams/matchups`;
        return await this.cachedFetch(url);
    }
    
    // Search for players
    async searchPlayers(query, limit = 10) {
        const url = `${this.baseURL}/api/players/search?q=${encodeURIComponent(query)}&limit=${limit}`;
        return await this.cachedFetch(url);
    }
    
    // Get team information
    async getTeamInfo(teamId) {
        const url = `${this.baseURL}/api/teams/info/${teamId}`;
        return await this.cachedFetch(url);
    }
    
    // Get injury reports
    async getCurrentInjuries(teamId = null) {
        const url = teamId 
            ? `${this.baseURL}/api/injuries/current?team_id=${teamId}`
            : `${this.baseURL}/api/injuries/current`;
        return await this.cachedFetch(url);
    }
    
    // Get weekly schedule
    async getWeeklySchedule(week = null) {
        const url = week 
            ? `${this.baseURL}/api/schedule/week?week=${week}`
            : `${this.baseURL}/api/schedule/week`;
        return await this.cachedFetch(url);
    }
    
    // Get current standings
    async getCurrentStandings() {
        const url = `${this.baseURL}/api/standings/current`;
        return await this.cachedFetch(url);
    }
    
    // Get trending players
    async getTrendingPlayers(limit = 10) {
        const url = `${this.baseURL}/api/players/trending?limit=${limit}`;
        return await this.cachedFetch(url);
    }
    
    // Tank01 compatibility methods
    async getTank01TeamStats(teamId) {
        const url = `${this.baseURL}/api/tank01/team-stats/${teamId}`;
        return await this.cachedFetch(url);
    }
    
    async getTank01PlayerLogs(playerId) {
        const url = `${this.baseURL}/api/tank01/player-game-logs/${playerId}`;
        return await this.cachedFetch(url);
    }
}

// Initialize ESPN Data Service
window.ESPNDataService = new ESPNDataService();

// Export for modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ESPNDataService;
}