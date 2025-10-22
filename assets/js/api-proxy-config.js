/**
 * API Proxy Configuration
 * Routes all Tank01 API calls through backend to avoid CORS issues
 */

const API_PROXY_CONFIG = {
    // Backend proxy URL
    BACKEND_URL: window.location.hostname === 'localhost' 
        ? 'http://localhost:5000'
        : 'https://betyard-ml-backend.onrender.com',
    
    // Proxy endpoints
    ENDPOINTS: {
        TANK01_GENERIC: '/api/proxy/tank01',
        NFL_SCHEDULE: '/api/proxy/nfl/schedule',
        NFL_ROSTER: '/api/proxy/nfl/roster',
        NFL_GAMES: '/api/proxy/nfl/games'
    },
    
    // Original Tank01 API config (for reference)
    TANK01_DIRECT: {
        BASE_URL: 'https://tank01-nfl-live-in-game-real-time-statistics-nfl.p.rapidapi.com',
        API_KEY: 'be76a86cb3msh01c346c2b0ef4ffp151e0djsn0b0e85e00bd3'
    }
};

/**
 * Proxy wrapper for Tank01 API calls
 * Automatically routes through backend proxy
 */
class Tank01Proxy {
    constructor() {
        this.backendUrl = API_PROXY_CONFIG.BACKEND_URL;
        console.log(`🔄 Tank01 Proxy initialized: ${this.backendUrl}`);
    }
    
    /**
     * Generic proxy call to any Tank01 endpoint
     * @param {string} endpoint - Tank01 endpoint name (e.g., 'getNFLTeamSchedule')
     * @param {object} params - Query parameters
     */
    async call(endpoint, params = {}) {
        try {
            // Build proxy URL with endpoint parameter
            const url = new URL(`${this.backendUrl}${API_PROXY_CONFIG.ENDPOINTS.TANK01_GENERIC}`);
            url.searchParams.append('endpoint', endpoint);
            
            // Add all other parameters
            Object.keys(params).forEach(key => {
                url.searchParams.append(key, params[key]);
            });
            
            console.log(`📡 Proxy request: ${endpoint}`, params);
            
            const response = await fetch(url.toString(), {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            if (!response.ok) {
                throw new Error(`Proxy error: ${response.status} ${response.statusText}`);
            }
            
            const data = await response.json();
            console.log(`✅ Proxy response: ${endpoint}`, data);
            return data;
            
        } catch (error) {
            console.error(`❌ Proxy error for ${endpoint}:`, error);
            throw error;
        }
    }
    
    /**
     * Get NFL team schedule
     */
    async getTeamSchedule(teamAbv, season = '2024') {
        const url = `${this.backendUrl}${API_PROXY_CONFIG.ENDPOINTS.NFL_SCHEDULE}?teamAbv=${teamAbv}&season=${season}`;
        const response = await fetch(url);
        if (!response.ok) throw new Error(`Schedule fetch failed: ${response.status}`);
        return await response.json();
    }
    
    /**
     * Get NFL team roster
     */
    async getTeamRoster(teamID, getStats = false) {
        const url = `${this.backendUrl}${API_PROXY_CONFIG.ENDPOINTS.NFL_ROSTER}?teamID=${teamID}&getStats=${getStats}`;
        const response = await fetch(url);
        if (!response.ok) throw new Error(`Roster fetch failed: ${response.status}`);
        return await response.json();
    }
    
    /**
     * Get NFL games for week
     */
    async getGamesForWeek(week = 'all', season = '2024') {
        const url = `${this.backendUrl}${API_PROXY_CONFIG.ENDPOINTS.NFL_GAMES}?gameWeek=${week}&season=${season}`;
        const response = await fetch(url);
        if (!response.ok) throw new Error(`Games fetch failed: ${response.status}`);
        return await response.json();
    }
}

// Create global proxy instance
window.Tank01Proxy = new Tank01Proxy();

console.log('🔧 API Proxy configured:', {
    backend: API_PROXY_CONFIG.BACKEND_URL,
    environment: window.location.hostname === 'localhost' ? 'LOCAL' : 'PRODUCTION'
});
