/**
 * ESPN Odds Service - Real Sportsbook Odds Integration
 * Fetches live betting odds from ESPN's API including:
 * - Moneyline, Spread, Over/Under
 * - Multiple sportsbook providers
 * - Live odds updates
 * - Historical odds movement
 */

class ESPNOddsService {
    constructor() {
        this.baseURL = 'https://sports.core.api.espn.com/v2/sports/football/leagues/nfl';
        this.cache = new Map();
        this.cacheDuration = 60000; // 1 minute cache
    }

    /**
     * Fetch odds for a specific game
     * @param {string} eventId - ESPN event ID
     * @returns {Promise<Object>} Odds data from all providers
     */
    async getGameOdds(eventId) {
        const cacheKey = `odds_${eventId}`;
        
        // Check cache
        if (this.cache.has(cacheKey)) {
            const cached = this.cache.get(cacheKey);
            if (Date.now() - cached.timestamp < this.cacheDuration) {
                console.log('📊 Using cached odds for event:', eventId);
                return cached.data;
            }
        }

        try {
            const url = `${this.baseURL}/events/${eventId}/competitions/${eventId}/odds`;
            console.log('📡 Fetching ESPN odds:', url);
            
            const response = await fetch(url);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            
            // Parse odds from all providers
            const oddsData = {
                eventId,
                providers: [],
                lastUpdate: new Date().toISOString()
            };

            if (data.items && data.items.length > 0) {
                for (const item of data.items) {
                    const provider = await this.parseProviderOdds(item);
                    if (provider) {
                        oddsData.providers.push(provider);
                    }
                }
            }

            // Cache the result
            this.cache.set(cacheKey, {
                data: oddsData,
                timestamp: Date.now()
            });

            console.log('✅ ESPN odds fetched:', oddsData);
            return oddsData;

        } catch (error) {
            console.error('❌ Failed to fetch ESPN odds:', error);
            return {
                eventId,
                providers: [],
                error: error.message
            };
        }
    }

    /**
     * Parse odds from a single provider
     * @param {Object} providerData - Raw provider data from ESPN
     * @returns {Promise<Object>} Parsed odds data
     */
    async parseProviderOdds(providerData) {
        try {
            // Fetch full provider details
            const response = await fetch(providerData.$ref);
            const data = await response.json();

            return {
                id: data.provider?.id || 'unknown',
                name: data.provider?.name || 'Unknown',
                priority: data.provider?.priority || 0,
                spread: data.spread,
                overUnder: data.overUnder,
                spreadOdds: data.spreadOdds,
                overOdds: data.overOdds,
                underOdds: data.underOdds,
                homeTeam: {
                    favorite: data.homeTeamOdds?.favorite,
                    moneyLine: data.homeTeamOdds?.moneyLine,
                    spreadOdds: data.homeTeamOdds?.spreadOdds,
                    current: data.homeTeamOdds?.current,
                    open: data.homeTeamOdds?.open
                },
                awayTeam: {
                    favorite: data.awayTeamOdds?.favorite,
                    moneyLine: data.awayTeamOdds?.moneyLine,
                    spreadOdds: data.awayTeamOdds?.spreadOdds,
                    current: data.awayTeamOdds?.current,
                    open: data.awayTeamOdds?.open
                },
                details: data.details,
                lastUpdate: new Date().toISOString()
            };
        } catch (error) {
            console.error('❌ Failed to parse provider odds:', error);
            return null;
        }
    }

    /**
     * Get odds movement/history for a game
     * @param {string} eventId - ESPN event ID
     * @param {string} providerId - Provider ID (default: ESPN BET = 58)
     * @returns {Promise<Array>} Historical odds data
     */
    async getOddsMovement(eventId, providerId = '58') {
        try {
            const url = `${this.baseURL}/events/${eventId}/competitions/${eventId}/odds/${providerId}/history/0/movement`;
            const response = await fetch(url);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }

            const data = await response.json();
            console.log('📈 Odds movement fetched:', data);
            return data;

        } catch (error) {
            console.error('❌ Failed to fetch odds movement:', error);
            return [];
        }
    }

    /**
     * Get win probabilities for a game
     * @param {string} eventId - ESPN event ID
     * @returns {Promise<Object>} Win probability data
     */
    async getWinProbabilities(eventId) {
        try {
            const url = `${this.baseURL}/events/${eventId}/competitions/${eventId}/probabilities`;
            const response = await fetch(url);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }

            const data = await response.json();
            console.log('🎯 Win probabilities:', data);
            return data;

        } catch (error) {
            console.error('❌ Failed to fetch win probabilities:', error);
            return null;
        }
    }

    /**
     * Convert American odds to decimal
     * @param {number} americanOdds - American odds (e.g., -110, +150)
     * @returns {number} Decimal odds
     */
    americanToDecimal(americanOdds) {
        if (americanOdds > 0) {
            return (americanOdds / 100) + 1;
        } else {
            return (100 / Math.abs(americanOdds)) + 1;
        }
    }

    /**
     * Calculate implied probability from American odds
     * @param {number} americanOdds - American odds
     * @returns {number} Implied probability (0-100)
     */
    impliedProbability(americanOdds) {
        if (americanOdds > 0) {
            return (100 / (americanOdds + 100)) * 100;
        } else {
            return (Math.abs(americanOdds) / (Math.abs(americanOdds) + 100)) * 100;
        }
    }

    /**
     * Format odds for display
     * @param {number} americanOdds - American odds
     * @returns {string} Formatted odds string
     */
    formatOdds(americanOdds) {
        if (!americanOdds) return 'N/A';
        return americanOdds > 0 ? `+${americanOdds}` : americanOdds.toString();
    }

    /**
     * Get best available odds across all providers
     * @param {string} eventId - ESPN event ID
     * @param {string} betType - Type of bet (moneyline, spread, total)
     * @param {string} side - Side of bet (home, away, over, under)
     * @returns {Promise<Object>} Best odds available
     */
    async getBestOdds(eventId, betType, side) {
        const gameOdds = await this.getGameOdds(eventId);
        
        if (!gameOdds.providers || gameOdds.providers.length === 0) {
            return null;
        }

        let bestOdds = null;
        let bestValue = -Infinity;

        for (const provider of gameOdds.providers) {
            let currentOdds = null;

            switch (betType.toLowerCase()) {
                case 'moneyline':
                    currentOdds = side === 'home' 
                        ? provider.homeTeam.moneyLine 
                        : provider.awayTeam.moneyLine;
                    break;
                case 'spread':
                    currentOdds = side === 'home'
                        ? provider.homeTeam.spreadOdds
                        : provider.awayTeam.spreadOdds;
                    break;
                case 'total':
                    currentOdds = side === 'over'
                        ? provider.overOdds
                        : provider.underOdds;
                    break;
            }

            if (currentOdds && currentOdds > bestValue) {
                bestValue = currentOdds;
                bestOdds = {
                    provider: provider.name,
                    odds: currentOdds,
                    formattedOdds: this.formatOdds(currentOdds),
                    impliedProb: this.impliedProbability(currentOdds),
                    decimalOdds: this.americanToDecimal(currentOdds)
                };
            }
        }

        return bestOdds;
    }

    /**
     * Clear cache
     */
    clearCache() {
        this.cache.clear();
        console.log('🗑️ ESPN odds cache cleared');
    }
}

// Initialize global instance
window.ESPNOdds = new ESPNOddsService();
console.log('🎰 ESPN Odds Service initialized - Real sportsbook data available!');
