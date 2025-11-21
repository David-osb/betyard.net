/**
 * BetYard Live Odds Integration
 * Fetches real-time DraftKings odds from backend
 */

const LIVE_ODDS_CONFIG = {
    apiUrl: 'https://betyard-ml-backend.onrender.com/api/live-odds',
    refreshInterval: 60000, // 1 minute
    timeout: 10000 // 10 seconds
};

/**
 * Fetch live odds for a player/stat combination
 * @param {string} playerName - "Josh Allen"
 * @param {string} statType - "passing_yards", "rushing_yards", etc.
 * @returns {Promise<Object>} Odds data
 */
async function fetchLiveOdds(playerName, statType) {
    const url = `${LIVE_ODDS_CONFIG.apiUrl}/${encodeURIComponent(playerName)}?stat=${statType}`;
    
    try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), LIVE_ODDS_CONFIG.timeout);
        
        const response = await fetch(url, {
            signal: controller.signal,
            headers: {
                'Accept': 'application/json'
            }
        });
        
        clearTimeout(timeoutId);
        
        if (!response.ok) {
            const error = await response.json();
            console.warn(`No odds available for ${playerName} ${statType}:`, error.error);
            return null;
        }
        
        const data = await response.json();
        return data;
        
    } catch (error) {
        if (error.name === 'AbortError') {
            console.error('Request timeout fetching odds');
        } else {
            console.error('Error fetching live odds:', error);
        }
        return null;
    }
}

/**
 * Display live odds in a player card
 * @param {HTMLElement} container - Element to display odds in
 * @param {string} playerName 
 * @param {string} statType 
 */
async function displayLiveOdds(container, playerName, statType) {
    if (!container) return;
    
    // Show loading state
    container.innerHTML = `
        <div class="live-odds-loading">
            <span class="spinner"></span> Fetching live odds...
        </div>
    `;
    
    const odds = await fetchLiveOdds(playerName, statType);
    
    if (!odds) {
        container.innerHTML = `
            <div class="live-odds-unavailable">
                <span class="icon">📊</span>
                <span>Live odds unavailable</span>
            </div>
        `;
        return;
    }
    
    // Display odds
    const overColor = odds.over_odds < -120 ? 'unfavorable' : odds.over_odds > -105 ? 'favorable' : 'neutral';
    const underColor = odds.under_odds < -120 ? 'unfavorable' : odds.under_odds > -105 ? 'favorable' : 'neutral';
    
    container.innerHTML = `
        <div class="live-odds-container">
            <div class="odds-header">
                <span class="live-badge">🔴 LIVE</span>
                <span class="sportsbook">DraftKings</span>
            </div>
            <div class="odds-line">
                <span class="label">Line:</span>
                <span class="value">${odds.line}</span>
            </div>
            <div class="odds-buttons">
                <button class="odds-btn over ${overColor}" data-bet="over">
                    <span class="label">OVER</span>
                    <span class="odds">${formatOdds(odds.over_odds)}</span>
                </button>
                <button class="odds-btn under ${underColor}" data-bet="under">
                    <span class="label">UNDER</span>
                    <span class="odds">${formatOdds(odds.under_odds)}</span>
                </button>
            </div>
            <div class="odds-footer">
                <span class="market">${odds.market_name}</span>
                <span class="timestamp">${formatTimestamp(odds.timestamp)}</span>
            </div>
        </div>
    `;
}

/**
 * Format American odds with + sign
 */
function formatOdds(odds) {
    return odds > 0 ? `+${odds}` : `${odds}`;
}

/**
 * Format timestamp to relative time
 */
function formatTimestamp(timestamp) {
    const date = new Date(timestamp);
    const now = new Date();
    const diff = Math.floor((now - date) / 1000);
    
    if (diff < 60) return 'Just now';
    if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
    if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
    return date.toLocaleDateString();
}

/**
 * Auto-refresh odds every interval
 */
function enableAutoRefresh(container, playerName, statType) {
    displayLiveOdds(container, playerName, statType);
    
    return setInterval(() => {
        displayLiveOdds(container, playerName, statType);
    }, LIVE_ODDS_CONFIG.refreshInterval);
}

// Export for use in other scripts
window.BetYardOdds = {
    fetch: fetchLiveOdds,
    display: displayLiveOdds,
    enableAutoRefresh: enableAutoRefresh
};
