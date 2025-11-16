/**
 * Advanced Analytics JavaScript Module
 * Handles real-time odds comparison, value betting, and market analysis
 */

class AdvancedAnalytics {
    constructor() {
        this.apiBaseUrl = 'https://betyard-ml-backend.onrender.com/api';
        this.currentSport = 'nfl';
        this.refreshInterval = null;
        this.liveFeedActive = false;
        this.bankroll = 1000;
        this.riskTolerance = 'moderate';
        
        // Risk tolerance multipliers
        this.riskMultipliers = {
            conservative: 0.02,
            moderate: 0.035,
            aggressive: 0.075
        };
        
        this.init();
    }
    
    init() {
        // Initialize the analytics panel when it's shown
        this.loadMarketOverview();
        this.loadValueBets();
        this.loadArbitrageOpportunities();
        this.updateBankrollDisplay();
        
        // Set up auto-refresh
        this.startAutoRefresh();
    }
    
    async loadMarketOverview() {
        try {
            showLoadingState('market-overview');
            
            const response = await fetch(`${this.apiBaseUrl}/odds/compare/${this.currentSport}`);
            const data = await response.json();
            
            if (data.success) {
                this.updateMarketStats(data.data);
            } else {
                console.error('Failed to load market overview:', data.error);
                this.showError('market-overview', 'Failed to load market data');
            }
        } catch (error) {
            console.error('Market overview error:', error);
            this.showError('market-overview', 'Connection error');
        }
    }
    
    updateMarketStats(marketData) {
        // Update market statistics
        document.getElementById('total-games').textContent = marketData.total_games || '--';
        document.getElementById('total-bookmakers').textContent = marketData.total_bookmakers || '--';
        document.getElementById('arbitrage-count').textContent = marketData.arbitrage_opportunities?.length || 0;
        
        // Calculate average market efficiency
        const games = marketData.best_odds_by_game || {};
        const efficiencies = Object.values(games).map(game => game.market_efficiency);
        const avgEfficiency = efficiencies.length > 0 
            ? efficiencies.reduce((a, b) => a + b, 0) / efficiencies.length 
            : 0;
        
        document.getElementById('avg-efficiency').textContent = `${(avgEfficiency * 100).toFixed(1)}%`;
        
        // Update efficiency bar
        const efficiencyBar = document.getElementById('efficiency-bar');
        efficiencyBar.style.width = `${avgEfficiency * 100}%`;
        
        // Update arbitrage opportunities
        this.updateArbitrageDisplay(marketData.arbitrage_opportunities || []);
        
        // Populate game selector for odds comparison
        this.populateGameSelector(games);
    }
    
    async loadValueBets() {
        try {
            // First get model predictions (simplified for demo)
            const mockPredictions = await this.getMockPredictions();
            
            const response = await fetch(`${this.apiBaseUrl}/odds/value-bets/${this.currentSport}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(mockPredictions)
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.updateValueBetsDisplay(data.data.value_bets);
                document.getElementById('value-bet-count').textContent = data.data.count;
            } else {
                console.error('Failed to load value bets:', data.error);
                this.showNoValueBets();
            }
        } catch (error) {
            console.error('Value bets error:', error);
            this.showNoValueBets();
        }
    }
    
    updateValueBetsDisplay(valueBets) {
        const tbody = document.getElementById('value-bets-tbody');
        const noValueBets = document.getElementById('no-value-bets');
        
        if (!valueBets || valueBets.length === 0) {
            tbody.innerHTML = '';
            noValueBets.style.display = 'block';
            return;
        }
        
        noValueBets.style.display = 'none';
        
        tbody.innerHTML = valueBets.map(bet => `
            <tr>
                <td>
                    <div class="game-info">
                        <strong>${bet.away_team} @ ${bet.home_team}</strong>
                        <small>${new Date(bet.commence_time).toLocaleDateString()}</small>
                    </div>
                </td>
                <td>${this.formatMarket(bet.market)}</td>
                <td><strong>${bet.outcome}</strong></td>
                <td>${this.formatOdds(bet.best_odds)}</td>
                <td>
                    <span class="bookmaker-badge">${bet.best_bookmaker}</span>
                </td>
                <td>${(bet.model_probability * 100).toFixed(1)}%</td>
                <td>${(bet.implied_probability * 100).toFixed(1)}%</td>
                <td class="edge-positive">+${(bet.edge * 100).toFixed(1)}%</td>
                <td>${(bet.kelly_fraction * 100).toFixed(1)}%</td>
                <td>
                    <button class="btn-small" onclick="analytics.placeBet('${bet.game_id}', '${bet.outcome}', ${bet.best_odds}, ${bet.kelly_fraction})">
                        <i class="material-icons">add_shopping_cart</i>
                    </button>
                </td>
            </tr>
        `).join('');
    }
    
    updateArbitrageDisplay(arbitrageOps) {
        const arbitrageList = document.getElementById('arbitrage-list');
        const noArbitrage = document.getElementById('no-arbitrage');
        
        if (!arbitrageOps || arbitrageOps.length === 0) {
            arbitrageList.innerHTML = '';
            noArbitrage.style.display = 'block';
            return;
        }
        
        noArbitrage.style.display = 'none';
        
        arbitrageList.innerHTML = arbitrageOps.map(arb => `
            <div class="arbitrage-item">
                <div class="arbitrage-game">${arb.game}</div>
                <div class="arbitrage-details">
                    <div>
                        <strong>Home:</strong> ${this.formatOdds(arb.home_odds)} 
                        <span class="bookmaker-badge">${arb.home_bookmaker}</span>
                    </div>
                    <div>
                        <strong>Away:</strong> ${this.formatOdds(arb.away_odds)} 
                        <span class="bookmaker-badge">${arb.away_bookmaker}</span>
                    </div>
                </div>
                <div class="arbitrage-profit">
                    ${arb.profit_margin.toFixed(2)}% Guaranteed Profit
                </div>
            </div>
        `).join('');
    }
    
    populateGameSelector(games) {
        const selector = document.getElementById('odds-game-select');
        selector.innerHTML = '<option value="">Select Game</option>';
        
        Object.entries(games).forEach(([gameId, game]) => {
            const option = document.createElement('option');
            option.value = gameId;
            option.textContent = `${game.away_team} @ ${game.home_team}`;
            selector.appendChild(option);
        });
    }
    
    async loadOddsComparison() {
        const gameId = document.getElementById('odds-game-select').value;
        if (!gameId) return;
        
        try {
            // This would load specific game odds comparison
            // For now, show placeholder
            const comparisonDiv = document.getElementById('odds-comparison');
            comparisonDiv.innerHTML = `
                <div class="odds-item">
                    <span class="odds-bookmaker">Loading odds comparison...</span>
                </div>
            `;
        } catch (error) {
            console.error('Odds comparison error:', error);
        }
    }
    
    async loadArbitrageOpportunities() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/odds/arbitrage/${this.currentSport}`);
            const data = await response.json();
            
            if (data.success) {
                this.updateArbitrageDisplay(data.data.arbitrage_opportunities);
            }
        } catch (error) {
            console.error('Arbitrage error:', error);
        }
    }
    
    updateBankrollDisplay() {
        document.getElementById('current-bankroll').value = this.bankroll;
        document.getElementById('risk-tolerance').value = this.riskTolerance;
        this.updateAllocationBars();
    }
    
    updateAllocationBars() {
        const maxBet = this.bankroll * this.riskMultipliers[this.riskTolerance];
        const conservativeBet = this.bankroll * 0.01;
        const moderateBet = this.bankroll * 0.025;
        const aggressiveBet = this.bankroll * 0.05;
        
        const barsContainer = document.getElementById('allocation-bars');
        barsContainer.innerHTML = `
            <div class="allocation-bar">
                <span class="allocation-label">Conservative (1%)</span>
                <span class="allocation-amount">$${conservativeBet.toFixed(0)}</span>
            </div>
            <div class="allocation-bar">
                <span class="allocation-label">Moderate (2.5%)</span>
                <span class="allocation-amount">$${moderateBet.toFixed(0)}</span>
            </div>
            <div class="allocation-bar">
                <span class="allocation-label">Aggressive (5%)</span>
                <span class="allocation-amount">$${aggressiveBet.toFixed(0)}</span>
            </div>
            <div class="allocation-bar">
                <span class="allocation-label">Max Recommended</span>
                <span class="allocation-amount">$${maxBet.toFixed(0)}</span>
            </div>
        `;
    }
    
    startAutoRefresh() {
        // Refresh data every 30 seconds
        this.refreshInterval = setInterval(() => {
            this.loadMarketOverview();
            this.loadValueBets();
        }, 30000);
    }
    
    stopAutoRefresh() {
        if (this.refreshInterval) {
            clearInterval(this.refreshInterval);
            this.refreshInterval = null;
        }
    }
    
    toggleLiveFeed() {
        const liveToggle = document.getElementById('live-toggle');
        const liveDot = document.getElementById('live-dot');
        const liveStatus = document.getElementById('live-status');
        
        this.liveFeedActive = !this.liveFeedActive;
        
        if (this.liveFeedActive) {
            liveToggle.innerHTML = '<i class="material-icons">pause</i>';
            liveDot.classList.add('connected');
            liveStatus.textContent = 'Live';
            this.startLiveFeed();
        } else {
            liveToggle.innerHTML = '<i class="material-icons">play_arrow</i>';
            liveDot.classList.remove('connected');
            liveStatus.textContent = 'Paused';
            this.stopLiveFeed();
        }
    }
    
    startLiveFeed() {
        // Simulate live odds updates
        this.liveFeedInterval = setInterval(() => {
            this.addLiveUpdate();
        }, 5000);
    }
    
    stopLiveFeed() {
        if (this.liveFeedInterval) {
            clearInterval(this.liveFeedInterval);
        }
    }
    
    addLiveUpdate() {
        const ticker = document.getElementById('odds-ticker');
        const changesList = document.getElementById('changes-list');
        
        // Add simulated live update
        const teams = ['Bills', 'Dolphins', 'Chiefs', 'Chargers', 'Cowboys', 'Giants'];
        const randomTeam = teams[Math.floor(Math.random() * teams.length)];
        const change = (Math.random() - 0.5) * 20;
        const isPositive = change > 0;
        
        const tickerItem = document.createElement('div');
        tickerItem.className = 'ticker-item';
        tickerItem.innerHTML = `
            <span class="ticker-game">${randomTeam} ML</span>
            <span class="ticker-change ${isPositive ? 'positive' : 'negative'}">
                ${isPositive ? '+' : ''}${change.toFixed(0)}
            </span>
        `;
        
        ticker.insertBefore(tickerItem, ticker.firstChild);
        
        // Limit ticker items
        while (ticker.children.length > 10) {
            ticker.removeChild(ticker.lastChild);
        }
        
        // Add to changes list
        const changeItem = document.createElement('div');
        changeItem.className = 'change-item';
        changeItem.innerHTML = `
            <div>
                <strong>${randomTeam}</strong> line moved ${isPositive ? 'up' : 'down'}
            </div>
            <div class="change-time">${new Date().toLocaleTimeString()}</div>
        `;
        
        changesList.insertBefore(changeItem, changesList.firstChild);
        
        // Limit changes list
        while (changesList.children.length > 8) {
            changesList.removeChild(changesList.lastChild);
        }
    }
    
    placeBet(gameId, outcome, odds, kellyFraction) {
        const betAmount = this.bankroll * kellyFraction;
        
        const confirmation = confirm(
            `Place bet on ${outcome}?\n\n` +
            `Odds: ${this.formatOdds(odds)}\n` +
            `Recommended bet: $${betAmount.toFixed(2)} (${(kellyFraction * 100).toFixed(1)}% of bankroll)\n\n` +
            `This will open your sportsbook app.`
        );
        
        if (confirmation) {
            // In a real app, this would integrate with sportsbook APIs or redirect to betting sites
            alert('Bet tracking added to your portfolio!');
        }
    }
    
    formatOdds(americanOdds) {
        return americanOdds > 0 ? `+${americanOdds}` : `${americanOdds}`;
    }
    
    formatMarket(market) {
        const marketNames = {
            'h2h': 'Moneyline',
            'spreads': 'Point Spread',
            'totals': 'Over/Under',
            'moneyline': 'Moneyline'
        };
        return marketNames[market] || market;
    }
    
    showLoadingState(elementId) {
        const element = document.getElementById(elementId);
        if (element) {
            element.innerHTML = '<div class="loading">Loading...</div>';
        }
    }
    
    showError(elementId, message) {
        const element = document.getElementById(elementId);
        if (element) {
            element.innerHTML = `<div class="error">${message}</div>`;
        }
    }
    
    showNoValueBets() {
        document.getElementById('value-bets-tbody').innerHTML = '';
        document.getElementById('no-value-bets').style.display = 'block';
        document.getElementById('value-bet-count').textContent = '0';
    }
    
    async getMockPredictions() {
        // Mock predictions for demonstration
        // In real implementation, this would come from your ML models
        return {
            "game_1": {
                "home_team": "Buffalo Bills",
                "away_team": "Miami Dolphins",
                "commence_time": "2025-10-25T20:00:00Z",
                "predictions": {
                    "home_win_probability": 0.68,
                    "away_win_probability": 0.32,
                    "over_probability": 0.55,
                    "under_probability": 0.45
                }
            },
            "game_2": {
                "home_team": "Kansas City Chiefs",
                "away_team": "Los Angeles Chargers",
                "commence_time": "2025-10-25T23:30:00Z",
                "predictions": {
                    "home_win_probability": 0.72,
                    "away_win_probability": 0.28,
                    "over_probability": 0.58,
                    "under_probability": 0.42
                }
            }
        };
    }
}

// Global functions for HTML onclick handlers
function refreshMarketData() {
    if (window.analytics) {
        window.analytics.loadMarketOverview();
    }
}

function filterValueBets() {
    // Implement value bet filtering
    const filter = document.getElementById('value-bet-filter').value;
    console.log('Filtering value bets:', filter);
}

function analyzeValueBets() {
    if (window.analytics) {
        window.analytics.loadValueBets();
    }
}

function loadOddsComparison() {
    if (window.analytics) {
        window.analytics.loadOddsComparison();
    }
}

function openBankrollSettings() {
    alert('Bankroll settings panel would open here');
}

function updateBankroll() {
    const newBankroll = parseFloat(document.getElementById('current-bankroll').value) || 1000;
    if (window.analytics) {
        window.analytics.bankroll = newBankroll;
        window.analytics.updateAllocationBars();
    }
}

function updateRiskTolerance() {
    const newTolerance = document.getElementById('risk-tolerance').value;
    if (window.analytics) {
        window.analytics.riskTolerance = newTolerance;
        window.analytics.updateAllocationBars();
    }
}

function toggleLiveFeed() {
    if (window.analytics) {
        window.analytics.toggleLiveFeed();
    }
}

// Initialize analytics when the panel is shown
document.addEventListener('DOMContentLoaded', function() {
    // Initialize analytics when analytics panel becomes visible
    const observer = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutation) {
            if (mutation.type === 'attributes' && mutation.attributeName === 'class') {
                const analyticsPanel = document.getElementById('analytics-panel');
                if (analyticsPanel && analyticsPanel.classList.contains('active')) {
                    if (!window.analytics) {
                        window.analytics = new AdvancedAnalytics();
                    }
                }
            }
        });
    });
    
    const analyticsPanel = document.getElementById('analytics-panel');
    if (analyticsPanel) {
        observer.observe(analyticsPanel, { attributes: true });
    }
});

// Clean up when leaving the page
window.addEventListener('beforeunload', function() {
    if (window.analytics) {
        window.analytics.stopAutoRefresh();
        window.analytics.stopLiveFeed();
    }
});