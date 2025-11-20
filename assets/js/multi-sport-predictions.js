/**
 * Multi-Sport Predictions Integration
 * Handles NBA, NHL, MLB player prop predictions from backend API
 * Version: 1.0.0
 */

class MultiSportPredictions {
    constructor() {
        // Backend API URL (auto-detect environment)
        this.baseURL = this.detectBackendURL();
        this.cache = new Map();
        this.cacheDuration = 5 * 60 * 1000; // 5 minutes
        
        console.log('🎯 Multi-Sport Predictions initialized:', this.baseURL);
    }
    
    detectBackendURL() {
        // Use ML_CONFIG if available
        if (window.ML_CONFIG) {
            const activeProvider = window.ML_CONFIG.ACTIVE;
            const url = window.ML_CONFIG[activeProvider];
            console.log(`🔧 Using ${activeProvider} backend:`, url);
            return url;
        }
        
        // Auto-detect environment
        const isLocal = window.location.hostname === 'localhost' || 
                       window.location.hostname === '127.0.0.1';
        
        return isLocal 
            ? 'http://localhost:10000' 
            : 'https://betyard-ml-backend.onrender.com';
    }
    
    /**
     * Fetch NBA team players with prop predictions
     */
    async fetchNBATeam(teamCode) {
        const cacheKey = `nba_${teamCode}`;
        
        // Check cache
        const cached = this.getCached(cacheKey);
        if (cached) return cached;
        
        try {
            console.log(`🏀 Fetching NBA team: ${teamCode}`);
            const response = await fetch(`${this.baseURL}/players/nba/team/${teamCode}`);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();
            
            if (data.success) {
                console.log(`✅ NBA ${teamCode}: ${data.count} players loaded`);
                this.setCache(cacheKey, data);
                return data;
            } else {
                throw new Error(data.error || 'Failed to fetch NBA team');
            }
        } catch (error) {
            console.error(`❌ NBA fetch error (${teamCode}):`, error.message);
            return null;
        }
    }
    
    /**
     * Fetch NHL team players with prop predictions
     */
    async fetchNHLTeam(teamCode) {
        const cacheKey = `nhl_${teamCode}`;
        
        // Check cache
        const cached = this.getCached(cacheKey);
        if (cached) return cached;
        
        try {
            console.log(`🏒 Fetching NHL team: ${teamCode}`);
            const response = await fetch(`${this.baseURL}/players/nhl/team/${teamCode}`);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();
            
            if (data.success) {
                console.log(`✅ NHL ${teamCode}: ${data.total_skaters} skaters, ${data.total_goalies} goalies loaded`);
                this.setCache(cacheKey, data);
                return data;
            } else {
                throw new Error(data.error || 'Failed to fetch NHL team');
            }
        } catch (error) {
            console.error(`❌ NHL fetch error (${teamCode}):`, error.message);
            return null;
        }
    }
    
    /**
     * Fetch MLB team players with prop predictions
     */
    async fetchMLBTeam(teamCode) {
        const cacheKey = `mlb_${teamCode}`;
        
        // Check cache
        const cached = this.getCached(cacheKey);
        if (cached) return cached;
        
        try {
            console.log(`⚾ Fetching MLB team: ${teamCode}`);
            const response = await fetch(`${this.baseURL}/players/mlb/team/${teamCode}`);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();
            
            if (data.success) {
                console.log(`✅ MLB ${teamCode}: ${data.total_hitters} hitters, ${data.total_pitchers} pitchers loaded`);
                this.setCache(cacheKey, data);
                return data;
            } else {
                throw new Error(data.error || 'Failed to fetch MLB team');
            }
        } catch (error) {
            console.error(`❌ MLB fetch error (${teamCode}):`, error.message);
            return null;
        }
    }
    
    /**
     * Render NBA player props as HTML cards
     */
    renderNBAPlayers(players) {
        if (!players || players.length === 0) {
            return '<p style="color: #64748b; text-align: center; padding: 20px;">No NBA players found</p>';
        }
        
        return players.map(player => `
            <div class="player-prop-card nba-card" style="background: white; border-radius: 12px; padding: 20px; margin-bottom: 16px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
                    <div>
                        <h3 style="color: #1e293b; margin: 0; font-size: 18px; font-weight: 600;">${player.player_name}</h3>
                        <p style="color: #64748b; margin: 4px 0 0 0; font-size: 14px;">${player.position} • ${player.team} • ${player.games_played} games</p>
                    </div>
                    <div style="font-size: 32px;">🏀</div>
                </div>
                
                <div class="props-grid" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px;">
                    ${player.props.points ? this.renderNBAProp('Points', player.props.points, '🎯') : ''}
                    ${player.props.rebounds ? this.renderNBAProp('Rebounds', player.props.rebounds, '🏀') : ''}
                    ${player.props.assists ? this.renderNBAProp('Assists', player.props.assists, '🎯') : ''}
                    ${player.props.threes_made ? this.renderNBAProp('3-Pointers', player.props.threes_made, '🔥') : ''}
                </div>
            </div>
        `).join('');
    }
    
    renderNBAProp(label, prop, icon) {
        if (!prop || !prop.line) return '';
        
        const recommendationColor = {
            'OVER': '#10b981',
            'UNDER': '#ef4444',
            'NO BET': '#64748b'
        };
        
        const recommendation = prop.recommendation || 'NO BET';
        const color = recommendationColor[recommendation] || '#64748b';
        
        return `
            <div class="prop-item" style="background: #f8fafc; border-radius: 8px; padding: 12px;">
                <div style="display: flex; align-items: center; margin-bottom: 6px;">
                    <span style="margin-right: 6px;">${icon}</span>
                    <span style="font-size: 12px; font-weight: 600; color: #64748b; text-transform: uppercase;">${label}</span>
                </div>
                <div style="font-size: 20px; font-weight: 700; color: #1e293b; margin-bottom: 4px;">
                    O/U ${prop.line}
                </div>
                <div style="font-size: 12px; color: #64748b; margin-bottom: 8px;">
                    Avg: ${prop.average ? prop.average.toFixed(1) : 'N/A'} | ${prop.over_probability ? prop.over_probability.toFixed(1) : '50'}% OVER
                </div>
                <div style="background: ${color}; color: white; padding: 4px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; text-align: center;">
                    ${recommendation}
                </div>
            </div>
        `;
    }
    
    /**
     * Render NHL players (skaters and goalies)
     */
    renderNHLPlayers(skaters, goalies) {
        let html = '';
        
        // Render skaters
        if (skaters && skaters.length > 0) {
            html += '<h3 style="color: #1e293b; margin: 20px 0 12px 0; font-size: 18px; font-weight: 600;">🏒 Skaters</h3>';
            html += skaters.slice(0, 5).map(player => `
                <div class="player-prop-card nhl-card" style="background: white; border-radius: 12px; padding: 20px; margin-bottom: 16px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
                        <div>
                            <h3 style="color: #1e293b; margin: 0; font-size: 18px; font-weight: 600;">${player.player_name}</h3>
                            <p style="color: #64748b; margin: 4px 0 0 0; font-size: 14px;">${player.position} • ${player.team} • ${player.games_played} games</p>
                        </div>
                        <div style="font-size: 32px;">🏒</div>
                    </div>
                    
                    <div class="props-grid" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px;">
                        ${this.renderNHLGoalProp(player.props.anytime_goal)}
                        ${this.renderNHLStatProp('Assists', player.props.assists)}
                        ${this.renderNHLStatProp('Shots', player.props.shots)}
                    </div>
                </div>
            `).join('');
        }
        
        // Render goalies
        if (goalies && goalies.length > 0) {
            html += '<h3 style="color: #1e293b; margin: 30px 0 12px 0; font-size: 18px; font-weight: 600;">🥅 Goalies</h3>';
            html += goalies.map(player => `
                <div class="player-prop-card nhl-goalie-card" style="background: white; border-radius: 12px; padding: 20px; margin-bottom: 16px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
                        <div>
                            <h3 style="color: #1e293b; margin: 0; font-size: 18px; font-weight: 600;">${player.player_name}</h3>
                            <p style="color: #64748b; margin: 4px 0 0 0; font-size: 14px;">${player.position} • ${player.team} • ${player.games_played} games</p>
                        </div>
                        <div style="font-size: 32px;">🥅</div>
                    </div>
                    
                    <div class="props-grid" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px;">
                        ${this.renderNHLStatProp('Saves', player.props.saves)}
                        ${this.renderNHLGoalsAgainst(player.props.goals_against)}
                    </div>
                </div>
            `).join('');
        }
        
        return html || '<p style="color: #64748b; text-align: center; padding: 20px;">No NHL players found</p>';
    }
    
    renderNHLGoalProp(prop) {
        const betColor = prop.recommendation === 'BET' ? '#10b981' : '#64748b';
        return `
            <div class="prop-item" style="background: #f8fafc; border-radius: 8px; padding: 12px;">
                <div style="display: flex; align-items: center; margin-bottom: 6px;">
                    <span style="margin-right: 6px;">🚨</span>
                    <span style="font-size: 12px; font-weight: 600; color: #64748b; text-transform: uppercase;">Anytime Goal</span>
                </div>
                <div style="font-size: 20px; font-weight: 700; color: #1e293b; margin-bottom: 4px;">
                    ${prop.odds > 0 ? '+' : ''}${prop.odds}
                </div>
                <div style="font-size: 12px; color: #64748b; margin-bottom: 8px;">
                    ${prop.probability.toFixed(1)}% probability
                </div>
                <div style="background: ${betColor}; color: white; padding: 4px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; text-align: center;">
                    ${prop.recommendation}
                </div>
            </div>
        `;
    }
    
    renderNHLStatProp(label, prop) {
        const recommendationColor = {
            'OVER': '#10b981',
            'UNDER': '#ef4444',
            'NO BET': '#64748b'
        };
        const color = recommendationColor[prop.recommendation] || '#64748b';
        
        return `
            <div class="prop-item" style="background: #f8fafc; border-radius: 8px; padding: 12px;">
                <div style="display: flex; align-items: center; margin-bottom: 6px;">
                    <span style="margin-right: 6px;">📊</span>
                    <span style="font-size: 12px; font-weight: 600; color: #64748b; text-transform: uppercase;">${label}</span>
                </div>
                <div style="font-size: 20px; font-weight: 700; color: #1e293b; margin-bottom: 4px;">
                    O/U ${prop.line}
                </div>
                <div style="font-size: 12px; color: #64748b; margin-bottom: 8px;">
                    Avg: ${prop.average.toFixed(1)} | ${prop.over_probability.toFixed(1)}% OVER
                </div>
                <div style="background: ${color}; color: white; padding: 4px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; text-align: center;">
                    ${prop.recommendation}
                </div>
            </div>
        `;
    }
    
    renderNHLGoalsAgainst(prop) {
        return `
            <div class="prop-item" style="background: #f8fafc; border-radius: 8px; padding: 12px;">
                <div style="display: flex; align-items: center; margin-bottom: 6px;">
                    <span style="margin-right: 6px;">🥅</span>
                    <span style="font-size: 12px; font-weight: 600; color: #64748b; text-transform: uppercase;">Goals Against</span>
                </div>
                <div style="font-size: 20px; font-weight: 700; color: #1e293b; margin-bottom: 4px;">
                    U 2.5
                </div>
                <div style="font-size: 12px; color: #64748b; margin-bottom: 8px;">
                    ${prop.under_2_5_probability.toFixed(1)}% Under 2.5
                </div>
            </div>
        `;
    }
    
    /**
     * Render MLB players (hitters and pitchers)
     */
    renderMLBPlayers(hitters, pitchers) {
        let html = '';
        
        // Render hitters
        if (hitters && hitters.length > 0) {
            html += '<h3 style="color: #1e293b; margin: 20px 0 12px 0; font-size: 18px; font-weight: 600;">⚾ Hitters</h3>';
            html += hitters.slice(0, 5).map(player => `
                <div class="player-prop-card mlb-card" style="background: white; border-radius: 12px; padding: 20px; margin-bottom: 16px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
                        <div>
                            <h3 style="color: #1e293b; margin: 0; font-size: 18px; font-weight: 600;">${player.player_name}</h3>
                            <p style="color: #64748b; margin: 4px 0 0 0; font-size: 14px;">${player.position} • ${player.team} • ${player.games_played} games</p>
                        </div>
                        <div style="font-size: 32px;">⚾</div>
                    </div>
                    
                    <div class="props-grid" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px;">
                        ${this.renderMLBHitsProp(player.props.hits)}
                        ${this.renderMLBHRProp(player.props.home_run)}
                        ${this.renderMLBStatProp('RBIs', player.props.rbis)}
                    </div>
                </div>
            `).join('');
        }
        
        // Render pitchers
        if (pitchers && pitchers.length > 0) {
            html += '<h3 style="color: #1e293b; margin: 30px 0 12px 0; font-size: 18px; font-weight: 600;">🥎 Pitchers</h3>';
            html += pitchers.slice(0, 3).map(player => `
                <div class="player-prop-card mlb-pitcher-card" style="background: white; border-radius: 12px; padding: 20px; margin-bottom: 16px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
                        <div>
                            <h3 style="color: #1e293b; margin: 0; font-size: 18px; font-weight: 600;">${player.player_name}</h3>
                            <p style="color: #64748b; margin: 4px 0 0 0; font-size: 14px;">${player.position} • ${player.team} • ${player.games_played} games</p>
                        </div>
                        <div style="font-size: 32px;">🥎</div>
                    </div>
                    
                    <div class="props-grid" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px;">
                        ${this.renderMLBStatProp('Strikeouts', player.props.strikeouts)}
                    </div>
                </div>
            `).join('');
        }
        
        return html || '<p style="color: #64748b; text-align: center; padding: 20px;">No MLB players found</p>';
    }
    
    renderMLBHitsProp(prop) {
        const recommendationColor = {
            'OVER': '#10b981',
            'UNDER': '#ef4444',
            'NO BET': '#64748b'
        };
        const color = recommendationColor[prop.recommendation] || '#64748b';
        
        return `
            <div class="prop-item" style="background: #f8fafc; border-radius: 8px; padding: 12px;">
                <div style="display: flex; align-items: center; margin-bottom: 6px;">
                    <span style="margin-right: 6px;">💥</span>
                    <span style="font-size: 12px; font-weight: 600; color: #64748b; text-transform: uppercase;">2+ Hits</span>
                </div>
                <div style="font-size: 20px; font-weight: 700; color: #1e293b; margin-bottom: 4px;">
                    O/U ${prop.line}
                </div>
                <div style="font-size: 12px; color: #64748b; margin-bottom: 8px;">
                    ${prop.over_probability.toFixed(1)}% OVER
                </div>
                <div style="background: ${color}; color: white; padding: 4px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; text-align: center;">
                    ${prop.recommendation}
                </div>
            </div>
        `;
    }
    
    renderMLBHRProp(prop) {
        const betColor = prop.recommendation === 'BET' ? '#10b981' : '#64748b';
        return `
            <div class="prop-item" style="background: #f8fafc; border-radius: 8px; padding: 12px;">
                <div style="display: flex; align-items: center; margin-bottom: 6px;">
                    <span style="margin-right: 6px;">⚡</span>
                    <span style="font-size: 12px; font-weight: 600; color: #64748b; text-transform: uppercase;">Home Run</span>
                </div>
                <div style="font-size: 20px; font-weight: 700; color: #1e293b; margin-bottom: 4px;">
                    ${prop.odds > 0 ? '+' : ''}${prop.odds}
                </div>
                <div style="font-size: 12px; color: #64748b; margin-bottom: 8px;">
                    ${prop.probability.toFixed(1)}% probability
                </div>
                <div style="background: ${betColor}; color: white; padding: 4px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; text-align: center;">
                    ${prop.recommendation}
                </div>
            </div>
        `;
    }
    
    renderMLBStatProp(label, prop) {
        const recommendationColor = {
            'OVER': '#10b981',
            'UNDER': '#ef4444',
            'NO BET': '#64748b'
        };
        const color = recommendationColor[prop.recommendation] || '#64748b';
        
        return `
            <div class="prop-item" style="background: #f8fafc; border-radius: 8px; padding: 12px;">
                <div style="display: flex; align-items: center; margin-bottom: 6px;">
                    <span style="margin-right: 6px;">📊</span>
                    <span style="font-size: 12px; font-weight: 600; color: #64748b; text-transform: uppercase;">${label}</span>
                </div>
                <div style="font-size: 20px; font-weight: 700; color: #1e293b; margin-bottom: 4px;">
                    O/U ${prop.line}
                </div>
                <div style="font-size: 12px; color: #64748b; margin-bottom: 8px;">
                    Avg: ${prop.average.toFixed(1)} | ${prop.over_probability.toFixed(1)}% OVER
                </div>
                <div style="background: ${color}; color: white; padding: 4px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; text-align: center;">
                    ${prop.recommendation}
                </div>
            </div>
        `;
    }
    
    // Cache helpers
    getCached(key) {
        const cached = this.cache.get(key);
        if (cached && Date.now() - cached.timestamp < this.cacheDuration) {
            console.log(`📦 Using cached data for: ${key}`);
            return cached.data;
        }
        return null;
    }
    
    setCache(key, data) {
        this.cache.set(key, {
            data: data,
            timestamp: Date.now()
        });
    }
}

// Initialize global instance
if (typeof window !== 'undefined') {
    window.MultiSportPredictions = MultiSportPredictions;
    window.multiSportPredictions = new MultiSportPredictions();
    console.log('✅ Multi-Sport Predictions module loaded');
}
