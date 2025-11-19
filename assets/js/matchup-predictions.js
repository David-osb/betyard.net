// Display TD Scorers in Predictions Panel (NEW TWO-TEAM CARD LAYOUT)
async function displayMatchupPredictions(awayTeam, homeTeam) {
    console.log(`🏈 Loading matchup predictions for ${awayTeam} @ ${homeTeam}...`);
    
    // Hide default state, show matchup view
    const defaultState = document.getElementById('predictions-default-state');
    const matchupView = document.getElementById('predictions-matchup-view');
    if (defaultState) defaultState.style.display = 'none';
    if (matchupView) matchupView.style.display = 'block';
    
    // Update matchup title
    const matchupTitle = document.getElementById('matchup-title');
    if (matchupTitle) {
        matchupTitle.innerHTML = `<span style="color: #3b82f6;">${awayTeam}</span> @ <span style="color: #ef4444;">${homeTeam}</span>`;
    }
    
    // Show loading state
    document.getElementById('away-team-content').innerHTML = '<div style="text-align: center; padding: 40px;"><div class="loading-spinner"></div><p style="color: #64748b;">Loading predictions...</p></div>';
    document.getElementById('home-team-content').innerHTML = '<div style="text-align: center; padding: 40px;"><div class="loading-spinner"></div><p style="color: #64748b;">Loading predictions...</p></div>';
    
    // Update team headers
    document.getElementById('away-team-name').textContent = awayTeam;
    document.getElementById('home-team-name').textContent = homeTeam;
    
    // Fetch all players from both teams
    const awayPlayers = [];
    const homePlayers = [];
    
    // Get team codes from team names
    const getTeamCode = (teamName) => {
        const teamMap = {
            'Arizona Cardinals': 'ARI', 'Atlanta Falcons': 'ATL', 'Baltimore Ravens': 'BAL',
            'Buffalo Bills': 'BUF', 'Carolina Panthers': 'CAR', 'Chicago Bears': 'CHI',
            'Cincinnati Bengals': 'CIN', 'Cleveland Browns': 'CLE', 'Dallas Cowboys': 'DAL',
            'Denver Broncos': 'DEN', 'Detroit Lions': 'DET', 'Green Bay Packers': 'GB',
            'Houston Texans': 'HOU', 'Indianapolis Colts': 'IND', 'Jacksonville Jaguars': 'JAX',
            'Kansas City Chiefs': 'KC', 'Las Vegas Raiders': 'LV', 'Los Angeles Chargers': 'LAC',
            'Los Angeles Rams': 'LAR', 'Miami Dolphins': 'MIA', 'Minnesota Vikings': 'MIN',
            'New England Patriots': 'NE', 'New Orleans Saints': 'NO', 'New York Giants': 'NYG',
            'New York Jets': 'NYJ', 'Philadelphia Eagles': 'PHI', 'Pittsburgh Steelers': 'PIT',
            'San Francisco 49ers': 'SF', 'Seattle Seahawks': 'SEA', 'Tampa Bay Buccaneers': 'TB',
            'Tennessee Titans': 'TEN', 'Washington Commanders': 'WAS'
        };
        return teamMap[teamName] || teamName;
    };
    
    const awayCode = getTeamCode(awayTeam);
    const homeCode = getTeamCode(homeTeam);
    
    // Fetch top players from backend for each team
    try {
        const [awayResponse, homeResponse] = await Promise.all([
            fetch(`https://betyard-ml-backend.onrender.com/players/team/${awayCode}`),
            fetch(`https://betyard-ml-backend.onrender.com/players/team/${homeCode}`)
        ]);
        
        const awayData = await awayResponse.json();
        const homeData = await homeResponse.json();
        
        if (awayData.success && homeData.success) {
            // Fetch predictions for each player we found
            for (const position of ['QB', 'RB', 'WR', 'TE']) {
                const awayPlayer = awayData.players[position];
                const homePlayer = homeData.players[position];
                
                if (awayPlayer) {
                    try {
                        const prediction = await window.BetYardML.getPrediction(
                            awayPlayer.name,
                            awayCode,
                            homeCode,
                            position
                        );
                        if (prediction) {
                            awayPlayers.push({ position, ...prediction });
                        }
                    } catch (error) {
                        console.warn(`Failed to fetch ${position} prediction for ${awayPlayer.name}:`, error);
                    }
                }
                
                if (homePlayer) {
                    try {
                        const prediction = await window.BetYardML.getPrediction(
                            homePlayer.name,
                            homeCode,
                            awayCode,
                            position
                        );
                        if (prediction) {
                            homePlayers.push({ position, ...prediction });
                        }
                    } catch (error) {
                        console.warn(`Failed to fetch ${position} prediction for ${homePlayer.name}:`, error);
                    }
                }
            }
        } else {
            console.error('Failed to fetch team rosters:', awayData, homeData);
        }
    } catch (error) {
        console.error('Error fetching team players:', error);
    }
    
    // Calculate game-level predictions from player data
    const calculateGamePredictions = (awayData, homeData, awayTeamName, homeTeamName) => {
        // Sum up predicted touchdowns from all players
        const awayTotalTDs = awayData.reduce((sum, p) => sum + (p.touchdowns || p.passing_touchdowns || p.anytime_td_probability || 0), 0);
        const homeTotalTDs = homeData.reduce((sum, p) => sum + (p.touchdowns || p.passing_touchdowns || p.anytime_td_probability || 0), 0);
        
        // Estimate total points (TDs * 7 + field goals estimate)
        const awayPoints = Math.round(awayTotalTDs * 7 + 6); // Add ~6 for FGs
        const homePoints = Math.round(homeTotalTDs * 7 + 6);
        
        // Moneyline prediction
        const pointDiff = homePoints - awayPoints;
        const moneylineTeam = pointDiff > 0 ? homeTeamName : awayTeamName;
        const moneylineConfidence = Math.min(55 + Math.abs(pointDiff) * 2, 85); // 55-85% range
        
        // Spread prediction
        const spreadLine = pointDiff;
        const spreadTeam = spreadLine > 0 ? `${homeTeamName} -${Math.abs(spreadLine)}` : `${awayTeamName} -${Math.abs(spreadLine)}`;
        const spreadConfidence = Math.min(60 + Math.abs(pointDiff), 80);
        
        // Total prediction
        const totalPoints = awayPoints + homePoints;
        const totalPrediction = totalPoints > 45 ? 'OVER' : 'UNDER';
        const totalLine = Math.round(totalPoints / 0.5) * 0.5; // Round to nearest 0.5
        const totalConfidence = Math.min(65, 75);
        
        return {
            moneyline: { team: moneylineTeam, confidence: moneylineConfidence.toFixed(0) },
            spread: { line: spreadTeam, confidence: spreadConfidence.toFixed(0) },
            total: { prediction: `${totalPrediction} ${totalLine}`, confidence: totalConfidence.toFixed(0), awayPoints, homePoints }
        };
    };
    
    const gamePreds = calculateGamePredictions(awayPlayers, homePlayers, awayTeam, homeTeam);
    
    // Update game-level predictions
    document.getElementById('ml-prediction').innerHTML = `<span style="color: #10b981;">${gamePreds.moneyline.team}</span>`;
    document.getElementById('ml-confidence').textContent = `${gamePreds.moneyline.confidence}% Confidence`;
    
    document.getElementById('spread-prediction').innerHTML = `<span style="color: #3b82f6;">${gamePreds.spread.line}</span>`;
    document.getElementById('spread-confidence').textContent = `${gamePreds.spread.confidence}% Confidence`;
    
    document.getElementById('total-prediction').innerHTML = `<span style="color: #f59e0b;">${gamePreds.total.prediction}</span>`;
    document.getElementById('total-confidence').innerHTML = `${gamePreds.total.confidence}% Confidence <span style="opacity: 0.6; font-size: 11px;">(${gamePreds.total.awayPoints}-${gamePreds.total.homePoints})</span>`;
    
    // Generate player cards for each team
    const generatePlayerCard = (player) => {
        const positionIcons = { QB: 'psychology', RB: 'directions_run', WR: 'sports', TE: 'sports_football' };
        const positionColors = { QB: '#8b5cf6', RB: '#10b981', WR: '#f59e0b', TE: '#3b82f6' };
        
        const anytimeTD = (player.anytime_td_probability * 100).toFixed(1);
        const firstTD = (player.first_td_probability * 100).toFixed(1);
        const multiTD = (player.multi_td_probability * 100).toFixed(1);
        const tdOdds = player.anytime_td_probability > 0 ? Math.round((1 / player.anytime_td_probability - 1) * 100) : 999;
        
        // Position-specific stats
        let mainStat = '';
        let secondaryStat = '';
        if (player.position === 'QB') {
            mainStat = `${player.passingYards || player.passing_yards || '--'} Pass Yds`;
            secondaryStat = `${player.touchdowns || player.passing_touchdowns || '--'} Pass TDs`;
        } else if (player.position === 'RB') {
            mainStat = `${player.rushingYards || player.rushing_yards || '--'} Rush Yds`;
            secondaryStat = `${player.receptions || '--'} Rec`;
        } else if (player.position === 'WR' || player.position === 'TE') {
            mainStat = `${player.receivingYards || player.receiving_yards || '--'} Rec Yds`;
            secondaryStat = `${player.receptions || '--'} Rec`;
        }
        
        return `
            <div style="background: #f8fafc; border-radius: 10px; padding: 16px; margin-bottom: 16px; border-left: 4px solid ${positionColors[player.position]};">
                <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 12px;">
                    <div style="flex: 1;">
                        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 4px;">
                            <i class="material-icons" style="color: ${positionColors[player.position]}; font-size: 20px;">${positionIcons[player.position]}</i>
                            <span style="font-size: 18px; font-weight: 700; color: #1e293b;">${player.player_name || 'Unknown'}</span>
                            <span style="background: ${positionColors[player.position]}; color: white; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600;">${player.position}</span>
                        </div>
                        <div style="font-size: 14px; color: #64748b; margin-left: 28px;">
                            ${mainStat} • ${secondaryStat}
                        </div>
                    </div>
                </div>
                
                <!-- TD Probabilities -->
                <div style="background: white; border-radius: 8px; padding: 12px; margin-top: 12px;">
                    <div style="font-size: 12px; font-weight: 600; color: #64748b; margin-bottom: 8px; text-transform: uppercase;">Touchdown Probabilities</div>
                    <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px;">
                        <div style="text-align: center;">
                            <div style="font-size: 11px; color: #64748b; margin-bottom: 4px;">Anytime TD</div>
                            <div style="font-size: 20px; font-weight: 700; color: #10b981;">${anytimeTD}%</div>
                            <div style="font-size: 10px; color: #6b7280;">+${tdOdds}</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-size: 11px; color: #64748b; margin-bottom: 4px;">First TD</div>
                            <div style="font-size: 20px; font-weight: 700; color: #3b82f6;">${firstTD}%</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-size: 11px; color: #64748b; margin-bottom: 4px;">2+ TDs</div>
                            <div style="font-size: 20px; font-weight: 700; color: #f59e0b;">${multiTD}%</div>
                        </div>
                    </div>
                    ${player.defense_adjustment && player.defense_adjustment !== 1.0 ? 
                        `<div style="margin-top: 8px; padding: 6px; background: ${player.defense_adjustment > 1.0 ? '#d1fae5' : '#fee2e2'}; border-radius: 4px; text-align: center;">
                            <span style="font-size: 11px; color: ${player.defense_adjustment > 1.0 ? '#059669' : '#dc2626'}; font-weight: 600;">
                                ${player.defense_adjustment > 1.0 ? '↗' : '↘'} ${player.defense_adjustment > 1.0 ? '+' : ''}${((player.defense_adjustment - 1.0) * 100).toFixed(0)}% vs Defense
                            </span>
                        </div>` 
                        : ''}
                </div>
            </div>
        `;
    };
    
    // Populate away team
    if (awayPlayers.length > 0) {
        document.getElementById('away-team-content').innerHTML = awayPlayers.map(generatePlayerCard).join('');
    } else {
        document.getElementById('away-team-content').innerHTML = '<div style="text-align: center; padding: 40px; color: #94a3b8;"><i class="material-icons" style="font-size: 48px; opacity: 0.5;">error_outline</i><p>No predictions available</p></div>';
    }
    
    // Populate home team
    if (homePlayers.length > 0) {
        document.getElementById('home-team-content').innerHTML = homePlayers.map(generatePlayerCard).join('');
    } else {
        document.getElementById('home-team-content').innerHTML = '<div style="text-align: center; padding: 40px; color: #94a3b8;"><i class="material-icons" style="font-size: 48px; opacity: 0.5;">error_outline</i><p>No predictions available</p></div>';
    }
    
    console.log(`✅ Loaded ${awayPlayers.length} away players and ${homePlayers.length} home players`);
    console.log(`📊 Game Predictions: ML: ${gamePreds.moneyline.team}, Spread: ${gamePreds.spread.line}, Total: ${gamePreds.total.prediction}`);
}

// Back to Games button function
function backToGames() {
    // Hide predictions panel
    const predictionsPanel = document.getElementById('predictions-panel');
    if (predictionsPanel) {
        predictionsPanel.style.display = 'none';
        predictionsPanel.classList.remove('active');
    }
    
    // Hide matchup view, show default state
    const defaultState = document.getElementById('predictions-default-state');
    const matchupView = document.getElementById('predictions-matchup-view');
    if (defaultState) defaultState.style.display = 'block';
    if (matchupView) matchupView.style.display = 'none';
    
    // Show game cards and UI elements
    const gameCentricContainer = document.getElementById('game-centric-container');
    const newsContainer = document.getElementById('news-container');
    const sportSelector = document.getElementById('sport-selector');
    
    if (gameCentricContainer) gameCentricContainer.style.display = 'block';
    if (newsContainer) newsContainer.style.display = 'block';
    if (sportSelector) sportSelector.style.display = 'block';
    
    // Scroll to top
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Make functions globally available immediately
window.displayMatchupPredictions = displayMatchupPredictions;
window.backToGames = backToGames;
console.log('✅ displayMatchupPredictions and backToGames functions loaded and ready');
