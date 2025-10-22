// NBA Games Module
// Handles NBA game loading and display

class NBAGames {
    constructor() {
        this.games = [];
        this.currentWeek = this.getCurrentNBAWeek();
    }

    // Get current NBA week (NBA season runs October - April)
    getCurrentNBAWeek() {
        const now = new Date();
        const year = now.getFullYear();
        const month = now.getMonth() + 1; // 1-12
        
        // NBA season: October (month 10) to April (month 4)
        // Simplified week calculation
        if (month >= 10) {
            // Start of season (October - December)
            return Math.ceil((now.getDate() + (month - 10) * 30) / 7);
        } else if (month <= 4) {
            // End of season (January - April)
            return Math.ceil(((month + 2) * 30 + now.getDate()) / 7);
        } else {
            // Off-season
            return 0;
        }
    }

    // Fetch NBA games from API or use static schedule
    async loadGames() {
        console.log('🏀 Loading NBA games for current week...');
        
        // For now, use static upcoming games as placeholder
        // TODO: Replace with real NBA API (ESPN, NBA.com, etc.)
        this.games = this.getStaticNBAGames();
        
        return this.games;
    }

    // Static NBA games for demonstration
    // TODO: Replace with real API integration
    getStaticNBAGames() {
        const today = new Date();
        const dayOfWeek = today.getDay(); // 0 = Sunday, 6 = Saturday
        
        // Generate upcoming games for the week
        const upcomingGames = [
            // Tuesday games
            {
                gameId: 'nba_001',
                gameDate: this.getDateString(2 - dayOfWeek), // Tuesday
                gameTime: '7:30 PM ET',
                awayTeam: { code: 'LAL', name: 'Lakers', score: 0 },
                homeTeam: { code: 'BOS', name: 'Celtics', score: 0 },
                status: 'SCHEDULED',
                venue: 'TD Garden'
            },
            {
                gameId: 'nba_002',
                gameDate: this.getDateString(2 - dayOfWeek),
                gameTime: '8:00 PM ET',
                awayTeam: { code: 'GSW', name: 'Warriors', score: 0 },
                homeTeam: { code: 'PHX', name: 'Suns', score: 0 },
                status: 'SCHEDULED',
                venue: 'Footprint Center'
            },
            // Wednesday games
            {
                gameId: 'nba_003',
                gameDate: this.getDateString(3 - dayOfWeek),
                gameTime: '7:00 PM ET',
                awayTeam: { code: 'MIL', name: 'Bucks', score: 0 },
                homeTeam: { code: 'MIA', name: 'Heat', score: 0 },
                status: 'SCHEDULED',
                venue: 'Kaseya Center'
            },
            {
                gameId: 'nba_004',
                gameDate: this.getDateString(3 - dayOfWeek),
                gameTime: '7:30 PM ET',
                awayTeam: { code: 'DAL', name: 'Mavericks', score: 0 },
                homeTeam: { code: 'NYK', name: 'Knicks', score: 0 },
                status: 'SCHEDULED',
                venue: 'Madison Square Garden'
            },
            // Thursday games
            {
                gameId: 'nba_005',
                gameDate: this.getDateString(4 - dayOfWeek),
                gameTime: '7:30 PM ET',
                awayTeam: { code: 'DEN', name: 'Nuggets', score: 0 },
                homeTeam: { code: 'LAC', name: 'Clippers', score: 0 },
                status: 'SCHEDULED',
                venue: 'Crypto.com Arena'
            },
            {
                gameId: 'nba_006',
                gameDate: this.getDateString(4 - dayOfWeek),
                gameTime: '8:00 PM ET',
                awayTeam: { code: 'PHI', name: '76ers', score: 0 },
                homeTeam: { code: 'BKN', name: 'Nets', score: 0 },
                status: 'SCHEDULED',
                venue: 'Barclays Center'
            },
            // Friday games
            {
                gameId: 'nba_007',
                gameDate: this.getDateString(5 - dayOfWeek),
                gameTime: '7:00 PM ET',
                awayTeam: { code: 'MEM', name: 'Grizzlies', score: 0 },
                homeTeam: { code: 'ATL', name: 'Hawks', score: 0 },
                status: 'SCHEDULED',
                venue: 'State Farm Arena'
            },
            {
                gameId: 'nba_008',
                gameDate: this.getDateString(5 - dayOfWeek),
                gameTime: '8:30 PM ET',
                awayTeam: { code: 'SAC', name: 'Kings', score: 0 },
                homeTeam: { code: 'POR', name: 'Trail Blazers', score: 0 },
                status: 'SCHEDULED',
                venue: 'Moda Center'
            }
        ];

        return upcomingGames;
    }

    // Helper function to get date string
    getDateString(daysOffset) {
        const date = new Date();
        date.setDate(date.getDate() + daysOffset);
        const options = { weekday: 'short', month: 'short', day: 'numeric' };
        return date.toLocaleDateString('en-US', options);
    }

    // Render NBA games in game-centric UI format
    renderGames(containerId = 'game-centric-container') {
        const container = document.getElementById(containerId);
        if (!container) {
            console.error('❌ Container not found:', containerId);
            return;
        }

        if (this.games.length === 0) {
            container.innerHTML = `
                <div style="text-align: center; padding: 40px; background: rgba(255, 255, 255, 0.95); border-radius: 12px;">
                    <div style="font-size: 48px; margin-bottom: 16px;">🏀</div>
                    <h3 style="color: #1e293b;">No NBA games scheduled</h3>
                    <p style="color: #64748b;">Check back during the NBA season</p>
                </div>
            `;
            return;
        }

        let gamesHTML = `
            <div style="margin-bottom: 24px;">
                <h2 style="color: #1e293b; font-size: 28px; font-weight: 700; margin-bottom: 8px;">
                    🏀 NBA Games This Week
                </h2>
                <p style="color: #64748b; font-size: 14px;">
                    Select a game to make predictions • ${this.games.length} games available
                </p>
            </div>
            <div class="nba-games-grid">
        `;

        // Group games by date
        const gamesByDate = {};
        this.games.forEach(game => {
            if (!gamesByDate[game.gameDate]) {
                gamesByDate[game.gameDate] = [];
            }
            gamesByDate[game.gameDate].push(game);
        });

        // Render games grouped by date
        for (const [date, games] of Object.entries(gamesByDate)) {
            gamesHTML += `
                <div class="nba-date-section">
                    <h3 class="nba-date-header">${date}</h3>
                    <div class="nba-games-row">
            `;

            games.forEach(game => {
                gamesHTML += this.createNBAGameCard(game);
            });

            gamesHTML += `
                    </div>
                </div>
            `;
        }

        gamesHTML += `</div>`;

        container.innerHTML = gamesHTML;

        // Add click handlers
        this.attachGameClickHandlers();
    }

    // Create individual NBA game card
    createNBAGameCard(game) {
        const statusBadge = game.status === 'LIVE' 
            ? '<span class="nba-status-badge live">🔴 LIVE</span>'
            : '<span class="nba-status-badge scheduled">⏰ ' + game.gameTime + '</span>';

        return `
            <div class="nba-game-card" data-game-id="${game.gameId}" onclick="selectNBAGame('${game.gameId}')">
                ${statusBadge}
                <div class="nba-game-teams">
                    <div class="nba-team away">
                        <div class="nba-team-logo">🏀</div>
                        <div class="nba-team-info">
                            <div class="nba-team-code">${game.awayTeam.code}</div>
                            <div class="nba-team-name">${game.awayTeam.name}</div>
                        </div>
                        ${game.status === 'LIVE' ? `<div class="nba-score">${game.awayTeam.score}</div>` : ''}
                    </div>
                    <div class="nba-vs">@</div>
                    <div class="nba-team home">
                        <div class="nba-team-logo">🏀</div>
                        <div class="nba-team-info">
                            <div class="nba-team-code">${game.homeTeam.code}</div>
                            <div class="nba-team-name">${game.homeTeam.name}</div>
                        </div>
                        ${game.status === 'LIVE' ? `<div class="nba-score">${game.homeTeam.score}</div>` : ''}
                    </div>
                </div>
                <div class="nba-game-venue">${game.venue}</div>
            </div>
        `;
    }

    // Attach click handlers to game cards
    attachGameClickHandlers() {
        // Game click handlers will be added here
        console.log('🏀 NBA game cards rendered and ready for interaction');
    }
}

// Global function to select NBA game
function selectNBAGame(gameId) {
    console.log('🏀 NBA Game selected:', gameId);
    // TODO: Implement game selection logic
    // This will eventually load player stats and enable predictions
}

// Export for use in main app
window.NBAGames = NBAGames;

console.log('🏀 NBA Games module loaded!');
