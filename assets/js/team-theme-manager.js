/* ===============================================
 * NFL TEAM COLORS DATABASE & DYNAMIC THEMING
 * =============================================== */

const NFL_TEAM_COLORS = {
    'ARI': {
        name: 'Arizona Cardinals',
        primary: '#97233F',
        secondary: '#000000',
        accent: '#FFB612',
        light: '#FFFFFF',
        gradient: 'linear-gradient(135deg, #97233F 0%, #000000 100%)'
    },
    'ATL': {
        name: 'Atlanta Falcons', 
        primary: '#A71930',
        secondary: '#000000',
        accent: '#A5ACAF',
        light: '#FFFFFF',
        gradient: 'linear-gradient(135deg, #A71930 0%, #000000 100%)'
    },
    'BAL': {
        name: 'Baltimore Ravens',
        primary: '#241773',
        secondary: '#000000', 
        accent: '#9E7C0C',
        light: '#FFFFFF',
        gradient: 'linear-gradient(135deg, #241773 0%, #000000 100%)'
    },
    'BUF': {
        name: 'Buffalo Bills',
        primary: '#00338D',
        secondary: '#C60C30',
        accent: '#FFFFFF',
        light: '#F8F9FA',
        gradient: 'linear-gradient(135deg, #00338D 0%, #C60C30 100%)'
    },
    'CAR': {
        name: 'Carolina Panthers',
        primary: '#0085CA',
        secondary: '#000000',
        accent: '#BFC0BF',
        light: '#FFFFFF',
        gradient: 'linear-gradient(135deg, #0085CA 0%, #000000 100%)'
    },
    'CHI': {
        name: 'Chicago Bears',
        primary: '#0B162A',
        secondary: '#C83803',
        accent: '#FFFFFF',
        light: '#F8F9FA',
        gradient: 'linear-gradient(135deg, #0B162A 0%, #C83803 100%)'
    },
    'CIN': {
        name: 'Cincinnati Bengals',
        primary: '#FB4F14',
        secondary: '#000000',
        accent: '#FFFFFF',
        light: '#F8F9FA',
        gradient: 'linear-gradient(135deg, #FB4F14 0%, #000000 100%)'
    },
    'CLE': {
        name: 'Cleveland Browns',
        primary: '#311D00',
        secondary: '#FF3C00',
        accent: '#FFFFFF',
        light: '#F8F9FA',
        gradient: 'linear-gradient(135deg, #311D00 0%, #FF3C00 100%)'
    },
    'DAL': {
        name: 'Dallas Cowboys',
        primary: '#003594',
        secondary: '#041E42',
        accent: '#869397',
        light: '#FFFFFF',
        gradient: 'linear-gradient(135deg, #003594 0%, #041E42 100%)'
    },
    'DEN': {
        name: 'Denver Broncos',
        primary: '#FB4F14',
        secondary: '#002244',
        accent: '#FFFFFF',
        light: '#F8F9FA',
        gradient: 'linear-gradient(135deg, #FB4F14 0%, #002244 100%)'
    },
    'DET': {
        name: 'Detroit Lions',
        primary: '#0076B6',
        secondary: '#B0B7BC',
        accent: '#000000',
        light: '#FFFFFF',
        gradient: 'linear-gradient(135deg, #0076B6 0%, #B0B7BC 100%)'
    },
    'GB': {
        name: 'Green Bay Packers',
        primary: '#203731',
        secondary: '#FFB612',
        accent: '#FFFFFF',
        light: '#F8F9FA',
        gradient: 'linear-gradient(135deg, #203731 0%, #FFB612 100%)'
    },
    'HOU': {
        name: 'Houston Texans',
        primary: '#03202F',
        secondary: '#A71930',
        accent: '#FFFFFF',
        light: '#F8F9FA',
        gradient: 'linear-gradient(135deg, #03202F 0%, #A71930 100%)'
    },
    'IND': {
        name: 'Indianapolis Colts',
        primary: '#002C5F',
        secondary: '#A2AAAD',
        accent: '#FFFFFF',
        light: '#F8F9FA',
        gradient: 'linear-gradient(135deg, #002C5F 0%, #A2AAAD 100%)'
    },
    'JAX': {
        name: 'Jacksonville Jaguars',
        primary: '#006778',
        secondary: '#9F792C',
        accent: '#000000',
        light: '#FFFFFF',
        gradient: 'linear-gradient(135deg, #006778 0%, #9F792C 100%)'
    },
    'KC': {
        name: 'Kansas City Chiefs',
        primary: '#E31837',
        secondary: '#FFB612',
        accent: '#FFFFFF',
        light: '#F8F9FA',
        gradient: 'linear-gradient(135deg, #E31837 0%, #FFB612 100%)'
    },
    'LAC': {
        name: 'Los Angeles Chargers',
        primary: '#0080C6',
        secondary: '#FFC20E',
        accent: '#FFFFFF',
        light: '#F8F9FA',
        gradient: 'linear-gradient(135deg, #0080C6 0%, #FFC20E 100%)'
    },
    'LAR': {
        name: 'Los Angeles Rams',
        primary: '#003594',
        secondary: '#FFA300',
        accent: '#FFFFFF',
        light: '#F8F9FA',
        gradient: 'linear-gradient(135deg, #003594 0%, #FFA300 100%)'
    },
    'LV': {
        name: 'Las Vegas Raiders',
        primary: '#000000',
        secondary: '#A5ACAF',
        accent: '#FFFFFF',
        light: '#F8F9FA',
        gradient: 'linear-gradient(135deg, #000000 0%, #A5ACAF 100%)'
    },
    'MIA': {
        name: 'Miami Dolphins',
        primary: '#008E97',
        secondary: '#FC4C02',
        accent: '#005778',
        light: '#FFFFFF',
        gradient: 'linear-gradient(135deg, #008E97 0%, #FC4C02 100%)'
    },
    'MIN': {
        name: 'Minnesota Vikings',
        primary: '#4F2683',
        secondary: '#FFC62F',
        accent: '#FFFFFF',
        light: '#F8F9FA',
        gradient: 'linear-gradient(135deg, #4F2683 0%, #FFC62F 100%)'
    },
    'NE': {
        name: 'New England Patriots',
        primary: '#002244',
        secondary: '#C60C30',
        accent: '#B0B7BC',
        light: '#FFFFFF',
        gradient: 'linear-gradient(135deg, #002244 0%, #C60C30 100%)'
    },
    'NO': {
        name: 'New Orleans Saints',
        primary: '#D3BC8D',
        secondary: '#101820',
        accent: '#FFFFFF',
        light: '#F8F9FA',
        gradient: 'linear-gradient(135deg, #D3BC8D 0%, #101820 100%)'
    },
    'NYG': {
        name: 'New York Giants',
        primary: '#0B2265',
        secondary: '#A71930',
        accent: '#A5ACAF',
        light: '#FFFFFF',
        gradient: 'linear-gradient(135deg, #0B2265 0%, #A71930 100%)'
    },
    'NYJ': {
        name: 'New York Jets',
        primary: '#125740',
        secondary: '#000000',
        accent: '#FFFFFF',
        light: '#F8F9FA',
        gradient: 'linear-gradient(135deg, #125740 0%, #000000 100%)'
    },
    'PHI': {
        name: 'Philadelphia Eagles',
        primary: '#004C54',
        secondary: '#A5ACAF',
        accent: '#ACC0C6',
        light: '#FFFFFF',
        gradient: 'linear-gradient(135deg, #004C54 0%, #A5ACAF 100%)'
    },
    'PIT': {
        name: 'Pittsburgh Steelers',
        primary: '#FFB612',
        secondary: '#101820',
        accent: '#A5ACAF',
        light: '#FFFFFF',
        gradient: 'linear-gradient(135deg, #FFB612 0%, #101820 100%)'
    },
    'SEA': {
        name: 'Seattle Seahawks',
        primary: '#002244',
        secondary: '#69BE28',
        accent: '#A5ACAF',
        light: '#FFFFFF',
        gradient: 'linear-gradient(135deg, #002244 0%, #69BE28 100%)'
    },
    'SF': {
        name: 'San Francisco 49ers',
        primary: '#AA0000',
        secondary: '#B3995D',
        accent: '#FFFFFF',
        light: '#F8F9FA',
        gradient: 'linear-gradient(135deg, #AA0000 0%, #B3995D 100%)'
    },
    'TB': {
        name: 'Tampa Bay Buccaneers',
        primary: '#D50A0A',
        secondary: '#FF7900',
        accent: '#0A0A08',
        light: '#FFFFFF',
        gradient: 'linear-gradient(135deg, #D50A0A 0%, #FF7900 100%)'
    },
    'TEN': {
        name: 'Tennessee Titans',
        primary: '#0C2340',
        secondary: '#4B92DB',
        accent: '#C8102E',
        light: '#FFFFFF',
        gradient: 'linear-gradient(135deg, #0C2340 0%, #4B92DB 100%)'
    },
    'WSH': {
        name: 'Washington Commanders',
        primary: '#5A1414',
        secondary: '#FFB612',
        accent: '#FFFFFF',
        light: '#F8F9FA',
        gradient: 'linear-gradient(135deg, #5A1414 0%, #FFB612 100%)'
    },
    'DEFAULT': {
        name: 'BetYard Default',
        primary: '#3b82f6',
        secondary: '#1e40af',
        accent: '#8b5cf6',
        light: '#FFFFFF',
        gradient: 'linear-gradient(135deg, #3b82f6 0%, #1e40af 100%)'
    }
};

/* ===============================================
 * DYNAMIC TEAM THEME APPLICATION
 * =============================================== */

class TeamThemeManager {
    constructor() {
        this.currentTheme = 'DEFAULT';
        this.styleElement = null;
        this.init();
    }

    init() {
        // Create dynamic style element
        this.styleElement = document.createElement('style');
        this.styleElement.id = 'team-theme-styles';
        document.head.appendChild(this.styleElement);
        
        // Load user's favorite team theme
        this.loadUserTheme();
    }

    loadUserTheme() {
        const userData = JSON.parse(localStorage.getItem('userData') || '{}');
        const favoriteTeam = userData.favoriteTeam || 'DEFAULT';
        
        console.log('🎨 Loading team theme for:', favoriteTeam);
        this.applyTeamTheme(favoriteTeam);
    }

    applyTeamTheme(teamCode) {
        const teamColors = NFL_TEAM_COLORS[teamCode] || NFL_TEAM_COLORS['DEFAULT'];
        this.currentTheme = teamCode;
        
        console.log('🏈 Applying theme for:', teamColors.name);
        
        const dynamicCSS = `
            /* Dynamic Team Theme: ${teamColors.name} */
            :root {
                --team-primary: ${teamColors.primary};
                --team-secondary: ${teamColors.secondary};
                --team-accent: ${teamColors.accent};
                --team-light: ${teamColors.light};
                --team-gradient: ${teamColors.gradient};
            }

            /* Header Theming */
            .mdl-layout__header {
                background: ${teamColors.gradient} !important;
            }

            /* Sidebar Theming */
            .sidebar-header {
                background: ${teamColors.gradient} !important;
            }

            .sidebar-item.active {
                background: linear-gradient(90deg, ${teamColors.primary}20 0%, ${teamColors.primary}10 100%) !important;
                color: ${teamColors.primary} !important;
                border-right-color: ${teamColors.primary} !important;
            }

            .sidebar-item:hover {
                background: linear-gradient(90deg, ${teamColors.primary}15 0%, ${teamColors.primary}08 100%) !important;
                color: ${teamColors.primary} !important;
            }

            /* User Profile Card */
            .user-profile-card {
                background: ${teamColors.primary}12 !important;
                border-color: ${teamColors.primary}30 !important;
            }

            /* Buttons and CTAs */
            .btn-primary, .mdl-button--raised.mdl-button--colored {
                background: ${teamColors.gradient} !important;
                color: ${teamColors.light} !important;
            }

            .btn-primary:hover {
                background: ${teamColors.primary} !important;
                box-shadow: 0 6px 20px ${teamColors.primary}40 !important;
            }

            /* Prediction Cards */
            .prediction-card {
                border-left: 4px solid ${teamColors.primary} !important;
            }

            .prediction-card:hover {
                border-left-color: ${teamColors.secondary} !important;
                box-shadow: 0 8px 25px ${teamColors.primary}20 !important;
            }

            /* Game Cards */
            .game-card.selected {
                border-color: ${teamColors.primary} !important;
                background: ${teamColors.primary}10 !important;
            }

            .game-card:hover {
                border-color: ${teamColors.primary}60 !important;
                box-shadow: 0 4px 15px ${teamColors.primary}25 !important;
            }

            /* Progress Bars and Meters */
            .progress-bar-fill, .confidence-meter-fill {
                background: ${teamColors.gradient} !important;
            }

            /* Links and Accents */
            a, .link-color {
                color: ${teamColors.primary} !important;
            }

            a:hover, .link-color:hover {
                color: ${teamColors.secondary} !important;
            }

            /* News Cards */
            .news-card:hover {
                border-left-color: ${teamColors.primary} !important;
            }

            /* Stats and Highlights */
            .stat-highlight {
                color: ${teamColors.primary} !important;
            }

            .stat-card {
                border-top: 3px solid ${teamColors.primary} !important;
            }

            /* Loading Spinners */
            .spinner, .loading-spinner {
                border-top-color: ${teamColors.primary} !important;
            }

            /* Form Elements */
            input:focus, textarea:focus, select:focus {
                border-color: ${teamColors.primary} !important;
                box-shadow: 0 0 0 3px ${teamColors.primary}20 !important;
            }

            /* Team Spirit Animation */
            @keyframes teamPulse {
                0%, 100% { 
                    box-shadow: 0 0 0 0 ${teamColors.primary}40; 
                }
                50% { 
                    box-shadow: 0 0 0 10px ${teamColors.primary}00; 
                }
            }

            .team-spirit-pulse {
                animation: teamPulse 2s infinite;
            }

            /* Mobile Theme Adjustments */
            @media (max-width: 768px) {
                .mobile-team-header {
                    background: ${teamColors.gradient} !important;
                }
                
                .mobile-nav-item.active {
                    background: ${teamColors.primary} !important;
                    color: ${teamColors.light} !important;
                }
            }
        `;

        // Apply the dynamic styles
        this.styleElement.textContent = dynamicCSS;
        
        // Store current theme
        localStorage.setItem('currentTeamTheme', teamCode);
        
        // Track theme change
        if (typeof gtag !== 'undefined') {
            gtag('event', 'theme_change', {
                event_category: 'customization',
                event_label: teamColors.name
            });
        }

        console.log('✅ Team theme applied:', teamColors.name);
    }

    // Method to change theme programmatically
    changeTheme(teamCode) {
        this.applyTeamTheme(teamCode);
        
        // Update user data
        const userData = JSON.parse(localStorage.getItem('userData') || '{}');
        userData.favoriteTeam = teamCode;
        localStorage.setItem('userData', JSON.stringify(userData));
    }

    // Get current team colors
    getCurrentColors() {
        return NFL_TEAM_COLORS[this.currentTheme] || NFL_TEAM_COLORS['DEFAULT'];
    }

    // Preview theme without saving
    previewTheme(teamCode) {
        this.applyTeamTheme(teamCode);
    }

    // Restore saved theme
    restoreTheme() {
        this.loadUserTheme();
    }
}

// Initialize team theme manager when page loads
let teamThemeManager;

document.addEventListener('DOMContentLoaded', function() {
    setTimeout(() => {
        teamThemeManager = new TeamThemeManager();
        console.log('🎨 Team Theme Manager initialized');
    }, 500);
});

// Export for global use
window.TeamThemeManager = TeamThemeManager;
window.NFL_TEAM_COLORS = NFL_TEAM_COLORS;