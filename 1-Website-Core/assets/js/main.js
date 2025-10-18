// Main NFL QB Predictor Application JavaScript
// Contains: Data management, API calls, UI logic, prediction engine

console.log('📄 Main application JavaScript loaded from external file');

// 2025 NFL Schedule Data - ACCURATE FROM PRO FOOTBALL REFERENCE
// Based on official Pro Football Reference data (Current date: October 16, 2025 - Week 7 in progress)
const nfl2025Schedule = {
	6: [
		{homeTeam: 'New York Giants', awayTeam: 'Philadelphia Eagles', date: 'Oct 9, 2025', time: '8:15 PM ET', tv: 'TNF', result: 'NYG 34, PHI 17'},
		{homeTeam: 'New York Jets', awayTeam: 'Denver Broncos', date: 'Oct 12, 2025', time: '9:30 AM ET', tv: 'CBS', result: 'DEN 13, NYJ 11'},
		{homeTeam: 'Indianapolis Colts', awayTeam: 'Arizona Cardinals', date: 'Oct 12, 2025', time: '1:00 PM ET', tv: 'CBS', result: 'IND 31, ARI 27'},
		{homeTeam: 'Baltimore Ravens', awayTeam: 'Los Angeles Rams', date: 'Oct 12, 2025', time: '1:00 PM ET', tv: 'CBS', result: 'LAR 17, BAL 3'},
		{homeTeam: 'Carolina Panthers', awayTeam: 'Dallas Cowboys', date: 'Oct 12, 2025', time: '1:00 PM ET', tv: 'FOX', result: 'CAR 30, DAL 27'},
		{homeTeam: 'Pittsburgh Steelers', awayTeam: 'Cleveland Browns', date: 'Oct 12, 2025', time: '1:00 PM ET', tv: 'CBS', result: 'PIT 23, CLE 9'},
		{homeTeam: 'Miami Dolphins', awayTeam: 'Los Angeles Chargers', date: 'Oct 12, 2025', time: '1:00 PM ET', tv: 'CBS', result: 'LAC 29, MIA 27'},
		{homeTeam: 'New Orleans Saints', awayTeam: 'New England Patriots', date: 'Oct 12, 2025', time: '1:00 PM ET', tv: 'FOX', result: 'NE 25, NO 19'},
		{homeTeam: 'Jacksonville Jaguars', awayTeam: 'Seattle Seahawks', date: 'Oct 12, 2025', time: '1:00 PM ET', tv: 'FOX', result: 'SEA 20, JAX 12'},
		{homeTeam: 'Las Vegas Raiders', awayTeam: 'Tennessee Titans', date: 'Oct 12, 2025', time: '4:05 PM ET', tv: 'CBS', result: 'LV 20, TEN 10'},
		{homeTeam: 'Green Bay Packers', awayTeam: 'Cincinnati Bengals', date: 'Oct 12, 2025', time: '4:25 PM ET', tv: 'FOX', result: 'GB 27, CIN 18'},
		{homeTeam: 'Tampa Bay Buccaneers', awayTeam: 'San Francisco 49ers', date: 'Oct 12, 2025', time: '4:25 PM ET', tv: 'FOX', result: 'TB 30, SF 19'},
		{homeTeam: 'Kansas City Chiefs', awayTeam: 'Detroit Lions', date: 'Oct 12, 2025', time: '8:20 PM ET', tv: 'NBC', result: 'KC 30, DET 17'},
		{homeTeam: 'Atlanta Falcons', awayTeam: 'Buffalo Bills', date: 'Oct 13, 2025', time: '7:15 PM ET', tv: 'ESPN'},
		{homeTeam: 'Washington Commanders', awayTeam: 'Chicago Bears', date: 'Oct 13, 2025', time: '8:15 PM ET', tv: 'ABC'}
	],
	// ... (other weeks omitted for brevity, but should be included in full modularization)
};

// Team name mappings for schedule
const teamMappings = {
	'buffalo-bills': 'Buffalo Bills',
	'miami-dolphins': 'Miami Dolphins',
	'new-england-patriots': 'New England Patriots',
	'new-york-jets': 'New York Jets',
	'baltimore-ravens': 'Baltimore Ravens',
	'cincinnati-bengals': 'Cincinnati Bengals',
	'cleveland-browns': 'Cleveland Browns',
	'pittsburgh-steelers': 'Pittsburgh Steelers',
	'houston-texans': 'Houston Texans',
	'indianapolis-colts': 'Indianapolis Colts',
	'jacksonville-jaguars': 'Jacksonville Jaguars',
	'tennessee-titans': 'Tennessee Titans',
	'denver-broncos': 'Denver Broncos',
	'kansas-city-chiefs': 'Kansas City Chiefs',
	'las-vegas-raiders': 'Las Vegas Raiders',
	'los-angeles-chargers': 'Los Angeles Chargers',
	'dallas-cowboys': 'Dallas Cowboys',
	'new-york-giants': 'New York Giants',
	'philadelphia-eagles': 'Philadelphia Eagles',
	'washington-commanders': 'Washington Commanders',
	'chicago-bears': 'Chicago Bears',
	'detroit-lions': 'Detroit Lions',
	'green-bay-packers': 'Green Bay Packers',
	'minnesota-vikings': 'Minnesota Vikings',
	'atlanta-falcons': 'Atlanta Falcons',
	'carolina-panthers': 'Carolina Panthers',
	'new-orleans-saints': 'New Orleans Saints',
	'tampa-bay-buccaneers': 'Tampa Bay Buccaneers',
	'arizona-cardinals': 'Arizona Cardinals',
	'los-angeles-rams': 'Los Angeles Rams',
	'san-francisco-49ers': 'San Francisco 49ers',
	'seattle-seahawks': 'Seattle Seahawks'
};

// ... (move all other main logic from the HTML script block here, including event listeners, API calls, UI logic, etc.)