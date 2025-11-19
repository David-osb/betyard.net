"""
Fetch real NFL player statistics from ESPN API - 2025 Season (Weeks 1-11)
Fetches all QB/RB/WR/TE players from all 32 teams
For QBs: Only counts RUSHING TDs (anytime TD scorer bets don't include passing TDs)
"""

import requests
import json
import time

# ESPN API endpoints
ESPN_BASE = "https://site.web.api.espn.com/apis"
PLAYER_GAMELOG_URL = f"{ESPN_BASE}/common/v3/sports/football/nfl/athletes/{{player_id}}/gamelog"
TEAM_ROSTER_URL = f"{ESPN_BASE}/site/v2/sports/football/nfl/teams/{{team_id}}/roster"

# NFL Team IDs for ESPN API
NFL_TEAMS = {
    'ARI': 22, 'ATL': 1, 'BAL': 33, 'BUF': 2, 'CAR': 29, 'CHI': 3,
    'CIN': 4, 'CLE': 5, 'DAL': 6, 'DEN': 7, 'DET': 8, 'GB': 9,
    'HOU': 34, 'IND': 11, 'JAX': 30, 'KC': 12, 'LV': 13, 'LAC': 24,
    'LAR': 14, 'MIA': 15, 'MIN': 16, 'NE': 17, 'NO': 18, 'NYG': 19,
    'NYJ': 20, 'PHI': 21, 'PIT': 23, 'SF': 25, 'SEA': 26, 'TB': 27,
    'TEN': 10, 'WAS': 28
}

def fetch_with_retry(url, retries=2, delay=1):
    """Fetch URL with retry logic"""
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=15)
            if response.status_code == 200:
                return response.json()
        except:
            pass
        
        if attempt < retries - 1:
            time.sleep(delay)
    
    return None

def get_team_roster(team_code):
    """Fetch all offensive players from team roster with positions"""
    team_id = NFL_TEAMS.get(team_code)
    if not team_id:
        return []
    
    url = TEAM_ROSTER_URL.format(team_id=team_id)
    data = fetch_with_retry(url)
    
    if not data or 'athletes' not in data:
        return []
    
    players = []
    for group in data.get('athletes', []):
        if group.get('position') != 'offense':
            continue
            
        for athlete in group.get('items', []):
            try:
                position_data = athlete.get('position', {})
                position_abbr = position_data.get('abbreviation', '')
                
                # Only include QB, RB, WR, TE
                if position_abbr in ['QB', 'RB', 'WR', 'TE']:
                    players.append({
                        'id': athlete.get('id'),
                        'name': athlete.get('displayName', 'Unknown'),
                        'team': team_code,
                        'position': position_abbr  # Add position from roster
                    })
            except:
                pass
    
    return players

def get_player_gamelog_with_position(player_id):
    """
    Fetch player game log and determine position from stats
    
    Returns: (position, gamelog_data)
    position: 'QB', 'RB', 'WR', 'TE', or None
    gamelog_data: list of games with TD stats
    
    For QBs: ONLY counts rushing TDs (anytime TD scorer excludes passing TDs)
    For others: Counts rushing + receiving TDs
    """
    url = PLAYER_GAMELOG_URL.format(player_id=player_id)
    data = fetch_with_retry(url, retries=2, delay=0.5)
    
    if not data:
        return None, []
    
    try:
        # Get stat names to find TD indices
        stat_names = data.get('names', [])
        if not stat_names:
            return None, []
        
        # Determine position from stat categories
        categories = data.get('categories', [])
        position = None
        if any(cat.get('name') == 'passing' for cat in categories):
            position = 'QB'
        elif any(cat.get('name') == 'rushing' for cat in categories):
            # Could be RB or QB - check for receiving stats
            if any(cat.get('name') == 'receiving' for cat in categories):
                position = 'RB'  # RBs have both rushing and receiving
            else:
                position = 'RB'  # Pure rusher
        elif any(cat.get('name') == 'receiving' for cat in categories):
            # WR or TE - we'll assume WR for now (can refine later)
            position = 'WR'
        
        if not position or position not in ['QB', 'RB', 'WR', 'TE']:
            return None, []
        
        # Find TD stat indices
        rushing_td_idx = -1
        receiving_td_idx = -1
        passing_td_idx = -1
        yards_idx = -1
        
        for i, name in enumerate(stat_names):
            if name == 'rushingTouchdowns':
                rushing_td_idx = i
            elif name == 'receivingTouchdowns':
                receiving_td_idx = i
            elif name == 'passingTouchdowns':
                passing_td_idx = i
            elif 'Yards' in name and yards_idx == -1:
                yards_idx = i
        
        # Get games from seasonTypes (2025 season)
        games = []
        season_types = data.get('seasonTypes', [])
        for season_type in season_types:
            # Only use 2025 Regular Season
            if '2025' not in season_type.get('displayName', ''):
                continue
                
            categories = season_type.get('categories', [])
            for category in categories:
                events = category.get('events', [])
                for event in events:
                    stats = event.get('stats', [])
                    if not stats:
                        continue
                    
                    # Calculate TDs based on position
                    total_tds = 0
                    
                    if position == 'QB':
                        # QBs: ONLY rushing TDs count for anytime TD scorer
                        if rushing_td_idx >= 0 and rushing_td_idx < len(stats):
                            try:
                                rush_tds = stats[rushing_td_idx]
                                if rush_tds not in ['-', '', None]:
                                    total_tds = float(rush_tds)
                            except:
                                pass
                    else:
                        # RB/WR/TE: Rushing + Receiving TDs
                        if rushing_td_idx >= 0 and rushing_td_idx < len(stats):
                            try:
                                rush_tds = stats[rushing_td_idx]
                                if rush_tds not in ['-', '', None]:
                                    total_tds += float(rush_tds)
                            except:
                                pass
                        
                        if receiving_td_idx >= 0 and receiving_td_idx < len(stats):
                            try:
                                rec_tds = stats[receiving_td_idx]
                                if rec_tds not in ['-', '', None]:
                                    total_tds += float(rec_tds)
                            except:
                                pass
                    
                    # Get yards
                    yards = 0
                    if yards_idx >= 0 and yards_idx < len(stats):
                        try:
                            y = stats[yards_idx]
                            if y not in ['-', '', None]:
                                yards = float(y)
                        except:
                            pass
                    
                    games.append({
                        'game_id': event.get('eventId'),
                        'tds': total_tds,
                        'yards': yards
                    })
        
        return position, games
        
    except Exception as e:
        return None, []

def build_training_dataset():
    """Build training dataset from ALL NFL players (QB/RB/WR/TE) - 2025 season"""
    print("\n" + "="*80)
    print("🏈 FETCHING 2025 NFL DATA FROM ESPN API (ALL QB/RB/WR/TE PLAYERS)")
    print("="*80 + "\n")
    
    all_player_data = []
    position_counts = {'QB': 0, 'RB': 0, 'WR': 0, 'TE': 0}
    skipped = 0
    
    # Fetch rosters for all 32 teams
    for team_code in sorted(NFL_TEAMS.keys()):
        print(f"\n📋 {team_code}")
        roster = get_team_roster(team_code)
        print(f"  Found {len(roster)} skill position players")
        
        for player in roster:
            player_id = player['id']
            player_name = player['name']
            position = player['position']  # Get position from roster (already filtered to QB/RB/WR/TE)
            
            # Fetch gamelog
            _, gamelog = get_player_gamelog_with_position(player_id)
            
            if not gamelog or len(gamelog) == 0:
                skipped += 1
                continue
            
            # Calculate TD stats
            total_games = len(gamelog)
            games_with_td = sum(1 for g in gamelog if g['tds'] > 0)
            games_with_2plus_tds = sum(1 for g in gamelog if g['tds'] >= 2)
            total_tds = sum(g['tds'] for g in gamelog)
            td_probability = games_with_td / total_games if total_games > 0 else 0
            multi_td_probability = games_with_2plus_tds / total_games if total_games > 0 else 0
            avg_tds_per_game = total_tds / total_games if total_games > 0 else 0
            
            player_data = {
                'id': player_id,
                'name': player_name,
                'team': team_code,
                'position': position,
                'gamelog': gamelog,
                'td_stats': {
                    'games_played': total_games,
                    'games_with_td': games_with_td,
                    'games_with_2plus_tds': games_with_2plus_tds,
                    'total_tds': total_tds,
                    'td_probability': round(td_probability, 3),
                    'multi_td_probability': round(multi_td_probability, 3),
                    'avg_tds_per_game': round(avg_tds_per_game, 2)
                }
            }
            
            all_player_data.append(player_data)
            position_counts[position] += 1
            
            # Show progress for players with TDs
            if total_tds > 0:
                multi_info = f" ({games_with_2plus_tds} multi-TD)" if games_with_2plus_tds > 0 else ""
                print(f"  ✅ {player_name} ({position}): {total_tds} TDs in {total_games} games{multi_info}")
            
            # Rate limiting
            time.sleep(0.3)
    
    # Save to JSON
    output_file = 'espn_player_data.json'
    with open(output_file, 'w') as f:
        json.dump(all_player_data, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"✅ Saved {len(all_player_data)} players to {output_file}")
    print(f"⏭️  Skipped {skipped} players (no stats or non-skill positions)")
    print(f"{'='*80}")
    
    print("\n📊 Summary by Position:")
    for pos in ['QB', 'RB', 'WR', 'TE']:
        print(f"  {pos}: {position_counts[pos]} players")
    
    # Calculate average TD probabilities by position
    print("\n📈 Average TD Probabilities (2025 Season, Weeks 1-11):")
    for pos in ['QB', 'RB', 'WR', 'TE']:
        players_with_pos = [p for p in all_player_data if p['position'] == pos]
        if players_with_pos:
            avg_prob = sum(p['td_stats']['td_probability'] for p in players_with_pos) / len(players_with_pos)
            print(f"  {pos}: {avg_prob:.1%} (n={len(players_with_pos)})")
    
    if position_counts['QB'] > 0:
        print("\n⚠️  NOTE: QB probabilities are RUSHING TDs only (anytime TD scorer bets)")
    
    return all_player_data

if __name__ == '__main__':
    dataset = build_training_dataset()
    print(f"\n🎯 Dataset ready: {len(dataset)} players with 2025 season stats")
