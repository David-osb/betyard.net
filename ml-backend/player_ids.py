"""
DraftKings Player ID Database
Manually collected from DraftKings website
"""

# NFL Player IDs
NFL_PLAYER_IDS = {
    # QBs
    'josh allen': '11370',
    'patrick mahomes': '10838',
    'lamar jackson': '11145',
    'joe burrow': '12483',
    'jalen hurts': '12500',
    'justin herbert': '12492',
    'dak prescott': '10272',
    'jared goff': '10196',
    'cj stroud': '13897',
    'brock purdy': '13370',
    'tua tagovailoa': '12498',
    'jordan love': '12487',
    'matthew stafford': '9604',
    'geno smith': '9689',
    'kirk cousins': '9914',
    
    # RBs
    'christian mccaffrey': '11042',
    'derrick henry': '10565',
    'saquon barkley': '11141',
    'james cook': '13456',
    'josh jacobs': '12070',
    'kenneth walker': '13530',
    'breece hall': '13538',
    'tony pollard': '11797',
    'alvin kamara': '10638',
    'jonathan taylor': '12464',
    
    # WRs
    'tyreek hill': '10234',
    'stefon diggs': '10859',
    'davante adams': '10305',
    'justin jefferson': '12475',
    'cooper kupp': '10889',
    'amon-ra st brown': '13300',
    'ceedee lamb': '12485',
    'aj brown': '12063',
    'deebo samuel': '12069',
    'garrett wilson': '13532',
    
    # TEs
    'travis kelce': '10227',
    'mark andrews': '11599',
    'tj hockenson': '12074',
    'george kittle': '10891',
    'dallas goedert': '11184',
}

# NBA Player IDs
NBA_PLAYER_IDS = {
    'lebron james': '1966',
    'stephen curry': '3975',
    'kevin durant': '3704',
    'giannis antetokounmpo': '6583',
    'luka doncic': '9020',
    'nikola jokic': '6583',
    'joel embiid': '7326',
    'jayson tatum': '8692',
    'anthony davis': '6156',
    'damian lillard': '5431',
    'kawhi leonard': '5050',
    'paul george': '5230',
    'jimmy butler': '5230',
    'devin booker': '7673',
    'donovan mitchell': '8253',
    'anthony edwards': '10535',
    'ja morant': '10264',
    'trae young': '9294',
}

def get_player_id(player_name, sport='nfl'):
    """
    Get DraftKings player ID
    
    Args:
        player_name: "Josh Allen" or "josh allen"
        sport: 'nfl' or 'nba'
    
    Returns:
        player_id (string) or None
    """
    player_name = player_name.lower().strip()
    
    if sport.lower() == 'nfl':
        return NFL_PLAYER_IDS.get(player_name)
    elif sport.lower() == 'nba':
        return NBA_PLAYER_IDS.get(player_name)
    
    return None


def get_all_players(sport='nfl'):
    """Get all player IDs for a sport"""
    if sport.lower() == 'nfl':
        return NFL_PLAYER_IDS
    elif sport.lower() == 'nba':
        return NBA_PLAYER_IDS
    return {}


if __name__ == '__main__':
    print("\n📊 DraftKings Player Database")
    print(f"\nNFL Players: {len(NFL_PLAYER_IDS)}")
    print(f"NBA Players: {len(NBA_PLAYER_IDS)}")
    
    # Test lookups
    test_cases = [
        ('Josh Allen', 'nfl'),
        ('patrick mahomes', 'nfl'),
        ('LeBron James', 'nba'),
    ]
    
    print("\n🧪 Test Lookups:")
    for player, sport in test_cases:
        player_id = get_player_id(player, sport)
        print(f"  {player} ({sport.upper()}): {player_id}")
