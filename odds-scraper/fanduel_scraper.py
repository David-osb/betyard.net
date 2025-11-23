"""
FanDuel Odds Scraper
Uses FanDuel's public API endpoints to get real-time odds
"""

import requests
import json
import time
from datetime import datetime
from typing import Dict, List, Optional

class FanDuelScraper:
    """
    Scrapes live odds from FanDuel's public API
    No authentication required - uses same endpoints as their website
    """
    
    def __init__(self):
        self.base_url = "https://sportsbook.fanduel.com"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
            'Referer': 'https://sportsbook.fanduel.com/'
        })
        
    def get_nfl_events(self) -> List[Dict]:
        """
        Get all NFL events (games)
        Returns list of games with IDs and metadata
        """
        url = f"{self.base_url}/api/content-managed-page"
        params = {
            'page': 'SPORT',
            'eventTypeId': '6423',  # NFL
            '_ak': 'FhMFpcPWXMeyZxOx',
            'timezone': 'America/New_York'
        }
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            events = []
            # Parse events from response
            if 'attachments' in data and 'events' in data['attachments']:
                for event_id, event in data['attachments']['events'].items():
                    events.append({
                        'id': event_id,
                        'name': event.get('name', 'Unknown'),
                        'start_time': event.get('openDate'),
                        'home_team': event.get('homeTeamName'),
                        'away_team': event.get('awayTeamName'),
                        'in_play': event.get('inPlay', False)
                    })
            
            return events
            
        except Exception as e:
            print(f"Error fetching NFL events: {e}")
            return []
    
    def get_live_data(self, event_ids: List[str]) -> Dict:
        """
        Get live data for specific events
        Returns real-time odds and game updates
        """
        url = f"{self.base_url}/api/livedata"
        params = {
            'channel': 'WEB',
            'dataEntries': 'FULL_DETAILS,MARKET_DEFS,OB_EVENT_SCORES',
            'eventIds': ','.join(event_ids)
        }
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            print(f"Error fetching live data: {e}")
            return {}
    
    def get_market_prices(self, market_ids: List[str]) -> Dict:
        """
        Get current prices (odds) for specific markets
        This is the key endpoint for real-time odds
        """
        url = f"{self.base_url}/api/getMarketPrices"
        params = {
            'priceHistory': '0'
        }
        
        # POST request with market IDs in body
        payload = {
            'marketIds': market_ids
        }
        
        try:
            response = self.session.post(url, params=params, json=payload, timeout=10)
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            print(f"Error fetching market prices: {e}")
            return {}
    
    def get_player_props(self, player_name: str, stat_type: str = 'passing_yards') -> Dict:
        """
        Get player prop odds for a specific player/stat
        
        Args:
            player_name: "Josh Allen"
            stat_type: 'passing_yards', 'rushing_yards', etc.
        
        Returns:
            {
                'player': 'Josh Allen',
                'stat': 'passing_yards',
                'line': 245.5,
                'over_odds': -110,
                'under_odds': -110,
                'market_id': '12345',
                'timestamp': '2025-11-23T...'
            }
        """
        # First, get NFL events to find games
        events = self.get_nfl_events()
        
        if not events:
            return {
                'error': 'No NFL events found',
                'player': player_name,
                'stat': stat_type
            }
        
        # Get event IDs
        event_ids = [e['id'] for e in events[:5]]  # Limit to first 5 games
        
        # Get live data for these events
        live_data = self.get_live_data(event_ids)
        
        # Search for player markets
        player_markets = self._find_player_markets(live_data, player_name, stat_type)
        
        if not player_markets:
            return {
                'error': f'No markets found for {player_name} {stat_type}',
                'player': player_name,
                'stat': stat_type
            }
        
        # Get prices for the market
        market_id = player_markets[0]['id']
        prices = self.get_market_prices([market_id])
        
        # Parse odds
        odds_data = self._parse_market_odds(prices, market_id, player_name, stat_type)
        
        return odds_data
    
    def _find_player_markets(self, live_data: Dict, player_name: str, stat_type: str) -> List[Dict]:
        """
        Search live data for markets matching player and stat type
        """
        markets = []
        
        # Map stat types to FanDuel market names
        stat_map = {
            'passing_yards': ['passing yards', 'pass yds'],
            'passing_tds': ['passing touchdowns', 'pass td'],
            'rushing_yards': ['rushing yards', 'rush yds'],
            'rushing_tds': ['rushing touchdowns', 'rush td'],
            'receiving_yards': ['receiving yards', 'rec yds'],
            'receptions': ['receptions', 'rec'],
        }
        
        keywords = stat_map.get(stat_type, [stat_type.replace('_', ' ')])
        
        # Search through markets
        if 'attachments' in live_data and 'markets' in live_data['attachments']:
            for market_id, market in live_data['attachments']['markets'].items():
                market_name = market.get('marketName', '').lower()
                
                # Check if player name and stat type match
                if player_name.lower() in market_name:
                    if any(kw in market_name for kw in keywords):
                        markets.append({
                            'id': market_id,
                            'name': market.get('marketName'),
                            'type': market.get('marketType')
                        })
        
        return markets
    
    def _parse_market_odds(self, prices_data: Dict, market_id: str, player_name: str, stat_type: str) -> Dict:
        """
        Parse odds from getMarketPrices response
        """
        if 'markets' not in prices_data or market_id not in prices_data['markets']:
            return {
                'error': 'Market prices not found',
                'player': player_name,
                'stat': stat_type
            }
        
        market = prices_data['markets'][market_id]
        
        # Find Over/Under selections
        over_odds = None
        under_odds = None
        line = None
        
        if 'selections' in market:
            for selection in market['selections']:
                name = selection.get('name', '').lower()
                
                if 'over' in name:
                    over_odds = self._american_odds(selection.get('currentPriceUp'), selection.get('currentPriceDown'))
                    if 'handicap' in selection:
                        line = float(selection['handicap'])
                
                if 'under' in name:
                    under_odds = self._american_odds(selection.get('currentPriceUp'), selection.get('currentPriceDown'))
                    if 'handicap' in selection and line is None:
                        line = float(selection['handicap'])
        
        return {
            'player': player_name,
            'stat': stat_type,
            'line': line,
            'over_odds': over_odds,
            'under_odds': under_odds,
            'market_id': market_id,
            'timestamp': datetime.now().isoformat(),
            'source': 'fanduel'
        }
    
    def _american_odds(self, price_up: Optional[float], price_down: Optional[float]) -> Optional[int]:
        """
        Convert FanDuel fractional odds to American odds
        price_up/price_down represents fractional odds (e.g., 10/11)
        """
        if price_up is None or price_down is None:
            return None
        
        # Convert fractional to decimal
        decimal = (price_up / price_down) + 1
        
        # Convert decimal to American
        if decimal >= 2.0:
            return int((decimal - 1) * 100)
        else:
            return int(-100 / (decimal - 1))
    
    def get_all_player_props(self, players: List[str]) -> Dict:
        """
        Get props for multiple players at once
        More efficient than calling get_player_props multiple times
        """
        results = {}
        
        for player in players:
            for stat in ['passing_yards', 'rushing_yards', 'passing_tds']:
                key = f"{player}_{stat}"
                results[key] = self.get_player_props(player, stat)
                time.sleep(0.5)  # Rate limiting
        
        return results


# Example usage
if __name__ == '__main__':
    scraper = FanDuelScraper()
    
    # Get NFL events
    print("Fetching NFL events...")
    events = scraper.get_nfl_events()
    print(f"Found {len(events)} NFL events\n")
    
    # Get player props
    print("Fetching Josh Allen passing yards...")
    props = scraper.get_player_props('Josh Allen', 'passing_yards')
    print(json.dumps(props, indent=2))
