"""
Multi-Book Odds Scraper
Combines DraftKings and FanDuel odds for best lines
"""

from adaptive_odds_scraper import AdaptiveOddsScraper
from fanduel_scraper import FanDuelScraper
from typing import Dict, List
import time

class MultiBookScraper:
    """
    Fetch odds from multiple sportsbooks and find best lines
    """
    
    def __init__(self):
        self.draftkings = AdaptiveOddsScraper(use_smart_requester=False)
        self.fanduel = FanDuelScraper()
    
    def get_best_odds(self, player_name: str, stat_type: str) -> Dict:
        """
        Get odds from both books and return best available
        
        Returns:
            {
                'player': 'Josh Allen',
                'stat': 'passing_yards',
                'consensus_line': 245.0,
                'books': {
                    'draftkings': {
                        'line': 245.5,
                        'over': -110,
                        'under': -110
                    },
                    'fanduel': {
                        'line': 244.5,
                        'over': -115,
                        'under': -105
                    }
                },
                'best_over': {
                    'book': 'draftkings',
                    'line': 245.5,
                    'odds': -110
                },
                'best_under': {
                    'book': 'fanduel',
                    'line': 244.5,
                    'odds': -105
                },
                'edge': '+5 odds shopping advantage'
            }
        """
        books = {}
        
        # Get DraftKings odds
        print(f"📊 Fetching {player_name} {stat_type} from DraftKings...")
        dk_props = self.draftkings.get_player_props(player_name, stat_type)
        
        if 'books' in dk_props and 'draftkings' in dk_props['books']:
            books['draftkings'] = dk_props['books']['draftkings']
        
        # Wait to avoid rate limiting
        time.sleep(1)
        
        # Get FanDuel odds
        print(f"📊 Fetching {player_name} {stat_type} from FanDuel...")
        fd_props = self.fanduel.get_player_props(player_name, stat_type)
        
        if 'line' in fd_props and fd_props['line'] is not None:
            books['fanduel'] = {
                'line': fd_props['line'],
                'over': fd_props['over_odds'],
                'under': fd_props['under_odds'],
                'market_id': fd_props['market_id']
            }
        
        # No odds found
        if not books:
            return {
                'error': 'No odds available from any book',
                'player': player_name,
                'stat': stat_type
            }
        
        # Calculate consensus line
        lines = [book['line'] for book in books.values()]
        consensus_line = round(sum(lines) / len(lines), 1)
        
        # Find best odds
        best_over = self._find_best_over(books)
        best_under = self._find_best_under(books)
        
        # Calculate edge
        edge = self._calculate_edge(books, best_over, best_under)
        
        return {
            'player': player_name,
            'stat': stat_type,
            'consensus_line': consensus_line,
            'books': books,
            'best_over': best_over,
            'best_under': best_under,
            'edge': edge,
            'timestamp': dk_props.get('timestamp') or fd_props.get('timestamp')
        }
    
    def _find_best_over(self, books: Dict) -> Dict:
        """Find best over odds across all books"""
        best = None
        best_book = None
        
        for book_name, book_data in books.items():
            odds = book_data.get('over')
            if odds is not None:
                if best is None or odds > best:
                    best = odds
                    best_book = book_name
        
        if best_book:
            return {
                'book': best_book,
                'line': books[best_book]['line'],
                'odds': best
            }
        return None
    
    def _find_best_under(self, books: Dict) -> Dict:
        """Find best under odds across all books"""
        best = None
        best_book = None
        
        for book_name, book_data in books.items():
            odds = book_data.get('under')
            if odds is not None:
                if best is None or odds > best:
                    best = odds
                    best_book = book_name
        
        if best_book:
            return {
                'book': best_book,
                'line': books[best_book]['line'],
                'odds': best
            }
        return None
    
    def _calculate_edge(self, books: Dict, best_over: Dict, best_under: Dict) -> str:
        """
        Calculate the advantage of line shopping
        """
        if not best_over or not best_under or len(books) < 2:
            return "Single book - no edge"
        
        # Compare best vs worst over odds
        over_odds = [book['over'] for book in books.values() if book.get('over')]
        under_odds = [book['under'] for book in books.values() if book.get('under')]
        
        over_range = max(over_odds) - min(over_odds) if over_odds else 0
        under_range = max(under_odds) - min(under_odds) if under_odds else 0
        
        max_edge = max(over_range, under_range)
        
        if max_edge >= 10:
            return f"+{int(max_edge)} odds advantage (line shopping pays!)"
        elif max_edge >= 5:
            return f"+{int(max_edge)} odds edge"
        else:
            return "Minimal edge - books agree"
    
    def get_multiple_players(self, player_stats: List[tuple]) -> Dict:
        """
        Get odds for multiple player/stat combinations
        
        Args:
            player_stats: [('Josh Allen', 'passing_yards'), ('Patrick Mahomes', 'passing_tds')]
        
        Returns:
            Dictionary with results for each player/stat
        """
        results = {}
        
        for player, stat in player_stats:
            key = f"{player}_{stat}"
            print(f"\n{'='*60}")
            print(f"Processing: {player} - {stat}")
            print('='*60)
            
            results[key] = self.get_best_odds(player, stat)
            
            # Rate limiting
            time.sleep(2)
        
        return results


# Example usage
if __name__ == '__main__':
    scraper = MultiBookScraper()
    
    # Single player
    print("Fetching multi-book odds for Josh Allen passing yards...\n")
    odds = scraper.get_best_odds('Josh Allen', 'passing_yards')
    
    import json
    print(json.dumps(odds, indent=2))
    
    # Show recommendations
    if 'best_over' in odds and odds['best_over']:
        print(f"\n🎯 BEST OVER: {odds['best_over']['book']} @ {odds['best_over']['odds']} (line {odds['best_over']['line']})")
    
    if 'best_under' in odds and odds['best_under']:
        print(f"🎯 BEST UNDER: {odds['best_under']['book']} @ {odds['best_under']['odds']} (line {odds['best_under']['line']})")
    
    print(f"\n💰 {odds.get('edge', 'No edge data')}")
