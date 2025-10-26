"""
Real-Time Sportsbook Odds Comparison Service
Fetches live odds from multiple sportsbooks and identifies value bets
"""

import asyncio
import aiohttp
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BettingMarket(Enum):
    MONEYLINE = "h2h"
    SPREAD = "spreads"
    TOTALS = "totals"
    PLAYER_PROPS = "player_props"

@dataclass
class OddsData:
    bookmaker: str
    market: str
    outcome: str
    price: float  # American odds
    implied_probability: float
    last_update: datetime

@dataclass
class ValueBet:
    game_id: str
    home_team: str
    away_team: str
    commence_time: datetime
    market: str
    outcome: str
    best_odds: float
    best_bookmaker: str
    model_probability: float
    implied_probability: float
    edge: float  # Expected value percentage
    kelly_fraction: float  # Kelly Criterion bet sizing

class OddsComparisonService:
    """Service for fetching and comparing odds across multiple sportsbooks"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.the-odds-api.com/v4"
        self.session = None
        
        # Major US sportsbooks for comparison
        self.bookmakers = [
            'fanduel', 'draftkings', 'betmgm', 'caesars', 'bovada',
            'mybookieag', 'unibet_us', 'pointsbetus', 'superbook',
            'ballybet', 'wynnbet', 'betrivers'
        ]
        
        # Supported sports
        self.sports = {
            'nfl': 'americanfootball_nfl',
            'ncaaf': 'americanfootball_ncaaf',
            'nba': 'basketball_nba',
            'ncaab': 'basketball_ncaab',
            'mlb': 'baseball_mlb',
            'nhl': 'icehockey_nhl'
        }
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    def american_to_decimal(self, american_odds: float) -> float:
        """Convert American odds to decimal odds"""
        if american_odds > 0:
            return (american_odds / 100) + 1
        else:
            return (100 / abs(american_odds)) + 1
    
    def american_to_implied_probability(self, american_odds: float) -> float:
        """Convert American odds to implied probability"""
        if american_odds > 0:
            return 100 / (american_odds + 100)
        else:
            return abs(american_odds) / (abs(american_odds) + 100)
    
    def calculate_kelly_fraction(self, edge: float, odds: float) -> float:
        """Calculate optimal bet size using Kelly Criterion"""
        if edge <= 0:
            return 0
        
        decimal_odds = self.american_to_decimal(odds)
        kelly = edge / (decimal_odds - 1)
        
        # Cap at 25% for risk management
        return min(kelly, 0.25)
    
    async def fetch_odds(self, sport: str, markets: List[str] = None) -> Dict:
        """Fetch current odds for a sport from The Odds API"""
        if markets is None:
            markets = ['h2h', 'spreads', 'totals']
        
        sport_key = self.sports.get(sport.lower(), sport)
        
        params = {
            'apiKey': self.api_key,
            'regions': 'us',  # Focus on US sportsbooks
            'markets': ','.join(markets),
            'oddsFormat': 'american',
            'bookmakers': ','.join(self.bookmakers)
        }
        
        url = f"{self.base_url}/sports/{sport_key}/odds"
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"Fetched odds for {len(data)} {sport} games")
                    return data
                else:
                    logger.error(f"API request failed: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Error fetching odds: {e}")
            return []
    
    def parse_odds_data(self, raw_data: List[Dict]) -> List[OddsData]:
        """Parse raw API response into structured odds data"""
        odds_list = []
        
        for game in raw_data:
            game_id = game['id']
            home_team = game['home_team']
            away_team = game['away_team']
            commence_time = datetime.fromisoformat(game['commence_time'].replace('Z', '+00:00'))
            
            for bookmaker in game.get('bookmakers', []):
                bookmaker_name = bookmaker['key']
                last_update = datetime.fromisoformat(bookmaker['last_update'].replace('Z', '+00:00'))
                
                for market in bookmaker.get('markets', []):
                    market_key = market['key']
                    
                    for outcome in market.get('outcomes', []):
                        odds_data = OddsData(
                            bookmaker=bookmaker_name,
                            market=market_key,
                            outcome=outcome['name'],
                            price=outcome['price'],
                            implied_probability=self.american_to_implied_probability(outcome['price']),
                            last_update=last_update
                        )
                        odds_list.append(odds_data)
        
        return odds_list
    
    def find_best_odds(self, odds_data: List[OddsData], market: str, outcome: str) -> Optional[OddsData]:
        """Find the best odds for a specific market and outcome"""
        filtered_odds = [
            odds for odds in odds_data 
            if odds.market == market and odds.outcome == outcome
        ]
        
        if not filtered_odds:
            return None
        
        # For positive odds, higher is better; for negative odds, closer to 0 is better
        best_odds = max(filtered_odds, key=lambda x: x.price if x.price > 0 else -abs(x.price))
        return best_odds
    
    async def identify_value_bets(self, sport: str, model_predictions: Dict[str, Dict]) -> List[ValueBet]:
        """
        Identify value bets by comparing model predictions with market odds
        
        model_predictions format:
        {
            "game_id": {
                "home_team": "Team A",
                "away_team": "Team B",
                "commence_time": "2025-10-25T20:00:00Z",
                "predictions": {
                    "home_win_probability": 0.65,
                    "away_win_probability": 0.35,
                    "over_probability": 0.52,
                    "under_probability": 0.48
                }
            }
        }
        """
        value_bets = []
        
        # Fetch current odds
        raw_odds = await self.fetch_odds(sport)
        if not raw_odds:
            return value_bets
        
        odds_data = self.parse_odds_data(raw_odds)
        
        for game in raw_odds:
            game_id = game['id']
            home_team = game['home_team']
            away_team = game['away_team']
            commence_time = datetime.fromisoformat(game['commence_time'].replace('Z', '+00:00'))
            
            # Skip if no model prediction for this game
            if game_id not in model_predictions:
                continue
            
            predictions = model_predictions[game_id]['predictions']
            
            # Check moneyline value bets
            home_odds = self.find_best_odds(odds_data, 'h2h', home_team)
            away_odds = self.find_best_odds(odds_data, 'h2h', away_team)
            
            if home_odds:
                model_prob = predictions.get('home_win_probability', 0)
                edge = model_prob - home_odds.implied_probability
                
                if edge > 0.05:  # 5% minimum edge
                    kelly = self.calculate_kelly_fraction(edge, home_odds.price)
                    value_bet = ValueBet(
                        game_id=game_id,
                        home_team=home_team,
                        away_team=away_team,
                        commence_time=commence_time,
                        market='moneyline',
                        outcome=home_team,
                        best_odds=home_odds.price,
                        best_bookmaker=home_odds.bookmaker,
                        model_probability=model_prob,
                        implied_probability=home_odds.implied_probability,
                        edge=edge,
                        kelly_fraction=kelly
                    )
                    value_bets.append(value_bet)
            
            if away_odds:
                model_prob = predictions.get('away_win_probability', 0)
                edge = model_prob - away_odds.implied_probability
                
                if edge > 0.05:  # 5% minimum edge
                    kelly = self.calculate_kelly_fraction(edge, away_odds.price)
                    value_bet = ValueBet(
                        game_id=game_id,
                        home_team=home_team,
                        away_team=away_team,
                        commence_time=commence_time,
                        market='moneyline',
                        outcome=away_team,
                        best_odds=away_odds.price,
                        best_bookmaker=away_odds.bookmaker,
                        model_probability=model_prob,
                        implied_probability=away_odds.implied_probability,
                        edge=edge,
                        kelly_fraction=kelly
                    )
                    value_bets.append(value_bet)
        
        # Sort by edge (highest value first)
        value_bets.sort(key=lambda x: x.edge, reverse=True)
        
        logger.info(f"Found {len(value_bets)} value bets")
        return value_bets
    
    async def get_market_analysis(self, sport: str) -> Dict:
        """Get comprehensive market analysis including arbitrage opportunities"""
        raw_odds = await self.fetch_odds(sport)
        if not raw_odds:
            return {}
        
        odds_data = self.parse_odds_data(raw_odds)
        analysis = {
            'total_games': len(raw_odds),
            'total_bookmakers': len(set(odds.bookmaker for odds in odds_data)),
            'arbitrage_opportunities': [],
            'best_odds_by_game': {},
            'market_efficiency': {}
        }
        
        # Calculate market efficiency and find arbitrage opportunities
        for game in raw_odds:
            game_id = game['id']
            home_team = game['home_team']
            away_team = game['away_team']
            
            # Get all moneyline odds for this game
            home_odds_list = [odds for odds in odds_data if odds.market == 'h2h' and odds.outcome == home_team]
            away_odds_list = [odds for odds in odds_data if odds.market == 'h2h' and odds.outcome == away_team]
            
            if home_odds_list and away_odds_list:
                best_home = max(home_odds_list, key=lambda x: x.price if x.price > 0 else -abs(x.price))
                best_away = max(away_odds_list, key=lambda x: x.price if x.price > 0 else -abs(x.price))
                
                # Check for arbitrage opportunity
                total_implied_prob = best_home.implied_probability + best_away.implied_probability
                if total_implied_prob < 1.0:
                    profit_margin = (1.0 - total_implied_prob) * 100
                    analysis['arbitrage_opportunities'].append({
                        'game': f"{away_team} @ {home_team}",
                        'home_odds': best_home.price,
                        'home_bookmaker': best_home.bookmaker,
                        'away_odds': best_away.price,
                        'away_bookmaker': best_away.bookmaker,
                        'profit_margin': profit_margin
                    })
                
                analysis['best_odds_by_game'][game_id] = {
                    'home_team': home_team,
                    'away_team': away_team,
                    'best_home_odds': best_home.price,
                    'best_home_bookmaker': best_home.bookmaker,
                    'best_away_odds': best_away.price,
                    'best_away_bookmaker': best_away.bookmaker,
                    'market_efficiency': total_implied_prob
                }
        
        return analysis

# Example usage function
async def main():
    """Example usage of the odds comparison service"""
    API_KEY = "YOUR_ODDS_API_KEY"  # Replace with actual API key
    
    async with OddsComparisonService(API_KEY) as odds_service:
        # Get market analysis
        analysis = await odds_service.get_market_analysis('nfl')
        print(f"Market Analysis: {json.dumps(analysis, indent=2, default=str)}")
        
        # Example model predictions (this would come from your ML model)
        mock_predictions = {
            "game_id_1": {
                "home_team": "Buffalo Bills",
                "away_team": "Miami Dolphins", 
                "commence_time": "2025-10-25T20:00:00Z",
                "predictions": {
                    "home_win_probability": 0.68,
                    "away_win_probability": 0.32,
                    "over_probability": 0.55,
                    "under_probability": 0.45
                }
            }
        }
        
        # Find value bets
        value_bets = await odds_service.identify_value_bets('nfl', mock_predictions)
        
        print(f"\nFound {len(value_bets)} value bets:")
        for bet in value_bets[:5]:  # Show top 5
            print(f"Game: {bet.away_team} @ {bet.home_team}")
            print(f"Bet: {bet.outcome} at {bet.best_odds} ({bet.best_bookmaker})")
            print(f"Edge: {bet.edge:.2%}, Kelly: {bet.kelly_fraction:.2%}")
            print("---")

if __name__ == "__main__":
    asyncio.run(main())