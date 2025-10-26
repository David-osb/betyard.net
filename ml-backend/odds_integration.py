"""
Odds Comparison Integration for Flask App
Adds endpoints for real-time odds comparison and value bet identification
"""

import asyncio
import os
from datetime import datetime
from flask import jsonify, request
from odds_comparison_service import OddsComparisonService

# Add these endpoints to your app.py file

class OddsIntegration:
    """Integration class for odds comparison functionality"""
    
    def __init__(self, app, odds_api_key: str):
        self.app = app
        self.odds_api_key = odds_api_key
        self.setup_routes()
    
    def setup_routes(self):
        """Setup Flask routes for odds comparison"""
        
        @self.app.route('/api/odds/compare/<sport>', methods=['GET'])
        def compare_odds(sport):
            """Get real-time odds comparison for a sport"""
            try:
                # Run async function in event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                async def get_odds():
                    async with OddsComparisonService(self.odds_api_key) as odds_service:
                        return await odds_service.get_market_analysis(sport)
                
                result = loop.run_until_complete(get_odds())
                loop.close()
                
                return jsonify({
                    'success': True,
                    'data': result,
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Error comparing odds: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/odds/value-bets/<sport>', methods=['POST'])
        def find_value_bets(sport):
            """Find value bets by comparing model predictions with market odds"""
            try:
                # Get model predictions from request
                predictions = request.get_json()
                
                if not predictions:
                    return jsonify({
                        'success': False,
                        'error': 'Model predictions required'
                    }), 400
                
                # Run async function in event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                async def get_value_bets():
                    async with OddsComparisonService(self.odds_api_key) as odds_service:
                        return await odds_service.identify_value_bets(sport, predictions)
                
                value_bets = loop.run_until_complete(get_value_bets())
                loop.close()
                
                # Convert dataclasses to dictionaries for JSON serialization
                value_bets_dict = []
                for bet in value_bets:
                    value_bets_dict.append({
                        'game_id': bet.game_id,
                        'home_team': bet.home_team,
                        'away_team': bet.away_team,
                        'commence_time': bet.commence_time.isoformat(),
                        'market': bet.market,
                        'outcome': bet.outcome,
                        'best_odds': bet.best_odds,
                        'best_bookmaker': bet.best_bookmaker,
                        'model_probability': bet.model_probability,
                        'implied_probability': bet.implied_probability,
                        'edge': bet.edge,
                        'kelly_fraction': bet.kelly_fraction
                    })
                
                return jsonify({
                    'success': True,
                    'data': {
                        'value_bets': value_bets_dict,
                        'count': len(value_bets_dict)
                    },
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Error finding value bets: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/odds/best-lines/<sport>/<team>', methods=['GET'])
        def get_best_lines(sport, team):
            """Get best available lines for a specific team"""
            try:
                market = request.args.get('market', 'h2h')
                
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                async def get_lines():
                    async with OddsComparisonService(self.odds_api_key) as odds_service:
                        raw_odds = await odds_service.fetch_odds(sport, [market])
                        odds_data = odds_service.parse_odds_data(raw_odds)
                        return odds_service.find_best_odds(odds_data, market, team)
                
                best_odds = loop.run_until_complete(get_lines())
                loop.close()
                
                if best_odds:
                    return jsonify({
                        'success': True,
                        'data': {
                            'team': team,
                            'market': market,
                            'best_odds': best_odds.price,
                            'bookmaker': best_odds.bookmaker,
                            'implied_probability': best_odds.implied_probability,
                            'last_update': best_odds.last_update.isoformat()
                        }
                    })
                else:
                    return jsonify({
                        'success': False,
                        'error': f'No odds found for {team} in {market} market'
                    }), 404
                    
            except Exception as e:
                logger.error(f"Error getting best lines: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/odds/arbitrage/<sport>', methods=['GET'])
        def find_arbitrage_opportunities(sport):
            """Find arbitrage betting opportunities"""
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                async def get_arbitrage():
                    async with OddsComparisonService(self.odds_api_key) as odds_service:
                        analysis = await odds_service.get_market_analysis(sport)
                        return analysis.get('arbitrage_opportunities', [])
                
                arbitrage_ops = loop.run_until_complete(get_arbitrage())
                loop.close()
                
                return jsonify({
                    'success': True,
                    'data': {
                        'arbitrage_opportunities': arbitrage_ops,
                        'count': len(arbitrage_ops)
                    },
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Error finding arbitrage opportunities: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500

# Example integration function for your existing model predictions
def integrate_with_existing_predictions(app, odds_api_key):
    """
    Integration function that connects your existing QB/RB/WR/TE predictions
    with the odds comparison service to find value bets
    """
    
    @app.route('/api/predictions/value-analysis/<position>', methods=['GET'])
    def get_value_analysis(position):
        """Get value betting analysis for QB/RB/WR/TE predictions"""
        try:
            # Get your existing model predictions
            week = request.args.get('week', type=int)
            season = request.args.get('season', type=int, default=2024)
            
            # This would call your existing prediction endpoints
            if position == 'qb':
                predictions_response = app.test_client().get(f'/api/predict/qb?week={week}&season={season}')
            elif position == 'rb':
                predictions_response = app.test_client().get(f'/api/predict/rb?week={week}&season={season}')
            elif position == 'wr':
                predictions_response = app.test_client().get(f'/api/predict/wr?week={week}&season={season}')
            elif position == 'te':
                predictions_response = app.test_client().get(f'/api/predict/te?week={week}&season={season}')
            else:
                return jsonify({'success': False, 'error': 'Invalid position'}), 400
            
            if predictions_response.status_code != 200:
                return jsonify({'success': False, 'error': 'Failed to get predictions'}), 500
            
            predictions_data = predictions_response.get_json()
            
            # Convert model predictions to format expected by odds service
            # This is a simplified example - you'd need to map your predictions
            # to game outcomes and probabilities
            formatted_predictions = {}
            
            for player in predictions_data.get('data', {}).get('predictions', []):
                # Extract team info and create game predictions
                # This is where you'd map player performance to team win probability
                pass
            
            # Get value bets using odds comparison
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            async def get_value_bets():
                async with OddsComparisonService(odds_api_key) as odds_service:
                    return await odds_service.identify_value_bets('nfl', formatted_predictions)
            
            value_bets = loop.run_until_complete(get_value_bets())
            loop.close()
            
            return jsonify({
                'success': True,
                'data': {
                    'position': position,
                    'week': week,
                    'season': season,
                    'model_predictions': predictions_data,
                    'value_bets': [bet.__dict__ for bet in value_bets],
                    'value_bet_count': len(value_bets)
                },
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error in value analysis: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500