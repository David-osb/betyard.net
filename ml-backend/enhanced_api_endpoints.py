"""
Enhanced API endpoints using comprehensive ESPN data
Integrates with existing Flask app to provide enhanced betting insights
"""

from flask import jsonify, request
import logging
from typing import Dict, List, Any
from enhanced_espn_service import EnhancedESPNService
from enhanced_ml_predictor import EnhancedMLPredictor

logger = logging.getLogger(__name__)

def init_enhanced_endpoints(app, models):
    """Initialize enhanced endpoints with the Flask app"""
    
    # Initialize services
    espn_service = EnhancedESPNService()
    ml_predictor = EnhancedMLPredictor(models, espn_service)
    
    @app.route('/api/enhanced/player/<player_id>/analysis')
    def get_enhanced_player_analysis(player_id):
        """
        Get comprehensive player analysis using Tier 1 ESPN endpoints
        """
        try:
            analysis = espn_service.get_comprehensive_player_analysis(player_id)
            
            return jsonify({
                'success': True,
                'player_id': player_id,
                'analysis': analysis,
                'data_sources': [
                    'ESPN Player Stats API',
                    'ESPN Game Log API', 
                    'ESPN Situational Splits API'
                ],
                'confidence': 'high',
                'last_updated': analysis.get('generated_at')
            })
            
        except Exception as e:
            logger.error(f"Enhanced player analysis error: {e}")
            return jsonify({
                'success': False,
                'error': str(e),
                'player_id': player_id
            }), 500
    
    @app.route('/api/enhanced/predict', methods=['POST'])
    def enhanced_prediction():
        """
        Generate enhanced predictions using comprehensive ESPN data
        """
        try:
            data = request.get_json()
            
            # Required parameters
            player_id = data.get('player_id')
            position = data.get('position')
            prediction_type = data.get('prediction_type')
            team_id = data.get('team_id')
            
            # Optional parameters
            opponent_id = data.get('opponent_id')
            is_home = data.get('is_home', True)
            weather_conditions = data.get('weather_conditions', {})
            
            if not all([player_id, position, prediction_type, team_id]):
                return jsonify({
                    'success': False,
                    'error': 'Missing required parameters: player_id, position, prediction_type, team_id'
                }), 400
            
            # Generate enhanced prediction
            prediction = ml_predictor.predict_enhanced_performance(
                player_id=player_id,
                position=position,
                prediction_type=prediction_type,
                team_id=team_id,
                opponent_id=opponent_id,
                is_home=is_home,
                weather_conditions=weather_conditions
            )
            
            return jsonify({
                'success': True,
                'prediction': {
                    'player_name': prediction.player_name,
                    'position': prediction.position,
                    'prediction_type': prediction.prediction_type,
                    'predicted_value': prediction.predicted_value,
                    'confidence': prediction.confidence,
                    'range': {
                        'low': prediction.range_low,
                        'high': prediction.range_high
                    },
                    'key_factors': prediction.key_factors,
                    'recent_trends': prediction.recent_trends,
                    'matchup_factors': prediction.matchup_factors,
                    'situational_modifiers': prediction.situational_modifiers
                },
                'betting_insights': {
                    'recommendation': prediction.betting_insights.betting_recommendation,
                    'confidence': prediction.betting_insights.confidence,
                    'key_factors': prediction.betting_insights.key_factors,
                    'recent_trends': prediction.betting_insights.recent_trends,
                    'matchup_advantages': prediction.betting_insights.matchup_advantages
                },
                'data_sources': [
                    'ESPN Player Statistics API',
                    'ESPN Game Log API',
                    'ESPN Situational Splits API',
                    'ESPN Team Statistics API',
                    'XGBoost ML Models'
                ],
                'model_version': 'enhanced_v1.0'
            })
            
        except Exception as e:
            logger.error(f"Enhanced prediction error: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/enhanced/team/<team_id>/insights')
    def get_team_insights(team_id):
        """
        Get comprehensive team insights for betting analysis
        """
        try:
            # Get team statistics
            team_stats = espn_service.get_team_statistics(team_id)
            roster = espn_service.get_roster_with_depth(team_id)
            
            # Analyze key players
            key_players = []
            for position, players in roster.items():
                if position in ['QB', 'RB', 'WR', 'TE'] and players:
                    starter = players[0]  # First player is typically starter
                    key_players.append({
                        'position': position,
                        'name': starter['name'],
                        'id': starter['id'],
                        'depth_rank': starter['depth_rank'],
                        'experience': starter['experience']
                    })
            
            return jsonify({
                'success': True,
                'team_id': team_id,
                'team_statistics': {
                    'offensive_rankings': team_stats.offensive_rankings if hasattr(team_stats, 'offensive_rankings') else {},
                    'defensive_rankings': team_stats.defensive_rankings if hasattr(team_stats, 'defensive_rankings') else {},
                    'efficiency_metrics': team_stats.efficiency_metrics if hasattr(team_stats, 'efficiency_metrics') else {},
                    'red_zone_stats': team_stats.red_zone_stats if hasattr(team_stats, 'red_zone_stats') else {}
                },
                'key_players': key_players,
                'roster_depth': {pos: len(players) for pos, players in roster.items()},
                'betting_insights': {
                    'offensive_strength': _analyze_offensive_strength(team_stats),
                    'defensive_strength': _analyze_defensive_strength(team_stats),
                    'key_matchup_factors': _identify_matchup_factors(team_stats)
                },
                'data_sources': [
                    'ESPN Team Statistics API',
                    'ESPN Team Roster API'
                ]
            })
            
        except Exception as e:
            logger.error(f"Team insights error: {e}")
            return jsonify({
                'success': False,
                'error': str(e),
                'team_id': team_id
            }), 500
    
    @app.route('/api/enhanced/matchup')
    def get_matchup_analysis():
        """
        Get comprehensive matchup analysis between two teams
        """
        try:
            home_team_id = request.args.get('home_team')
            away_team_id = request.args.get('away_team')
            
            if not home_team_id or not away_team_id:
                return jsonify({
                    'success': False,
                    'error': 'Missing required parameters: home_team and away_team'
                }), 400
            
            # Get team data
            home_stats = espn_service.get_team_statistics(home_team_id)
            away_stats = espn_service.get_team_statistics(away_team_id)
            
            # Analyze matchup
            matchup_analysis = _analyze_team_matchup(home_stats, away_stats)
            
            return jsonify({
                'success': True,
                'matchup': {
                    'home_team_id': home_team_id,
                    'away_team_id': away_team_id,
                    'analysis': matchup_analysis,
                    'betting_angles': _generate_betting_angles(home_stats, away_stats),
                    'key_player_matchups': _identify_key_player_matchups(home_team_id, away_team_id),
                    'confidence': 'high'
                },
                'data_sources': [
                    'ESPN Team Statistics API',
                    'ESPN Efficiency Metrics'
                ]
            })
            
        except Exception as e:
            logger.error(f"Matchup analysis error: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/enhanced/betting-insights/<position>')
    def get_position_betting_insights(position):
        """
        Get position-specific betting insights across all teams
        """
        try:
            # Get all team statistics
            all_team_stats = espn_service.get_team_statistics()
            
            # Analyze position-specific trends
            insights = _analyze_position_trends(position, all_team_stats)
            
            return jsonify({
                'success': True,
                'position': position,
                'insights': insights,
                'market_trends': _get_market_trends(position),
                'top_plays': _identify_top_plays(position, insights),
                'risk_factors': _identify_risk_factors(position, insights),
                'data_sources': [
                    'ESPN League Statistics',
                    'ESPN Team Rankings'
                ]
            })
            
        except Exception as e:
            logger.error(f"Position betting insights error: {e}")
            return jsonify({
                'success': False,
                'error': str(e),
                'position': position
            }), 500
    
    @app.route('/api/enhanced/trending')
    def get_trending_insights():
        """
        Get trending players and betting opportunities
        """
        try:
            trending_players = espn_service.get_trending_players(limit=20)
            
            # Generate betting insights for trending players
            insights = []
            for player in trending_players[:10]:  # Limit processing
                try:
                    betting_insight = espn_service.generate_betting_insights(
                        player['id'], 'fantasy_points'
                    )
                    insights.append({
                        'player': player,
                        'betting_insight': {
                            'predicted_value': betting_insight.predicted_value,
                            'confidence': betting_insight.confidence,
                            'recommendation': betting_insight.betting_recommendation,
                            'key_factors': betting_insight.key_factors[:3]  # Top 3 factors
                        }
                    })
                except Exception as e:
                    logger.warning(f"Error generating insight for player {player['id']}: {e}")
                    continue
            
            return jsonify({
                'success': True,
                'trending_insights': insights,
                'market_summary': {
                    'total_trending_players': len(trending_players),
                    'high_confidence_plays': len([i for i in insights if i['betting_insight']['confidence'] > 0.8]),
                    'moderate_confidence_plays': len([i for i in insights if 0.6 < i['betting_insight']['confidence'] <= 0.8])
                },
                'data_sources': [
                    'ESPN Trending Players',
                    'ESPN News API'
                ]
            })
            
        except Exception as e:
            logger.error(f"Trending insights error: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    # Helper functions
    def _analyze_offensive_strength(team_stats):
        """Analyze team offensive strength for betting insights"""
        if not hasattr(team_stats, 'offensive_rankings'):
            return "Limited data available"
        
        rankings = team_stats.offensive_rankings
        avg_rank = sum(rankings.values()) / max(len(rankings), 1)
        
        if avg_rank <= 8:
            return "Elite offense - high scoring potential"
        elif avg_rank <= 16:
            return "Above average offense - moderate scoring"
        elif avg_rank <= 24:
            return "Below average offense - limited scoring"
        else:
            return "Poor offense - low scoring potential"
    
    def _analyze_defensive_strength(team_stats):
        """Analyze team defensive strength"""
        if not hasattr(team_stats, 'defensive_rankings'):
            return "Limited data available"
        
        rankings = team_stats.defensive_rankings
        avg_rank = sum(rankings.values()) / max(len(rankings), 1)
        
        if avg_rank <= 8:
            return "Elite defense - tough matchup for opponents"
        elif avg_rank <= 16:
            return "Solid defense - average matchup"
        elif avg_rank <= 24:
            return "Vulnerable defense - favorable for opponents"
        else:
            return "Poor defense - high scoring potential for opponents"
    
    def _identify_matchup_factors(team_stats):
        """Identify key matchup factors for betting"""
        factors = []
        
        if hasattr(team_stats, 'red_zone_stats'):
            rz_efficiency = team_stats.red_zone_stats.get('red_zone_efficiency', 0.5)
            if rz_efficiency > 0.6:
                factors.append("Strong red zone offense")
            elif rz_efficiency < 0.4:
                factors.append("Struggles in red zone")
        
        if hasattr(team_stats, 'turnover_stats'):
            turnover_diff = team_stats.turnover_stats.get('turnover_differential', 0)
            if turnover_diff > 0.5:
                factors.append("Positive turnover differential")
            elif turnover_diff < -0.5:
                factors.append("Negative turnover differential")
        
        return factors
    
    def _analyze_team_matchup(home_stats, away_stats):
        """Analyze matchup between two teams"""
        return {
            'home_advantages': _get_team_advantages(home_stats),
            'away_advantages': _get_team_advantages(away_stats),
            'key_battles': _identify_key_battles(home_stats, away_stats),
            'scoring_projection': _project_scoring(home_stats, away_stats)
        }
    
    def _get_team_advantages(team_stats):
        """Get team's key advantages"""
        advantages = []
        
        if hasattr(team_stats, 'offensive_rankings'):
            for stat, rank in team_stats.offensive_rankings.items():
                if rank <= 5:  # Top 5 in league
                    advantages.append(f"Elite {stat} (#{rank})")
        
        return advantages[:3]  # Top 3 advantages
    
    def _identify_key_battles(home_stats, away_stats):
        """Identify key matchup battles"""
        return [
            "Home rush offense vs Away rush defense",
            "Away pass offense vs Home pass defense",
            "Red zone efficiency battle",
            "Turnover margin competition"
        ]
    
    def _project_scoring(home_stats, away_stats):
        """Project game scoring based on team stats"""
        # Simplified scoring projection
        home_proj = 21 + (sum(getattr(home_stats, 'efficiency_metrics', {}).values()) / 10)
        away_proj = 21 + (sum(getattr(away_stats, 'efficiency_metrics', {}).values()) / 10)
        
        return {
            'home_projected': round(home_proj, 1),
            'away_projected': round(away_proj, 1),
            'total_projected': round(home_proj + away_proj, 1)
        }
    
    def _generate_betting_angles(home_stats, away_stats):
        """Generate specific betting angles"""
        return [
            {
                'type': 'total_points',
                'recommendation': 'Over' if _project_scoring(home_stats, away_stats)['total_projected'] > 45 else 'Under',
                'confidence': 'moderate'
            },
            {
                'type': 'point_spread',
                'recommendation': 'Analyze home field advantage',
                'confidence': 'low'
            }
        ]
    
    def _identify_key_player_matchups(home_team_id, away_team_id):
        """Identify key player vs player/unit matchups"""
        return [
            "Starting QB vs Pass Rush",
            "Top WR vs #1 CB",
            "RB vs Run Defense",
            "TE vs Linebacker Coverage"
        ]
    
    def _analyze_position_trends(position, all_team_stats):
        """Analyze trends for specific position"""
        return {
            'league_average': f"Analysis for {position} position",
            'top_performers': f"Top {position} plays this week",
            'value_plays': f"Undervalued {position} options",
            'avoid_list': f"{position} players to avoid"
        }
    
    def _get_market_trends(position):
        """Get market trends for position"""
        return {
            'popular_plays': f"Most popular {position} bets",
            'contrarian_plays': f"Contrarian {position} opportunities",
            'line_movement': f"{position} line movement analysis"
        }
    
    def _identify_top_plays(position, insights):
        """Identify top plays for position"""
        return [
            f"Top {position} play #1",
            f"Top {position} play #2", 
            f"Top {position} play #3"
        ]
    
    def _identify_risk_factors(position, insights):
        """Identify risk factors for position"""
        return [
            f"{position} injury concerns",
            f"{position} weather impacts",
            f"{position} lineup uncertainty"
        ]
    
    logger.info("✅ Enhanced API endpoints initialized")
    return espn_service, ml_predictor