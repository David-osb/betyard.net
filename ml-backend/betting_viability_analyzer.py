#!/usr/bin/env python3
"""
BETTING ANALYSIS FOR NFL POSITION PREDICTIONS
===========================================

Analyzes whether the position predictions are suitable for betting purposes.
Identifies issues and provides betting-specific evaluation.
"""

import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def analyze_betting_viability():
    """Analyze if the predictions are good for betting"""
    logger.info("=" * 60)
    logger.info("BETTING VIABILITY ANALYSIS")
    logger.info("=" * 60)
    
    try:
        # Load predictions
        df = pd.read_csv('nfl_position_predictions_2025.csv')
        logger.info(f"Loaded predictions for {len(df)} players")
        
        # Analyze data quality issues
        logger.info("\n📊 DATA QUALITY ANALYSIS:")
        logger.info("-" * 40)
        
        # Check for missing data by position
        for position in ['QB', 'RB', 'WR', 'TE']:
            pos_data = df[df['Position'] == position]
            logger.info(f"\n{position} ({len(pos_data)} players):")
            
            if position == 'QB':
                targets = ['pass_yds_per_game', 'completions_per_game', 'attempts_per_game', 'pass_td_per_game']
            elif position == 'RB':
                targets = ['rush_yds_per_game', 'rush_td_per_game', 'carries_per_game', 'receptions_per_game']
            elif position == 'WR':
                targets = ['rec_yds_per_game', 'receptions_per_game', 'rec_td_per_game', 'targets_per_game']
            elif position == 'TE':
                targets = ['rec_yds_per_game', 'receptions_per_game', 'rec_td_per_game', 'targets_per_game', 'blocks_per_game']
            
            for target in targets:
                if target in df.columns:
                    valid_predictions = pos_data[target].notna().sum()
                    total_players = len(pos_data)
                    coverage = (valid_predictions / total_players) * 100 if total_players > 0 else 0
                    
                    if coverage == 0:
                        status = "❌ NO DATA"
                    elif coverage < 50:
                        status = "⚠️ POOR"
                    elif coverage < 80:
                        status = "🔶 PARTIAL"
                    else:
                        status = "✅ GOOD"
                    
                    logger.info(f"  {target}: {valid_predictions}/{total_players} ({coverage:.1f}%) {status}")
        
        # Analyze prediction reasonableness
        logger.info("\n🎯 PREDICTION REASONABLENESS:")
        logger.info("-" * 40)
        
        # QB Analysis
        qb_data = df[df['Position'] == 'QB'].copy()
        if not qb_data.empty and 'pass_yds_per_game' in qb_data.columns:
            qb_yards = qb_data['pass_yds_per_game'].dropna()
            logger.info(f"\nQB Passing Yards/Game:")
            logger.info(f"  Range: {qb_yards.min():.1f} - {qb_yards.max():.1f}")
            logger.info(f"  Average: {qb_yards.mean():.1f}")
            logger.info(f"  Median: {qb_yards.median():.1f}")
            
            # Check if realistic (NFL QBs typically: 200-350 yards/game)
            realistic_min, realistic_max = 150, 400
            unrealistic = qb_yards[(qb_yards < realistic_min) | (qb_yards > realistic_max)]
            if len(unrealistic) > 0:
                logger.info(f"  ⚠️ {len(unrealistic)} QBs have unrealistic predictions")
            else:
                logger.info(f"  ✅ All QB predictions seem realistic")
        
        # WR Analysis  
        wr_data = df[df['Position'] == 'WR'].copy()
        if not wr_data.empty and 'rec_yds_per_game' in wr_data.columns:
            wr_yards = wr_data['rec_yds_per_game'].dropna()
            logger.info(f"\nWR Receiving Yards/Game:")
            logger.info(f"  Range: {wr_yards.min():.1f} - {wr_yards.max():.1f}")
            logger.info(f"  Average: {wr_yards.mean():.1f}")
            logger.info(f"  Median: {wr_yards.median():.1f}")
            
            # Check if realistic (NFL WRs typically: 20-120 yards/game)
            realistic_min, realistic_max = 5, 150
            unrealistic = wr_yards[(wr_yards < realistic_min) | (wr_yards > realistic_max)]
            if len(unrealistic) > 0:
                logger.info(f"  ⚠️ {len(unrealistic)} WRs have unrealistic predictions")
            else:
                logger.info(f"  ✅ All WR predictions seem realistic")
        
        # Betting-specific analysis
        logger.info("\n💰 BETTING SUITABILITY ANALYSIS:")
        logger.info("-" * 40)
        
        betting_issues = []
        
        # Issue 1: Missing RB data
        rb_data = df[df['Position'] == 'RB']
        rb_targets = ['rush_yds_per_game', 'rush_td_per_game', 'carries_per_game']
        rb_coverage = 0
        for target in rb_targets:
            if target in df.columns:
                rb_coverage += rb_data[target].notna().sum()
        
        if rb_coverage == 0:
            betting_issues.append("❌ CRITICAL: No RB predictions available")
        
        # Issue 2: Model accuracy too high (data leakage indicator)
        logger.info(f"\n🔍 MODEL ACCURACY ANALYSIS:")
        logger.info(f"  QB Models: R² = 0.992-1.000 (PERFECT)")
        logger.info(f"  WR Models: R² = 0.989-1.000 (PERFECT)")
        logger.info(f"  TE Models: R² = 0.962-1.000 (NEAR PERFECT)")
        
        betting_issues.append("⚠️ MAJOR: Perfect accuracy suggests data leakage")
        betting_issues.append("⚠️ Models may not generalize to future performance")
        
        # Issue 3: Per-game vs season totals
        betting_issues.append("🔶 MINOR: Predictions are per-game, not season totals")
        
        # Issue 4: No injury/roster status
        betting_issues.append("🔶 MINOR: No injury or roster status consideration")
        
        # Issue 5: No opponent strength
        betting_issues.append("🔶 MINOR: No opponent strength or matchup analysis")
        
        # Overall betting assessment
        logger.info(f"\n🏈 OVERALL BETTING ASSESSMENT:")
        logger.info("-" * 40)
        
        for issue in betting_issues:
            logger.info(f"  {issue}")
        
        # Calculate betting readiness score
        critical_issues = len([x for x in betting_issues if "CRITICAL" in x])
        major_issues = len([x for x in betting_issues if "MAJOR" in x])
        minor_issues = len([x for x in betting_issues if "MINOR" in x])
        
        if critical_issues > 0:
            betting_score = 0
            recommendation = "NOT READY FOR BETTING"
        elif major_issues > 0:
            betting_score = 30
            recommendation = "NEEDS MAJOR IMPROVEMENTS"
        elif minor_issues > 2:
            betting_score = 60
            recommendation = "NEEDS MINOR IMPROVEMENTS"
        else:
            betting_score = 85
            recommendation = "READY FOR BETTING"
        
        logger.info(f"\n📊 BETTING READINESS SCORE: {betting_score}/100")
        logger.info(f"🎯 RECOMMENDATION: {recommendation}")
        
        # Specific recommendations
        logger.info(f"\n🔧 RECOMMENDED IMPROVEMENTS:")
        logger.info("-" * 40)
        
        if critical_issues > 0 or major_issues > 0:
            logger.info("1. ❗ Fix data leakage in models (perfect accuracy is suspicious)")
            logger.info("2. ❗ Implement proper temporal validation (past predicts future)")
            logger.info("3. ❗ Add RB prediction capabilities")
            logger.info("4. 📈 Target realistic accuracy (60-75% for sports betting)")
            logger.info("5. 🎯 Focus on specific betting markets (O/U yards, TDs, etc.)")
            logger.info("6. 📊 Add confidence intervals for predictions")
            logger.info("7. 🏈 Include matchup and opponent strength factors")
        else:
            logger.info("1. 📈 Add injury and roster status tracking")
            logger.info("2. 🏈 Include opponent strength analysis")
            logger.info("3. 📊 Convert to season totals for some betting markets")
            logger.info("4. 🎯 Add confidence intervals")
        
        logger.info("\n" + "=" * 60)
        
        return {
            'betting_score': betting_score,
            'recommendation': recommendation,
            'critical_issues': critical_issues,
            'major_issues': major_issues,
            'minor_issues': minor_issues
        }
        
    except Exception as e:
        logger.error(f"Error in analysis: {e}")
        return None

if __name__ == "__main__":
    results = analyze_betting_viability()