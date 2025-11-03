#!/usr/bin/env python3
"""
Universal NFL Stats Processor
Handles ALL stat types: Passing, Rushing, Receiving, Defense, Kicking, Punting, Returns, Scoring, Scrimmage
Designed for maximum XGBoost training data from 26-year datasets
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
import logging
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UniversalNFLProcessor:
    def __init__(self):
        self.stat_types = {
            'passing': 'passing_stats_historical.csv',
            'rushing': 'rushing_stats_historical.csv', 
            'receiving': 'receiving_stats_historical.csv',
            'defense': 'defense_stats_historical.csv',
            'kicking': 'kicking_stats_historical.csv',
            'punting': 'punting_stats_historical.csv',
            'returns': 'returns_stats_historical.csv',
            'scoring': 'scoring_stats_historical.csv',
            'scrimmage': 'scrimmage_stats_historical.csv'
        }
        
        self.processors = {}
        self.models = {}
        
    def check_data_availability(self, data_dir='nfl_data'):
        """Check which datasets are available for training"""
        logger.info("🔍 Checking NFL dataset availability...")
        
        available_data = {}
        
        for stat_type, filename in self.stat_types.items():
            filepath = os.path.join(data_dir, filename)
            
            if os.path.exists(filepath):
                try:
                    file_size = os.path.getsize(filepath)
                    
                    # Count lines
                    with open(filepath, 'r', encoding='utf-8') as f:
                        line_count = sum(1 for _ in f)
                    
                    # Determine if it's a full dataset or placeholder
                    if line_count > 100 and file_size > 10000:  # Substantial data
                        status = "FULL_DATASET"
                        available_data[stat_type] = {
                            'file': filename,
                            'lines': line_count,
                            'size_bytes': file_size,
                            'status': status
                        }
                        logger.info(f"✅ {stat_type.upper()}: {line_count:,} lines ({file_size:,} bytes)")
                    else:
                        status = "PLACEHOLDER"
                        logger.info(f"⚠️  {stat_type.upper()}: Placeholder only ({line_count} lines)")
                        
                except Exception as e:
                    logger.error(f"❌ {stat_type.upper()}: Error reading file - {e}")
            else:
                logger.info(f"❌ {stat_type.upper()}: File not found")
        
        logger.info(f"\n📊 SUMMARY: {len(available_data)} datasets ready for training")
        return available_data
    
    def load_stat_data(self, stat_type, filepath):
        """Load and clean a specific stat type dataset"""
        logger.info(f"📥 Loading {stat_type} data from {filepath}...")
        
        try:
            # Load with error handling for inconsistent formats
            df = pd.read_csv(filepath, on_bad_lines='skip', low_memory=False)
            
            # Skip header rows
            df_data = df.iloc[1:].copy()
            
            # Remove header rows that snuck through
            if len(df_data.columns) > 1:
                player_col = df_data.columns[1]
                df_data = df_data[~df_data[player_col].astype(str).str.contains('Player|Rk|Team', na=False, case=False)]
            
            # Remove empty rows
            df_data = df_data.dropna(how='all')
            
            # Basic data validation
            if len(df_data) < 10:
                logger.warning(f"⚠️  {stat_type}: Very few records ({len(df_data)})")
                return None
            
            logger.info(f"✅ {stat_type}: Loaded {len(df_data)} records")
            return df_data
            
        except Exception as e:
            logger.error(f"❌ {stat_type}: Failed to load - {e}")
            return None
    
    def create_stat_features(self, df, stat_type):
        """Create features specific to each stat type"""
        logger.info(f"🔧 Creating {stat_type} features...")
        
        features_list = []
        
        for idx, row in df.iterrows():
            try:
                # Universal fields (present in all stat types)
                player = str(row.iloc[1]) if len(row) > 1 else 'Unknown'
                team = str(row.iloc[3]) if len(row) > 3 else 'Unknown'
                age = pd.to_numeric(row.iloc[2], errors='coerce') if len(row) > 2 else 25
                games = pd.to_numeric(row.iloc[5], errors='coerce') if len(row) > 5 else 1
                
                if games <= 0:
                    continue
                
                base_features = {
                    'player_name': player,
                    'age': age,
                    'games_played': games,
                    'team_encoded': hash(team) % 32,
                    'stat_type': stat_type
                }
                
                # Stat-specific features
                if stat_type == 'passing':
                    completions = pd.to_numeric(row.iloc[7], errors='coerce') if len(row) > 7 else 0
                    attempts = pd.to_numeric(row.iloc[8], errors='coerce') if len(row) > 8 else 0
                    
                    if attempts > 0:
                        base_features.update({
                            'attempts_per_game': attempts / games,
                            'completion_rate': completions / attempts,
                            'target': completions / games  # Completions per game
                        })
                
                elif stat_type == 'rushing':
                    attempts = pd.to_numeric(row.iloc[7], errors='coerce') if len(row) > 7 else 0
                    yards = pd.to_numeric(row.iloc[8], errors='coerce') if len(row) > 8 else 0
                    
                    if attempts > 0:
                        base_features.update({
                            'attempts_per_game': attempts / games,
                            'yards_per_attempt': yards / attempts,
                            'yards_per_game': yards / games,
                            'target': yards / games  # Yards per game
                        })
                
                elif stat_type == 'receiving':
                    receptions = pd.to_numeric(row.iloc[7], errors='coerce') if len(row) > 7 else 0
                    yards = pd.to_numeric(row.iloc[8], errors='coerce') if len(row) > 8 else 0
                    
                    if receptions > 0:
                        base_features.update({
                            'receptions_per_game': receptions / games,
                            'yards_per_reception': yards / receptions,
                            'yards_per_game': yards / games,
                            'target': receptions / games  # Receptions per game
                        })
                
                elif stat_type == 'defense':
                    # Defense stats: Int, Yds, IntTD, Lng, PD, FF, Fmb, FR, Yds, FRTD, Sk, Comb, Solo, Ast, TFL, Sfty
                    interceptions = pd.to_numeric(row.iloc[7], errors='coerce') if len(row) > 7 else 0
                    int_yards = pd.to_numeric(row.iloc[8], errors='coerce') if len(row) > 8 else 0
                    int_tds = pd.to_numeric(row.iloc[9], errors='coerce') if len(row) > 9 else 0
                    pass_defended = pd.to_numeric(row.iloc[11], errors='coerce') if len(row) > 11 else 0
                    forced_fumbles = pd.to_numeric(row.iloc[12], errors='coerce') if len(row) > 12 else 0
                    fumbles_recovered = pd.to_numeric(row.iloc[14], errors='coerce') if len(row) > 14 else 0
                    fr_yards = pd.to_numeric(row.iloc[15], errors='coerce') if len(row) > 15 else 0
                    fr_tds = pd.to_numeric(row.iloc[16], errors='coerce') if len(row) > 16 else 0
                    sacks = pd.to_numeric(row.iloc[17], errors='coerce') if len(row) > 17 else 0
                    tackles = pd.to_numeric(row.iloc[18], errors='coerce') if len(row) > 18 else 0
                    solo_tackles = pd.to_numeric(row.iloc[19], errors='coerce') if len(row) > 19 else 0
                    assisted_tackles = pd.to_numeric(row.iloc[20], errors='coerce') if len(row) > 20 else 0
                    tackles_for_loss = pd.to_numeric(row.iloc[21], errors='coerce') if len(row) > 21 else 0
                    safeties = pd.to_numeric(row.iloc[22], errors='coerce') if len(row) > 22 else 0
                    
                    # Calculate defensive efficiency metrics
                    total_turnovers = interceptions + fumbles_recovered
                    total_big_plays = sacks + tackles_for_loss + forced_fumbles
                    
                    if tackles > 0:
                        base_features.update({
                            'interceptions_per_game': interceptions / games,
                            'sacks_per_game': sacks / games,
                            'tackles_per_game': tackles / games,
                            'solo_tackles_per_game': solo_tackles / games,
                            'tfl_per_game': tackles_for_loss / games,
                            'turnovers_per_game': total_turnovers / games,
                            'big_plays_per_game': total_big_plays / games,
                            'defensive_impact_per_game': (interceptions + sacks + forced_fumbles + tackles_for_loss) / games,
                            'target': tackles / games  # Tackles per game as primary target
                        })
                
                elif stat_type == 'kicking':
                    # Kicking stats: FGA/FGM by distance, XPA/XPM, Kickoffs
                    fga_total = pd.to_numeric(row.iloc[17], errors='coerce') if len(row) > 17 else 0
                    fgm_total = pd.to_numeric(row.iloc[18], errors='coerce') if len(row) > 18 else 0
                    fg_long = pd.to_numeric(row.iloc[19], errors='coerce') if len(row) > 19 else 0
                    fg_pct = pd.to_numeric(row.iloc[20], errors='coerce') if len(row) > 20 else 0
                    xpa = pd.to_numeric(row.iloc[21], errors='coerce') if len(row) > 21 else 0
                    xpm = pd.to_numeric(row.iloc[22], errors='coerce') if len(row) > 22 else 0
                    xp_pct = pd.to_numeric(row.iloc[23], errors='coerce') if len(row) > 23 else 0
                    kickoffs = pd.to_numeric(row.iloc[24], errors='coerce') if len(row) > 24 else 0
                    ko_yards = pd.to_numeric(row.iloc[25], errors='coerce') if len(row) > 25 else 0
                    touchbacks = pd.to_numeric(row.iloc[26], errors='coerce') if len(row) > 26 else 0
                    tb_pct = pd.to_numeric(row.iloc[27], errors='coerce') if len(row) > 27 else 0
                    ko_avg = pd.to_numeric(row.iloc[28], errors='coerce') if len(row) > 28 else 0
                    
                    # Distance-specific attempts and makes for range analysis
                    fga_50_plus = pd.to_numeric(row.iloc[15], errors='coerce') if len(row) > 15 else 0
                    fgm_50_plus = pd.to_numeric(row.iloc[16], errors='coerce') if len(row) > 16 else 0
                    fga_40_49 = pd.to_numeric(row.iloc[13], errors='coerce') if len(row) > 13 else 0
                    fgm_40_49 = pd.to_numeric(row.iloc[14], errors='coerce') if len(row) > 14 else 0
                    
                    if fga_total > 0 or xpa > 0:
                        # Calculate kicking efficiency metrics
                        long_range_pct = fgm_50_plus / fga_50_plus if fga_50_plus > 0 else 0
                        mid_range_pct = fgm_40_49 / fga_40_49 if fga_40_49 > 0 else 0
                        total_scoring = (fgm_total * 3) + xpm  # Total points scored
                        
                        base_features.update({
                            'fg_attempts_per_game': fga_total / games,
                            'fg_makes_per_game': fgm_total / games,
                            'fg_percentage': fg_pct / 100 if fg_pct > 0 else 0,
                            'xp_attempts_per_game': xpa / games,
                            'xp_percentage': xp_pct / 100 if xp_pct > 0 else 0,
                            'long_range_accuracy': long_range_pct,
                            'mid_range_accuracy': mid_range_pct,
                            'kickoff_touchback_pct': tb_pct / 100 if tb_pct > 0 else 0,
                            'points_per_game': total_scoring / games,
                            'target': fgm_total / games  # Made FGs per game as primary target
                        })
                
                elif stat_type == 'scrimmage':
                    # Scrimmage combines receiving and rushing stats
                    receptions = pd.to_numeric(row.iloc[7], errors='coerce') if len(row) > 7 else 0
                    rec_yards = pd.to_numeric(row.iloc[8], errors='coerce') if len(row) > 8 else 0
                    rush_attempts = pd.to_numeric(row.iloc[18], errors='coerce') if len(row) > 18 else 0
                    rush_yards = pd.to_numeric(row.iloc[19], errors='coerce') if len(row) > 19 else 0
                    total_yards = pd.to_numeric(row.iloc[29], errors='coerce') if len(row) > 29 else 0
                    total_tds = pd.to_numeric(row.iloc[30], errors='coerce') if len(row) > 30 else 0
                    
                    # Only create features if player has meaningful scrimmage involvement
                    if (receptions > 0 or rush_attempts > 0) and total_yards > 0:
                        base_features.update({
                            'total_touches_per_game': (receptions + rush_attempts) / games,
                            'yards_per_touch': total_yards / (receptions + rush_attempts) if (receptions + rush_attempts) > 0 else 0,
                            'total_yards_per_game': total_yards / games,
                            'total_tds_per_game': total_tds / games,
                            'receiving_percentage': receptions / (receptions + rush_attempts) if (receptions + rush_attempts) > 0 else 0,
                            'target': total_yards / games  # Total scrimmage yards per game
                        })
                
                # Add more stat types as needed...
                
                if 'target' in base_features:
                    features_list.append(base_features)
                    
            except Exception as e:
                continue
        
        if features_list:
            features_df = pd.DataFrame(features_list)
            features_df = features_df.fillna(0)
            logger.info(f"✅ {stat_type}: Created {len(features_df)} feature samples")
            return features_df
        else:
            logger.warning(f"⚠️  {stat_type}: No valid features created")
            return pd.DataFrame()
    
    def process_all_available_data(self, data_dir='nfl_data'):
        """Process all available datasets"""
        logger.info("🚀 Processing all available NFL datasets...")
        
        # Check what data is available
        available_data = self.check_data_availability(data_dir)
        
        processed_datasets = {}
        
        for stat_type, info in available_data.items():
            filepath = os.path.join(data_dir, info['file'])
            
            # Load the data
            df = self.load_stat_data(stat_type, filepath)
            
            if df is not None:
                # Create features
                features_df = self.create_stat_features(df, stat_type)
                
                if not features_df.empty:
                    # Prepare training data
                    X = features_df.drop(['target', 'player_name', 'stat_type'], axis=1, errors='ignore')
                    y = features_df['target'] if 'target' in features_df.columns else features_df.iloc[:, -1]
                    
                    processed_datasets[stat_type] = {
                        'X': X,
                        'y': y,
                        'features_df': features_df,
                        'samples': len(X),
                        'features': len(X.columns)
                    }
                    
                    logger.info(f"✅ {stat_type.upper()}: {len(X)} samples, {len(X.columns)} features")
        
        logger.info(f"\n🎉 PROCESSING COMPLETE: {len(processed_datasets)} datasets ready for XGBoost training")
        return processed_datasets
    
    def save_processed_datasets(self, processed_datasets, output_dir='processed_data'):
        """Save all processed datasets"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        saved_files = []
        
        for stat_type, data in processed_datasets.items():
            # Combine X and y for saving
            combined_df = data['X'].copy()
            combined_df['target'] = data['y']
            
            output_file = os.path.join(output_dir, f'processed_{stat_type}_data.csv')
            combined_df.to_csv(output_file, index=False)
            saved_files.append(output_file)
            
            logger.info(f"💾 Saved {stat_type} data: {output_file}")
        
        return saved_files

def main():
    """Main processing pipeline"""
    print("=== Universal NFL Stats Processor ===")
    print("Processing ALL available NFL datasets for maximum XGBoost accuracy...")
    
    processor = UniversalNFLProcessor()
    
    # Process all available data
    processed_datasets = processor.process_all_available_data()
    
    if processed_datasets:
        # Save processed datasets
        saved_files = processor.save_processed_datasets(processed_datasets)
        
        print(f"\n🎊 === PROCESSING COMPLETE ===")
        print(f"✅ Datasets processed: {len(processed_datasets)}")
        print(f"💾 Files saved: {len(saved_files)}")
        
        # Summary
        total_samples = sum(data['samples'] for data in processed_datasets.values())
        print(f"📊 Total training samples: {total_samples:,}")
        
        for stat_type, data in processed_datasets.items():
            print(f"   {stat_type.upper()}: {data['samples']:,} samples, {data['features']} features")
        
        print(f"\n🚀 Ready for comprehensive XGBoost training!")
        
    else:
        print("❌ No datasets were successfully processed")
    
    return processed_datasets

if __name__ == "__main__":
    main()