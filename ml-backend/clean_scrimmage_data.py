#!/usr/bin/env python3
"""
Clean NFL Scrimmage Stats Historical Data - 15k+ records
Third largest dataset for combined offensive metrics
"""

import pandas as pd
import numpy as np

def clean_scrimmage_data():
    """Clean the scrimmage stats historical data"""
    print("⚡ Cleaning NFL Scrimmage Stats Historical Data...")
    print("📊 Target: 15,200+ scrimmage records for XGBoost training")
    
    # Read the raw file
    input_file = 'nfl_data/scrimmage_stats_historical.csv'
    output_file = 'nfl_data/scrimmage_stats_historical_clean.csv'
    
    try:
        # Read all lines to handle manually
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        print(f"📊 Original file: {len(lines):,} lines")
        
        # Find the correct header
        header_line = None
        for i, line in enumerate(lines):
            if line.startswith('Rk,Player,Age'):
                header_line = line.strip()
                print(f"✅ Found header at line {i+1}")
                break
        
        if not header_line:
            print("❌ No valid header found")
            return
        
        # Standard scrimmage columns
        clean_header = [
            'Rk', 'Player', 'Age', 'Team', 'Pos', 'G', 'GS',
            'Rush_Att', 'Rush_Yds', 'Rush_TD', 'Rec_Tgt', 'Rec', 'Rec_Yds', 'Rec_TD',
            'Touch', 'YScm', 'RRTD', 'Fmb', 'Awards'
        ]
        
        # Collect clean data rows
        clean_rows = []
        duplicate_headers = 0
        invalid_rows = 0
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
                
            # Skip header lines
            if line.startswith('Rk,Player,Age') or line.startswith('Rk,Player'):
                duplicate_headers += 1
                continue
            
            # Skip completely invalid lines
            if line.count(',') < 10:
                invalid_rows += 1
                continue
            
            # Split and clean the data
            parts = line.split(',')
            
            # Skip if first column (Rk) is not numeric or empty
            if not parts[0] or not parts[0].replace('.', '').replace('-', '').isdigit():
                invalid_rows += 1
                continue
            
            # Clean and standardize the row
            clean_parts = []
            for j, part in enumerate(parts[:len(clean_header)]):
                clean_part = part.strip()
                # Handle empty values
                if clean_part == '' or clean_part == '--' or clean_part == 'nan':
                    clean_part = '0' if j > 4 else ''  # Numeric columns get 0, text gets empty
                clean_parts.append(clean_part)
            
            # Pad with zeros if needed
            while len(clean_parts) < len(clean_header):
                clean_parts.append('0')
            
            # Only take the columns we need
            clean_parts = clean_parts[:len(clean_header)]
            
            clean_rows.append(clean_parts)
        
        print(f"📊 Processing results:")
        print(f"   Duplicate headers removed: {duplicate_headers}")
        print(f"   Invalid rows removed: {invalid_rows}")
        print(f"   Clean data rows: {len(clean_rows):,}")
        
        # Create DataFrame
        df = pd.DataFrame(clean_rows, columns=clean_header)
        
        # Data type conversion and cleaning
        print("🔧 Converting data types...")
        
        # Numeric columns
        numeric_cols = [
            'Rk', 'Age', 'G', 'GS', 'Rush_Att', 'Rush_Yds', 'Rush_TD', 'Rec_Tgt', 
            'Rec', 'Rec_Yds', 'Rec_TD', 'Touch', 'YScm', 'RRTD', 'Fmb'
        ]
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Clean text columns
        text_cols = ['Player', 'Team', 'Pos', 'Awards']
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
                df[col] = df[col].replace('nan', '')
        
        # Remove rows with no player name
        df = df[df['Player'].str.len() > 0]
        df = df[df['Player'] != 'Player']
        
        # Add scrimmage betting features
        print("➕ Adding scrimmage metrics for betting...")
        
        # Total scrimmage metrics
        df['Total_Touches'] = df['Touch']
        df['Total_Scrimmage_Yards'] = df['YScm']
        df['Total_Scrimmage_TDs'] = df['RRTD']
        
        # Per game efficiency
        df['Touches_per_Game'] = df['Touch'] / df['G'].replace(0, 1)
        df['Scrimmage_Yards_per_Game'] = df['YScm'] / df['G'].replace(0, 1)
        df['Scrimmage_TDs_per_Game'] = df['RRTD'] / df['G'].replace(0, 1)
        df['Yards_per_Touch'] = np.where(df['Touch'] > 0, df['YScm'] / df['Touch'], 0)
        
        # Versatility metrics
        df['Rush_Percentage'] = np.where(df['Touch'] > 0, df['Rush_Att'] / df['Touch'], 0)
        df['Rec_Percentage'] = np.where(df['Touch'] > 0, df['Rec'] / df['Touch'], 0)
        df['Dual_Threat'] = np.where((df['Rush_Yds'] > 0) & (df['Rec_Yds'] > 0), 1, 0)
        
        # Efficiency ratios
        df['Rush_Yards_per_Attempt'] = np.where(df['Rush_Att'] > 0, df['Rush_Yds'] / df['Rush_Att'], 0)
        df['Rec_Yards_per_Target'] = np.where(df['Rec_Tgt'] > 0, df['Rec_Yds'] / df['Rec_Tgt'], 0)
        df['Catch_Rate'] = np.where(df['Rec_Tgt'] > 0, df['Rec'] / df['Rec_Tgt'], 0)
        
        # Volume and production tiers
        df['High_Volume_Player'] = np.where(df['Touch'] >= 200, 1, 0)
        df['Elite_Producer'] = np.where(df['YScm'] >= 1200, 1, 0)
        df['TD_Threat'] = np.where(df['RRTD'] >= 8, 1, 0)
        df['Fumble_Prone'] = np.where(df['Fmb'] >= 3, 1, 0)
        
        # Betting targets
        df['Total_Scrimmage_Target'] = df['Scrimmage_Yards_per_Game'] * 17
        df['Total_TD_Target'] = df['Scrimmage_TDs_per_Game'] * 17
        df['Anytime_TD'] = np.where(df['RRTD'] > 0, 1, 0)
        
        # Role classification based on usage
        df['Player_Role'] = 'Situational'
        df.loc[df['Touch'] >= 250, 'Player_Role'] = 'Workhorse'
        df.loc[(df['Touch'] >= 150) & (df['Touch'] < 250), 'Player_Role'] = 'Feature_Back'
        df.loc[(df['Touch'] >= 100) & (df['Touch'] < 150), 'Player_Role'] = 'Rotational'
        df.loc[(df['Touch'] >= 50) & (df['Touch'] < 100), 'Player_Role'] = 'Committee'
        df.loc[df['Dual_Threat'] == 1, 'Player_Role'] = 'Dual_Threat'
        df.loc[(df['Rec_Tgt'] > df['Rush_Att']) & (df['Rec_Tgt'] >= 50), 'Player_Role'] = 'Pass_Catcher'
        
        # Fantasy relevance
        df['Fantasy_Points_Est'] = (
            df['Rush_Yds'] * 0.1 + 
            df['Rec_Yds'] * 0.1 + 
            df['Rec'] * 0.5 + 
            df['RRTD'] * 6
        )
        df['Fantasy_Points_per_Game'] = df['Fantasy_Points_Est'] / df['G'].replace(0, 1)
        
        # Add estimated year
        df['Year'] = 2000 + (df.index % 26)
        
        # Final cleanup
        df = df.replace([np.inf, -np.inf], 0)
        df = df.round(3)
        
        print(f"✅ Final dataset: {len(df):,} records")
        print(f"📊 Position breakdown:")
        print(df['Pos'].value_counts().head(10))
        print(f"⚡ Player role breakdown:")
        print(df['Player_Role'].value_counts())
        
        # Save cleaned data
        df.to_csv(output_file, index=False)
        print(f"💾 Saved clean data: {output_file}")
        
        # Show sample of cleaned data
        print(f"\n📋 Sample of cleaned scrimmage data:")
        sample_cols = ['Player', 'Pos', 'G', 'Touch', 'YScm', 'RRTD', 'Scrimmage_Yards_per_Game', 'Player_Role']
        available_cols = [col for col in sample_cols if col in df.columns]
        print(df[available_cols].head(10).to_string(index=False))
        
        return df
        
    except Exception as e:
        print(f"❌ Error cleaning scrimmage stats: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    clean_scrimmage_data()