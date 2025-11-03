#!/usr/bin/env python3
"""
Clean NFL Receiving Stats Historical Data - 14k+ records
Second largest dataset for receiving props and metrics
"""

import pandas as pd
import numpy as np

def clean_receiving_data():
    """Clean the receiving stats historical data"""
    print("🏈 Cleaning NFL Receiving Stats Historical Data...")
    print("📊 Target: 14,500+ receiving records for XGBoost training")
    
    # Read the raw file
    input_file = 'nfl_data/receiving_stats_historical.csv'
    output_file = 'nfl_data/receiving_stats_historical_clean.csv'
    
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
        
        # Standard receiving columns
        clean_header = [
            'Rk', 'Player', 'Age', 'Team', 'Pos', 'G', 'GS',
            'Tgt', 'Rec', 'Yds', 'Avg', 'Lng', 'TD', 'First', 'Succ', 'Awards'
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
                
            # Skip header lines (25 duplicates expected)
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
            'Rk', 'Age', 'G', 'GS', 'Tgt', 'Rec', 'Yds', 'Avg', 'Lng', 'TD', 'First', 'Succ'
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
        
        # Add receiving betting features
        print("➕ Adding receiving metrics for betting...")
        
        # Core receiving efficiency
        df['Catch_Rate'] = np.where(df['Tgt'] > 0, df['Rec'] / df['Tgt'], 0)
        df['Yards_per_Target'] = np.where(df['Tgt'] > 0, df['Yds'] / df['Tgt'], 0)
        df['Yards_per_Reception'] = np.where(df['Rec'] > 0, df['Yds'] / df['Rec'], 0)
        df['TD_per_Reception'] = np.where(df['Rec'] > 0, df['TD'] / df['Rec'], 0)
        
        # Per game metrics
        df['Targets_per_Game'] = df['Tgt'] / df['G'].replace(0, 1)
        df['Receptions_per_Game'] = df['Rec'] / df['G'].replace(0, 1)
        df['Receiving_Yards_per_Game'] = df['Yds'] / df['G'].replace(0, 1)
        df['Receiving_TDs_per_Game'] = df['TD'] / df['G'].replace(0, 1)
        
        # Receiving volume and efficiency tiers
        df['High_Volume_Receiver'] = np.where(df['Tgt'] >= 100, 1, 0)
        df['Red_Zone_Threat'] = np.where(df['TD'] >= 8, 1, 0)
        df['Consistent_Producer'] = np.where((df['Rec'] >= 60) & (df['Yds'] >= 800), 1, 0)
        df['Deep_Threat'] = np.where(df['Lng'] >= 40, 1, 0)
        
        # Betting prop calculations
        df['Rec_Yards_Target'] = df['Receiving_Yards_per_Game'] * 17  # Season projection
        df['Receptions_Target'] = df['Receptions_per_Game'] * 17
        df['Rec_TD_Target'] = df['Receiving_TDs_per_Game'] * 17
        df['Anytime_Rec_TD'] = np.where(df['TD'] > 0, 1, 0)
        
        # Quality metrics
        df['First_Down_Rate'] = np.where(df['Rec'] > 0, df['First'] / df['Rec'], 0)
        df['Success_Rate'] = np.where(df['Tgt'] > 0, df['Succ'] / df['Tgt'], 0)
        
        # Receiver role classification
        df['Receiver_Role'] = 'Depth'
        df.loc[df['Tgt'] >= 120, 'Receiver_Role'] = 'WR1'
        df.loc[(df['Tgt'] >= 80) & (df['Tgt'] < 120), 'Receiver_Role'] = 'WR2'
        df.loc[(df['Tgt'] >= 50) & (df['Tgt'] < 80), 'Receiver_Role'] = 'WR3'
        df.loc[(df['TD'] >= 8) & (df['Tgt'] < 50), 'Receiver_Role'] = 'Red_Zone'
        df.loc[df['Pos'].str.contains('TE', na=False), 'Receiver_Role'] = 'TE'
        
        # Add estimated year
        df['Year'] = 2000 + (df.index % 26)
        
        # Final cleanup
        df = df.replace([np.inf, -np.inf], 0)
        df = df.round(3)
        
        print(f"✅ Final dataset: {len(df):,} records")
        print(f"📊 Position breakdown:")
        print(df['Pos'].value_counts().head(10))
        print(f"🏈 Receiver role breakdown:")
        print(df['Receiver_Role'].value_counts())
        
        # Save cleaned data
        df.to_csv(output_file, index=False)
        print(f"💾 Saved clean data: {output_file}")
        
        # Show sample of cleaned data
        print(f"\n📋 Sample of cleaned receiving data:")
        sample_cols = ['Player', 'Pos', 'G', 'Tgt', 'Rec', 'Yds', 'TD', 'Receiving_Yards_per_Game', 'Receiver_Role']
        available_cols = [col for col in sample_cols if col in df.columns]
        print(df[available_cols].head(10).to_string(index=False))
        
        return df
        
    except Exception as e:
        print(f"❌ Error cleaning receiving stats: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    clean_receiving_data()