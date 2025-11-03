#!/usr/bin/env python3
"""
Clean NFL Rushing Stats Historical Data - 13k+ records
Fourth largest dataset for rushing props and metrics
"""

import pandas as pd
import numpy as np

def clean_rushing_data():
    """Clean the rushing stats historical data"""
    print("🏃 Cleaning NFL Rushing Stats Historical Data...")
    print("📊 Target: 13,200+ rushing records for XGBoost training")
    
    # Read the raw file
    input_file = 'nfl_data/rushing_stats_historical.csv'
    output_file = 'nfl_data/rushing_stats_historical_clean.csv'
    
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
        
        # Standard rushing columns
        clean_header = [
            'Rk', 'Player', 'Age', 'Team', 'Pos', 'G', 'GS',
            'Att', 'Yds', 'TD', 'Lng', 'YdsPerAtt', 'YdsPerG', 'AttPerG', 
            'First', 'FirstPct', 'BigRuns', 'Fmb', 'Awards'
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
                
            # Skip header lines (26 duplicates expected)
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
            'Rk', 'Age', 'G', 'GS', 'Att', 'Yds', 'TD', 'Lng', 'YdsPerAtt', 
            'YdsPerG', 'AttPerG', 'First', 'FirstPct', 'BigRuns', 'Fmb'
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
        
        # Add rushing betting features
        print("➕ Adding rushing metrics for betting...")
        
        # Core rushing efficiency
        df['Yards_per_Attempt'] = df['YdsPerAtt']
        df['Yards_per_Game'] = df['YdsPerG']
        df['Attempts_per_Game'] = df['AttPerG']
        df['TD_per_Attempt'] = np.where(df['Att'] > 0, df['TD'] / df['Att'], 0)
        df['TD_per_Game'] = df['TD'] / df['G'].replace(0, 1)
        
        # Volume and efficiency tiers
        df['High_Volume_Rusher'] = np.where(df['Att'] >= 200, 1, 0)
        df['Goal_Line_Back'] = np.where(df['TD'] >= 10, 1, 0)
        df['Efficient_Runner'] = np.where(df['YdsPerAtt'] >= 4.5, 1, 0)
        df['Big_Play_Threat'] = np.where(df['Lng'] >= 40, 1, 0)
        
        # Consistency metrics
        df['First_Down_Rate'] = df['FirstPct'] / 100.0
        df['Big_Run_Rate'] = np.where(df['Att'] > 0, df['BigRuns'] / df['Att'], 0)
        df['Fumble_Rate'] = np.where(df['Att'] > 0, df['Fmb'] / df['Att'], 0)
        
        # Betting prop calculations
        df['Rush_Yards_Target'] = df['YdsPerG'] * 17  # Season projection
        df['Rush_Attempts_Target'] = df['AttPerG'] * 17
        df['Rush_TD_Target'] = df['TD_per_Game'] * 17
        df['Anytime_Rush_TD'] = np.where(df['TD'] > 0, 1, 0)
        
        # Running back role classification
        df['Rusher_Role'] = 'Change_of_Pace'
        df.loc[df['Att'] >= 250, 'Rusher_Role'] = 'Workhorse'
        df.loc[(df['Att'] >= 150) & (df['Att'] < 250), 'Rusher_Role'] = 'Feature_Back'
        df.loc[(df['Att'] >= 100) & (df['Att'] < 150), 'Rusher_Role'] = 'Committee'
        df.loc[(df['TD'] >= 8) & (df['Att'] < 100), 'Rusher_Role'] = 'Goal_Line'
        df.loc[df['YdsPerAtt'] >= 5.0, 'Rusher_Role'] = 'Explosive'
        df.loc[df['Pos'].str.contains('QB', na=False), 'Rusher_Role'] = 'Mobile_QB'
        
        # Performance tiers
        df['Elite_Rusher'] = np.where((df['Yds'] >= 1000) & (df['YdsPerAtt'] >= 4.0), 1, 0)
        df['Volume_Dependent'] = np.where((df['Att'] >= 200) & (df['YdsPerAtt'] < 4.0), 1, 0)
        df['Touchdown_Vulture'] = np.where((df['TD'] >= 8) & (df['Yds'] < 500), 1, 0)
        
        # Game script dependency
        df['Workload_Heavy'] = np.where(df['AttPerG'] >= 15, 1, 0)
        df['Situational_Runner'] = np.where(df['AttPerG'] < 8, 1, 0)
        
        # Durability factors
        df['Injury_Risk'] = np.where((df['Att'] >= 300) | (df['Fmb'] >= 4), 1, 0)
        df['Games_Missed'] = 17 - df['G']
        df['Availability'] = df['G'] / 17.0
        
        # Add estimated year
        df['Year'] = 2000 + (df.index % 26)
        
        # Final cleanup
        df = df.replace([np.inf, -np.inf], 0)
        df = df.round(3)
        
        print(f"✅ Final dataset: {len(df):,} records")
        print(f"📊 Position breakdown:")
        print(df['Pos'].value_counts().head(10))
        print(f"🏃 Rusher role breakdown:")
        print(df['Rusher_Role'].value_counts())
        
        # Save cleaned data
        df.to_csv(output_file, index=False)
        print(f"💾 Saved clean data: {output_file}")
        
        # Show sample of cleaned data
        print(f"\n📋 Sample of cleaned rushing data:")
        sample_cols = ['Player', 'Pos', 'G', 'Att', 'Yds', 'TD', 'YdsPerAtt', 'Rusher_Role']
        available_cols = [col for col in sample_cols if col in df.columns]
        print(df[available_cols].head(10).to_string(index=False))
        
        return df
        
    except Exception as e:
        print(f"❌ Error cleaning rushing stats: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    clean_rushing_data()