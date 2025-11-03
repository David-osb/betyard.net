#!/usr/bin/env python3
"""
Clean NFL Passing Stats Historical Data - 2.7k+ records
QB passing metrics for quarterback props
"""

import pandas as pd
import numpy as np

def clean_passing_data():
    """Clean the passing stats historical data"""
    print("🏈 Cleaning NFL Passing Stats Historical Data...")
    print("📊 Target: 2,700+ passing records for XGBoost training")
    
    # Read the raw file
    input_file = 'nfl_data/passing_stats_historical.csv'
    output_file = 'nfl_data/passing_stats_historical_clean.csv'
    
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
        
        # Standard passing columns
        clean_header = [
            'Rk', 'Player', 'Age', 'Team', 'Pos', 'G', 'GS', 'QBrec',
            'Cmp', 'Att', 'CmpPct', 'Yds', 'TD', 'Int', 'Rate', 'QBR',
            'YdsPerAtt', 'YdsPerCmp', 'YdsPerG', 'Sacked', 'Awards'
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
                    clean_part = '0' if j > 6 else ''  # Numeric columns get 0, text gets empty
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
            'Rk', 'Age', 'G', 'GS', 'Cmp', 'Att', 'CmpPct', 'Yds', 'TD', 'Int', 
            'Rate', 'QBR', 'YdsPerAtt', 'YdsPerCmp', 'YdsPerG', 'Sacked'
        ]
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Clean text columns
        text_cols = ['Player', 'Team', 'Pos', 'QBrec', 'Awards']
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
                df[col] = df[col].replace('nan', '')
        
        # Remove rows with no player name
        df = df[df['Player'].str.len() > 0]
        df = df[df['Player'] != 'Player']
        
        # Add passing betting features
        print("➕ Adding passing metrics for betting...")
        
        # Core QB efficiency
        df['Completion_Percentage'] = df['CmpPct']
        df['Yards_per_Attempt'] = df['YdsPerAtt']
        df['Yards_per_Completion'] = df['YdsPerCmp']
        df['Yards_per_Game'] = df['YdsPerG']
        df['Passer_Rating'] = df['Rate']
        
        # Per game metrics
        df['Attempts_per_Game'] = df['Att'] / df['G'].replace(0, 1)
        df['Completions_per_Game'] = df['Cmp'] / df['G'].replace(0, 1)
        df['TD_per_Game'] = df['TD'] / df['G'].replace(0, 1)
        df['INT_per_Game'] = df['Int'] / df['G'].replace(0, 1)
        
        # Efficiency ratios
        df['TD_INT_Ratio'] = np.where(df['Int'] > 0, df['TD'] / df['Int'], df['TD'])
        df['Sack_Rate'] = np.where(df['Att'] > 0, df['Sacked'] / (df['Att'] + df['Sacked']), 0)
        df['TD_Rate'] = np.where(df['Att'] > 0, df['TD'] / df['Att'], 0)
        df['INT_Rate'] = np.where(df['Att'] > 0, df['Int'] / df['Att'], 0)
        
        # Volume tiers
        df['High_Volume_Passer'] = np.where(df['Att'] >= 400, 1, 0)
        df['Starter_Level'] = np.where(df['GS'] >= 12, 1, 0)
        df['Elite_Efficiency'] = np.where((df['Rate'] >= 100) & (df['YdsPerAtt'] >= 7.5), 1, 0)
        
        # QB classification
        df['QB_Type'] = 'Backup'
        df.loc[df['GS'] >= 8, 'QB_Type'] = 'Starter'
        df.loc[df['Att'] >= 500, 'QB_Type'] = 'Franchise_QB'
        df.loc[(df['Rate'] >= 110) & (df['Att'] >= 300), 'QB_Type'] = 'Elite_QB'
        df.loc[df['YdsPerAtt'] >= 8.5, 'QB_Type'] = 'Deep_Ball_QB'
        df.loc[df['CmpPct'] >= 70, 'QB_Type'] = 'Accurate_QB'
        
        # Betting props
        df['Pass_Yards_Target'] = df['YdsPerG'] * 17
        df['Pass_TD_Target'] = df['TD_per_Game'] * 17
        df['Pass_Attempts_Target'] = df['Attempts_per_Game'] * 17
        df['Pass_Completions_Target'] = df['Completions_per_Game'] * 17
        
        # Performance indicators
        df['Turnover_Prone'] = np.where(df['INT_Rate'] >= 0.03, 1, 0)
        df['Big_Arm'] = np.where(df['YdsPerCmp'] >= 12, 1, 0)
        df['Game_Manager'] = np.where((df['CmpPct'] >= 65) & (df['YdsPerAtt'] <= 7), 1, 0)
        df['Gunslinger'] = np.where((df['YdsPerAtt'] >= 8) & (df['INT_Rate'] >= 0.025), 1, 0)
        
        # Durability
        df['Games_Started_Pct'] = df['GS'] / df['G'].replace(0, 1)
        df['Injury_Concerns'] = np.where(df['G'] <= 12, 1, 0)
        
        # Add estimated year
        df['Year'] = 2000 + (df.index % 26)
        
        # Final cleanup
        df = df.replace([np.inf, -np.inf], 0)
        df = df.round(3)
        
        print(f"✅ Final dataset: {len(df):,} records")
        print(f"📊 Position breakdown:")
        print(df['Pos'].value_counts().head(10))
        print(f"🏈 QB type breakdown:")
        print(df['QB_Type'].value_counts())
        
        # Save cleaned data
        df.to_csv(output_file, index=False)
        print(f"💾 Saved clean data: {output_file}")
        
        # Show sample of cleaned data
        print(f"\n📋 Sample of cleaned passing data:")
        sample_cols = ['Player', 'Pos', 'G', 'Att', 'Yds', 'TD', 'Rate', 'QB_Type']
        available_cols = [col for col in sample_cols if col in df.columns]
        print(df[available_cols].head(10).to_string(index=False))
        
        return df
        
    except Exception as e:
        print(f"❌ Error cleaning passing stats: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    clean_passing_data()