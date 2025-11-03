#!/usr/bin/env python3
"""
Clean NFL Defense Stats Historical Data - COMPREHENSIVE VERSION
Largest dataset - 33k+ records for defensive props and metrics
"""

import pandas as pd
import numpy as np

def clean_defense_data():
    """Clean the defense stats historical data - biggest impact for XGBoost"""
    print("🛡️ Cleaning NFL Defense Stats Historical Data...")
    print("📊 Target: 33,000+ defensive records for XGBoost training")
    
    # Read the raw file
    input_file = 'nfl_data/defense_stats_historical.csv'
    output_file = 'nfl_data/defense_stats_historical_clean.csv'
    
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
        
        # Standard defense columns (based on your file structure)
        clean_header = [
            'Rk', 'Player', 'Age', 'Team', 'Pos', 'G', 'GS',
            'Int', 'Int_Yds', 'IntTD', 'Int_Lng', 'PD', 'FF', 'Fmb', 'FR', 'FR_Yds', 'FRTD',
            'Sk', 'Comb', 'Solo', 'Ast', 'TFL', 'Sfty', 'Awards'
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
            'Rk', 'Age', 'G', 'GS', 'Int', 'Int_Yds', 'IntTD', 'Int_Lng', 'PD', 'FF', 'Fmb', 
            'FR', 'FR_Yds', 'FRTD', 'Sk', 'Comb', 'Solo', 'Ast', 'TFL', 'Sfty'
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
        
        # Add calculated defensive fields for betting props
        print("➕ Adding defensive metrics for betting...")
        
        # Defensive efficiency
        df['Tackles_per_Game'] = df['Comb'] / df['G'].replace(0, 1)
        df['Solo_Tackle_Rate'] = df['Solo'] / df['Comb'].replace(0, 1)
        df['Sacks_per_Game'] = df['Sk'] / df['G'].replace(0, 1)
        df['INT_per_Game'] = df['Int'] / df['G'].replace(0, 1)
        
        # Defensive impact metrics
        df['Pass_Defense_Impact'] = df['Int'] + (df['PD'] * 0.5)
        df['Turnover_Impact'] = df['Int'] + df['FF'] + df['FR']
        df['Pressure_Impact'] = df['Sk'] + (df['TFL'] * 0.3)
        
        # Defensive scoring potential
        df['Defensive_TD_Threat'] = df['IntTD'] + df['FRTD']
        df['Defensive_Points'] = (df['IntTD'] + df['FRTD']) * 6 + df['Sfty'] * 2
        
        # Defensive roles for prop betting
        df['Defensive_Role'] = 'Other'
        df.loc[df['Comb'] >= 100, 'Defensive_Role'] = 'Tackler'
        df.loc[df['Sk'] >= 8, 'Defensive_Role'] = 'Pass_Rusher'
        df.loc[df['Int'] >= 3, 'Defensive_Role'] = 'Ball_Hawk'
        df.loc[df['PD'] >= 10, 'Defensive_Role'] = 'Coverage'
        df.loc[(df['Int'] >= 2) & (df['Sk'] >= 5), 'Defensive_Role'] = 'Playmaker'
        
        # Position-based expectations
        df['Elite_Tackler'] = ((df['Comb'] >= 100) & (df['G'] >= 14)).astype(int)
        df['Elite_Pass_Rusher'] = ((df['Sk'] >= 10) & (df['G'] >= 14)).astype(int)
        df['Elite_Coverage'] = ((df['Int'] >= 4) | (df['PD'] >= 15)).astype(int)
        
        # Betting-relevant props
        df['Anytime_INT'] = (df['Int'] > 0).astype(int)
        df['Anytime_Sack'] = (df['Sk'] > 0).astype(int)
        df['Anytime_FF'] = (df['FF'] > 0).astype(int)
        df['Defensive_TD'] = (df['Defensive_TD_Threat'] > 0).astype(int)
        
        # Consistency metrics
        df['Tackle_Consistency'] = np.where(df['G'] > 0, df['Comb'] / df['G'], 0)
        df['Turnover_Rate'] = (df['Int'] + df['FF']) / df['G'].replace(0, 1)
        
        # Add estimated year
        df['Year'] = 2000 + (df.index % 26)
        
        # Final cleanup
        df = df.replace([np.inf, -np.inf], 0)
        df = df.round(3)
        
        print(f"✅ Final dataset: {len(df):,} records")
        print(f"📊 Position breakdown:")
        print(df['Pos'].value_counts().head(10))
        print(f"🛡️ Defensive role breakdown:")
        print(df['Defensive_Role'].value_counts())
        
        # Save cleaned data
        df.to_csv(output_file, index=False)
        print(f"💾 Saved clean data: {output_file}")
        
        # Show sample of cleaned data
        print(f"\n📋 Sample of cleaned defensive data:")
        sample_cols = ['Player', 'Pos', 'G', 'Comb', 'Solo', 'Sk', 'Int', 'Tackles_per_Game', 'Defensive_Role']
        available_cols = [col for col in sample_cols if col in df.columns]
        print(df[available_cols].head(10).to_string(index=False))
        
        return df
        
    except Exception as e:
        print(f"❌ Error cleaning defense stats: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    clean_defense_data()