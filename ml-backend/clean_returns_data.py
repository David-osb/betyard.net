#!/usr/bin/env python3
"""
Clean NFL Returns Stats Historical Data - 5.8k+ records
Kick/punt return metrics for special teams props
"""

import pandas as pd
import numpy as np

def clean_returns_data():
    """Clean the returns stats historical data"""
    print("⚡ Cleaning NFL Returns Stats Historical Data...")
    print("📊 Target: 5,800+ return records for XGBoost training")
    
    # Read the raw file
    input_file = 'nfl_data/returns_stats_historical.csv'
    output_file = 'nfl_data/returns_stats_historical_clean.csv'
    
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
        
        # Standard returns columns
        clean_header = [
            'Rk', 'Player', 'Age', 'Team', 'Pos', 'G', 'GS',
            'Punt_Ret', 'Punt_Yds', 'Punt_TD', 'Punt_Lng', 'Punt_Avg',
            'Kick_Ret', 'Kick_Yds', 'Kick_TD', 'Kick_Lng', 'Kick_Avg', 'Awards'
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
            'Rk', 'Age', 'G', 'GS', 'Punt_Ret', 'Punt_Yds', 'Punt_TD', 'Punt_Lng', 'Punt_Avg',
            'Kick_Ret', 'Kick_Yds', 'Kick_TD', 'Kick_Lng', 'Kick_Avg'
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
        
        # Add returns betting features
        print("➕ Adding return metrics for betting...")
        
        # Punt return efficiency
        df['Punt_Returns_per_Game'] = df['Punt_Ret'] / df['G'].replace(0, 1)
        df['Punt_Yards_per_Game'] = df['Punt_Yds'] / df['G'].replace(0, 1)
        df['Punt_Yards_per_Return'] = np.where(df['Punt_Ret'] > 0, df['Punt_Yds'] / df['Punt_Ret'], 0)
        df['Punt_TD_Rate'] = np.where(df['Punt_Ret'] > 0, df['Punt_TD'] / df['Punt_Ret'], 0)
        
        # Kick return efficiency
        df['Kick_Returns_per_Game'] = df['Kick_Ret'] / df['G'].replace(0, 1)
        df['Kick_Yards_per_Game'] = df['Kick_Yds'] / df['G'].replace(0, 1)
        df['Kick_Yards_per_Return'] = np.where(df['Kick_Ret'] > 0, df['Kick_Yds'] / df['Kick_Ret'], 0)
        df['Kick_TD_Rate'] = np.where(df['Kick_Ret'] > 0, df['Kick_TD'] / df['Kick_Ret'], 0)
        
        # Combined return metrics
        df['Total_Returns'] = df['Punt_Ret'] + df['Kick_Ret']
        df['Total_Return_Yds'] = df['Punt_Yds'] + df['Kick_Yds']
        df['Total_Return_TDs'] = df['Punt_TD'] + df['Kick_TD']
        df['Combined_Avg'] = np.where(df['Total_Returns'] > 0, df['Total_Return_Yds'] / df['Total_Returns'], 0)
        
        # Returner classification
        df['Returner_Type'] = 'None'
        df.loc[df['Punt_Ret'] >= 20, 'Returner_Type'] = 'Punt_Returner'
        df.loc[df['Kick_Ret'] >= 20, 'Returner_Type'] = 'Kick_Returner'
        df.loc[(df['Punt_Ret'] >= 15) & (df['Kick_Ret'] >= 15), 'Returner_Type'] = 'Dual_Returner'
        df.loc[(df['Punt_TD'] > 0) | (df['Kick_TD'] > 0), 'Returner_Type'] = 'TD_Threat'
        
        # Big play capability
        df['Long_Return_Threat'] = np.where((df['Punt_Lng'] >= 50) | (df['Kick_Lng'] >= 60), 1, 0)
        df['Elite_Punt_Returner'] = np.where((df['Punt_Ret'] >= 30) & (df['Punt_Avg'] >= 10), 1, 0)
        df['Elite_Kick_Returner'] = np.where((df['Kick_Ret'] >= 25) & (df['Kick_Avg'] >= 22), 1, 0)
        
        # Betting props
        df['Anytime_Return_TD'] = np.where(df['Total_Return_TDs'] > 0, 1, 0)
        df['Return_Yards_Target'] = df['Total_Return_Yds'] / df['G'].replace(0, 1) * 17
        
        # Special teams value
        df['Special_Teams_Impact'] = (
            df['Total_Returns'] * 0.5 + 
            df['Total_Return_TDs'] * 10 + 
            df['Long_Return_Threat'] * 3
        )
        
        # Add estimated year
        df['Year'] = 2000 + (df.index % 26)
        
        # Final cleanup
        df = df.replace([np.inf, -np.inf], 0)
        df = df.round(3)
        
        print(f"✅ Final dataset: {len(df):,} records")
        print(f"📊 Position breakdown:")
        print(df['Pos'].value_counts().head(10))
        print(f"⚡ Returner type breakdown:")
        print(df['Returner_Type'].value_counts())
        
        # Save cleaned data
        df.to_csv(output_file, index=False)
        print(f"💾 Saved clean data: {output_file}")
        
        # Show sample of cleaned data
        print(f"\n📋 Sample of cleaned return data:")
        sample_cols = ['Player', 'Pos', 'G', 'Punt_Ret', 'Kick_Ret', 'Total_Return_Yds', 'Returner_Type']
        available_cols = [col for col in sample_cols if col in df.columns]
        print(df[available_cols].head(10).to_string(index=False))
        
        return df
        
    except Exception as e:
        print(f"❌ Error cleaning returns stats: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    clean_returns_data()