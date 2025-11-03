#!/usr/bin/env python3
"""
Fix defense data to match the correct header structure
"""

import pandas as pd
import numpy as np

def fix_defense_data():
    """Fix defense data to match the new header structure"""
    
    # Define the correct headers
    correct_headers = [
        'Rk', 'Player', 'Age', 'Team', 'Pos', 'G', 'GS',
        'Int', 'Yds', 'IntTD', 'Lng', 'PD', 'FF', 'Fmb', 'FR', 'Yds', 'FRTD',
        'Sk', 'Comb', 'Solo', 'Ast', 'TFL', 'Sfty', 'Awards'
    ]
    
    print("Reading defense_stats_historical.csv...")
    
    # Read the file, skip the current header
    df = pd.read_csv('nfl_data/defense_stats_historical.csv', header=0)
    
    print(f"Current data shape: {df.shape}")
    print(f"Current columns: {len(df.columns)}")
    print(f"Expected columns: {len(correct_headers)}")
    
    # If we have more columns than expected, trim to the correct number
    if len(df.columns) > len(correct_headers):
        print(f"Trimming from {len(df.columns)} to {len(correct_headers)} columns")
        df = df.iloc[:, :len(correct_headers)]
    elif len(df.columns) < len(correct_headers):
        print(f"Adding missing columns (have {len(df.columns)}, need {len(correct_headers)})")
        # Add missing columns with default values
        for i in range(len(df.columns), len(correct_headers)):
            df[f'col_{i}'] = 0
    
    # Assign correct column names
    df.columns = correct_headers
    
    print(f"Fixed data shape: {df.shape}")
    print(f"New columns: {list(df.columns)}")
    
    # Save the fixed data
    output_file = 'nfl_data/defense_stats_historical.csv'
    df.to_csv(output_file, index=False)
    print(f"\nFixed data saved to {output_file}")
    
    # Show sample
    print("\nSample of fixed data:")
    print(df.head())
    
    return df

if __name__ == "__main__":
    fix_defense_data()