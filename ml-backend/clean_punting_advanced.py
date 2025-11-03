#!/usr/bin/env python3
"""
Advanced cleaning of punting stats data with proper year assignment
"""
import pandas as pd
import numpy as np

def clean_punting_stats_advanced():
    """Clean punting stats with proper year detection"""
    
    print("Advanced cleaning of punting stats...")
    
    input_file = "nfl_data/punting_stats_historical.csv"
    output_file = "nfl_data/punting_stats_historical.csv"
    
    # Read raw data
    clean_data = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Process each line
    current_year = 2025  # Start from most recent
    year_blocks = []
    current_block = []
    
    for line in lines:
        line = line.strip()
        
        # Skip malformed lines
        if not line or "Punting,Punting" in line:
            continue
            
        # Detect new year blocks by header appearance
        if line.startswith("Rk,Player,Age,Team,Pos"):
            if current_block:
                year_blocks.append((current_year, current_block))
                current_block = []
                current_year -= 1
            continue
        
        # Skip League Average rows
        if "League Average" in line:
            continue
            
        # Collect data rows
        if line and not line.startswith(","):
            parts = line.split(',')
            if len(parts) >= 19 and parts[1]:  # Must have player name
                # Take first 19 columns (Rk through Awards)
                clean_row = parts[:19]
                current_block.append(clean_row)
    
    # Don't forget the last block
    if current_block:
        year_blocks.append((current_year, current_block))
    
    # Build final dataset
    all_data = []
    header = ["Rk", "Player", "Age", "Team", "Pos", "G", "GS", "Pnt", "Yds", "Y/P", 
              "RetYds", "NetYds", "NY/P", "Lng", "TB", "TB%", "Pnt20", "In20%", "Blck", "Awards", "Year"]
    
    for year, block_data in year_blocks:
        for row in block_data:
            # Add year as last column
            row_with_year = row + [str(year)]
            all_data.append(row_with_year)
    
    # Create DataFrame for validation and final cleaning
    df = pd.DataFrame(all_data, columns=header)
    
    # Clean up data types and remove any remaining artifacts
    df = df[df['Player'].str.strip() != '']  # Remove empty player names
    df = df[~df['Player'].str.contains('League Average', na=False)]  # Remove any missed league averages
    
    # Clean numeric columns
    numeric_cols = ['Age', 'G', 'GS', 'Pnt', 'Yds', 'Y/P', 'RetYds', 'NetYds', 'NY/P', 
                   'Lng', 'TB', 'TB%', 'Pnt20', 'In20%', 'Blck', 'Year']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove rows with too many missing values
    df = df.dropna(subset=['Player', 'Team', 'Year'])
    
    # Sort by year (descending) and player name
    df = df.sort_values(['Year', 'Player'], ascending=[False, True])
    
    # Reset index
    df = df.reset_index(drop=True)
    
    # Save cleaned data
    df.to_csv(output_file, index=False)
    
    print(f"✅ Advanced cleaning completed!")
    print(f"   Total records: {len(df)}")
    print(f"   Years covered: {df['Year'].min():.0f} - {df['Year'].max():.0f}")
    print(f"   Unique players: {df['Player'].nunique()}")
    print(f"   Years of data: {df['Year'].nunique()}")
    
    # Show sample of cleaned data
    print(f"\n📊 Sample of cleaned data:")
    sample_df = df.head(5)[['Player', 'Age', 'Team', 'Pos', 'Pnt', 'Yds', 'Y/P', 'Year']]
    print(sample_df.to_string(index=False))
    
    # Show year distribution
    print(f"\n📅 Data distribution by year:")
    year_counts = df['Year'].value_counts().sort_index(ascending=False)
    for year, count in year_counts.head(10).items():
        print(f"   {year:.0f}: {count} players")
    
    return df

if __name__ == "__main__":
    clean_punting_stats_advanced()