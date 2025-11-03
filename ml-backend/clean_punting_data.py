#!/usr/bin/env python3
"""
Clean punting stats data by removing duplicate headers and formatting issues
"""
import pandas as pd
import re

def clean_punting_stats():
    """Clean the punting stats historical data"""
    
    print("Cleaning punting stats data...")
    
    input_file = "nfl_data/punting_stats_historical.csv"
    output_file = "nfl_data/punting_stats_historical.csv"
    
    # Read the raw file and process line by line
    clean_lines = []
    header_found = False
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Define the proper header
    proper_header = "Rk,Player,Age,Team,Pos,G,GS,Pnt,Yds,Y/P,RetYds,NetYds,NY/P,Lng,TB,TB%,Pnt20,In20%,Blck,Awards,Year"
    
    current_year = 2025  # Start from most recent
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
        
        # Skip the malformed first line with repeated "Punting"
        if "Punting,Punting" in line:
            continue
            
        # Skip duplicate headers (keep only the first one)
        if line.startswith("Rk,Player,Age,Team,Pos"):
            if not header_found:
                clean_lines.append(proper_header)
                header_found = True
                # When we see a new header, it usually means a new year
                if len(clean_lines) > 1:  # Don't decrement for the very first header
                    current_year -= 1
            continue
        
        # Skip League Average rows
        if "League Average" in line:
            continue
            
        # Clean up data rows
        if line and not line.startswith("Rk,") and not line.startswith(","):
            # Remove the extra columns (-9999 and other artifacts)
            parts = line.split(',')
            
            # Take only the first 19 columns (Rk through Awards)
            if len(parts) >= 19:
                clean_parts = parts[:19]
                
                # Add year column
                clean_parts.append(str(current_year))
                
                # Clean up any empty or malformed entries
                cleaned_line = ','.join(clean_parts)
                
                # Only add if it looks like valid data (has player name)
                if len(clean_parts) > 1 and clean_parts[1] and clean_parts[1] != '':
                    clean_lines.append(cleaned_line)
    
    # Write cleaned data
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in clean_lines:
            f.write(line + '\n')
    
    print(f"✅ Cleaned punting stats saved to {output_file}")
    print(f"   Total data rows: {len(clean_lines) - 1}")  # -1 for header
    
    # Validate the cleaned data
    try:
        df = pd.read_csv(output_file)
        print(f"   Years covered: {df['Year'].min()} - {df['Year'].max()}")
        print(f"   Total players: {df['Player'].nunique()}")
        print(f"   Columns: {len(df.columns)}")
        print("   Sample of cleaned data:")
        print(df.head(3).to_string(index=False))
        
    except Exception as e:
        print(f"⚠️ Error validating cleaned data: {e}")

if __name__ == "__main__":
    clean_punting_stats()