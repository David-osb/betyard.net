#!/usr/bin/env python3
"""
Simple and robust punting stats cleaner
"""
import csv

def clean_punting_stats_simple():
    """Clean punting stats with simple approach"""
    
    print("Cleaning punting stats (simple approach)...")
    
    input_file = "nfl_data/punting_stats_historical.csv"
    output_file = "nfl_data/punting_stats_historical.csv"
    
    clean_rows = []
    header = ["Rk", "Player", "Age", "Team", "Pos", "G", "GS", "Pnt", "Yds", "Y/P", 
              "RetYds", "NetYds", "NY/P", "Lng", "TB", "TB%", "Pnt20", "In20%", "Blck", "Awards", "Year"]
    
    current_year = 2025
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines and malformed headers
        if not line or "Punting,Punting" in line:
            continue
            
        # Detect new year sections by header
        if line.startswith("Rk,Player,Age,Team,Pos"):
            if clean_rows:  # Not the first header
                current_year -= 1
            continue
        
        # Skip League Average
        if "League Average" in line:
            continue
            
        # Process data rows
        if line and not line.startswith(","):
            parts = line.split(',')
            
            # Must have at least player name and basic stats
            if len(parts) >= 10 and parts[1].strip():
                # Take exactly what we need (first 19 + year)
                data_row = []
                
                # Handle each column carefully
                for i in range(19):
                    if i < len(parts):
                        value = parts[i].strip()
                        # Clean up artifacts
                        if value in ['-9999', '-additional', '']:
                            value = ''
                        data_row.append(value)
                    else:
                        data_row.append('')
                
                # Add year
                data_row.append(str(current_year))
                
                # Only add rows with valid player names
                if data_row[1]:  # Player name exists
                    clean_rows.append(data_row)
    
    # Write cleaned data
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(clean_rows)
    
    print(f"✅ Cleaning completed!")
    print(f"   Total records: {len(clean_rows)}")
    
    # Show sample
    print(f"\n📊 First 5 cleaned records:")
    for i, row in enumerate(clean_rows[:5]):
        player_info = f"{row[1]} ({row[3]}, {row[20]})"
        stats = f"Punts: {row[7]}, Yards: {row[8]}, Avg: {row[9]}"
        print(f"   {i+1}. {player_info} - {stats}")
    
    # Count years
    years = {}
    for row in clean_rows:
        year = row[20]
        years[year] = years.get(year, 0) + 1
    
    print(f"\n📅 Year distribution:")
    for year in sorted(years.keys(), reverse=True)[:10]:
        print(f"   {year}: {years[year]} players")

if __name__ == "__main__":
    clean_punting_stats_simple()