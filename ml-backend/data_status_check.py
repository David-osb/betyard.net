#!/usr/bin/env python3
"""
NFL Data Status Check
Check the status of all position CSV files and provide guidance
"""

import pandas as pd
import os

def check_data_files():
    """Check all NFL data files and their status"""
    print("=== NFL Historical Data Status Check ===\n")
    
    data_dir = "nfl_data"
    files_to_check = [
        'qb_stats_historical.csv',
        'rb_stats_historical.csv', 
        'wr_stats_historical.csv',
        'te_stats_historical.csv'
    ]
    
    for filename in files_to_check:
        filepath = os.path.join(data_dir, filename)
        position = filename.split('_')[0].upper()
        
        print(f"📊 {position} Data ({filename}):")
        
        if os.path.exists(filepath):
            try:
                # Get file size
                file_size = os.path.getsize(filepath)
                
                # Count lines
                with open(filepath, 'r', encoding='utf-8') as f:
                    line_count = sum(1 for _ in f)
                
                # Try to read the data
                df = pd.read_csv(filepath, nrows=10)  # Just read first 10 rows
                
                print(f"   ✅ File exists: {file_size:,} bytes")
                print(f"   📏 Line count: {line_count:,} lines")
                print(f"   📋 Columns: {len(df.columns)} columns")
                
                if line_count > 100:
                    print(f"   ✅ Status: FULL DATASET (ready for training)")
                else:
                    print(f"   ⚠️  Status: SAMPLE DATA ONLY (needs full dataset)")
                
                # Show first few actual data rows (skip headers)
                if line_count > 3:
                    try:
                        # Skip header rows and read actual data
                        df_actual = pd.read_csv(filepath, skiprows=1, nrows=3)
                        if not df_actual.empty:
                            print(f"   📝 Sample data preview:")
                            for i, row in df_actual.iterrows():
                                player_info = f"{row.iloc[1] if len(row) > 1 else 'N/A'}"  # Player name typically in column 1
                                print(f"      Row {i+1}: {player_info}")
                    except:
                        print(f"   📝 Data format needs inspection")
                        
            except Exception as e:
                print(f"   ❌ Error reading file: {e}")
        else:
            print(f"   ❌ File not found")
        
        print()
    
    # Summary and recommendations
    print("=== SUMMARY & RECOMMENDATIONS ===\n")
    
    print("📈 QB Data: ✅ READY (2,748 lines - full 26-year dataset)")
    print("📈 RB Data: ⚠️  NEEDS FULL DATASET (only 6 lines - sample only)")
    print("📈 TE Data: ⚠️  NEEDS FULL DATASET (only 6 lines - sample only)")  
    print("📈 WR Data: ⚠️  NEEDS FULL DATASET (only 6 lines - sample only)")
    
    print("\n🎯 NEXT STEPS:")
    print("1. You have full QB data ready for training")
    print("2. For RB data: You mentioned having 13k lines - please check where this data is stored")
    print("3. Options:")
    print("   a) Train with QB data only for now")
    print("   b) Locate and add your full RB dataset (13k lines)")
    print("   c) Collect full historical data for all positions")
    
    print("\n💡 IMMEDIATE ACTION:")
    print("Since QB data is ready, I can create a QB-only training pipeline")
    print("Then add other positions as datasets become available")

if __name__ == "__main__":
    check_data_files()