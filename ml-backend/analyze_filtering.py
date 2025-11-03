"""
Analyze filtering of NFL historical data
Show what records are being removed and why
"""

import pandas as pd
import numpy as np

def analyze_data_filtering():
    """Analyze step-by-step filtering of the historical dataset"""
    
    print("🔍 ANALYZING NFL HISTORICAL DATA FILTERING")
    print("="*60)
    
    # Load raw data
    try:
        df = pd.read_csv('nfl_data/qb_stats_historical.csv', skiprows=1)
        print(f"📊 RAW DATA LOADED: {len(df)} total records")
        
        # Remove summary rows
        df_clean = df[df['Player'] != 'League Average'].copy()
        print(f"📊 AFTER REMOVING SUMMARY: {len(df_clean)} records")
        
        # Analyze by position
        if 'Pos' in df_clean.columns:
            position_counts = df_clean['Pos'].value_counts()
            print(f"\n🏈 BREAKDOWN BY POSITION:")
            for pos, count in position_counts.head(10).items():
                print(f"   {pos}: {count} records")
            
            # Filter to QBs only
            qb_only = df_clean[df_clean['Pos'] == 'QB'].copy()
            print(f"\n🎯 QUARTERBACKS ONLY: {len(qb_only)} records")
        
        # Analyze by attempt count
        if 'Att' in df_clean.columns:
            # Clean the QB data and convert attempts to numeric
            qb_only['Att'] = pd.to_numeric(qb_only['Att'], errors='coerce')
            qb_only = qb_only.dropna(subset=['Att'])  # Remove rows with invalid attempt data
            
            print(f"\n📈 ATTEMPT DISTRIBUTION:")
            print(f"   Total QBs with valid attempt data: {len(qb_only)}")
            print(f"   QBs with 0 attempts: {len(qb_only[qb_only['Att'] == 0])}")
            print(f"   QBs with 1-9 attempts: {len(qb_only[(qb_only['Att'] >= 1) & (qb_only['Att'] <= 9)])}")
            print(f"   QBs with 10-24 attempts: {len(qb_only[(qb_only['Att'] >= 10) & (qb_only['Att'] < 25)])}")
            print(f"   QBs with 25+ attempts: {len(qb_only[qb_only['Att'] >= 25])}")
            
            # Show examples of filtered out players
            print(f"\n❌ EXAMPLES OF FILTERED OUT QBs (< 25 attempts):")
            low_attempts = qb_only[(qb_only['Att'] < 25) & (qb_only['Att'] > 0)].copy()
            if len(low_attempts) > 0:
                sample_cols = ['Player', 'Age', 'Team', 'Att', 'Cmp'] if all(col in low_attempts.columns for col in ['Player', 'Age', 'Team', 'Att', 'Cmp']) else low_attempts.columns[:5]
                print(low_attempts[sample_cols].head(10).to_string(index=False))
            
            # Show examples of qualifying players
            print(f"\n✅ EXAMPLES OF QUALIFYING QBs (25+ attempts):")
            qualifying = qb_only[qb_only['Att'] >= 25].copy()
            if len(qualifying) > 0:
                sample_cols = ['Player', 'Age', 'Team', 'Att', 'Cmp'] if all(col in qualifying.columns for col in ['Player', 'Age', 'Team', 'Att', 'Cmp']) else qualifying.columns[:5]
                print(qualifying[sample_cols].head(10).to_string(index=False))
        
        # Summary of filtering
        print(f"\n📋 FILTERING SUMMARY:")
        print(f"   🔸 Total records: {len(df)}")
        print(f"   🔸 After cleaning: {len(df_clean)}")
        print(f"   🔸 QBs only: {len(qb_only)}")
        if 'Att' in df_clean.columns:
            qualifying_final = len(qb_only[qb_only['Att'] >= 25])
            print(f"   🔸 QBs with 25+ attempts: {qualifying_final}")
            print(f"   🔸 Records filtered out: {len(df) - qualifying_final}")
            
            # Breakdown of what was filtered
            non_qb = len(df_clean) - len(qb_only)
            low_volume = len(qb_only) - qualifying_final
            print(f"\n🗂️  WHAT WAS FILTERED OUT:")
            print(f"   📌 Non-QBs (RB, WR, TE, P, etc.): {non_qb}")
            print(f"   📌 QBs with < 25 attempts: {low_volume}")
            print(f"   📌 Summary/invalid rows: {len(df) - len(df_clean)}")
        
        return df_clean, qb_only
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None, None

if __name__ == "__main__":
    df_clean, qb_only = analyze_data_filtering()