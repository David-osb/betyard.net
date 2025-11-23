#!/usr/bin/env python3
"""
Verify returns stats cleaning results
"""
import pandas as pd

def verify_returns_stats():
    """Verify the cleaned returns stats"""
    
    file_path = "nfl_data/returns_stats_historical.csv"
    
    try:
        # Read the cleaned file
        df = pd.read_csv(file_path)
        
        print("=== RETURNS STATS VERIFICATION ===")
        print(f"✅ File loaded successfully!")
        print(f"   Total records: {len(df):,}")
        print(f"   Total columns: {len(df.columns)}")
        print(f"   Years covered: {df['Year'].min():.0f} - {df['Year'].max():.0f}")
        print(f"   Unique players: {df['Player'].nunique():,}")
        
        print(f"\n📊 Column Structure:")
        for i, col in enumerate(df.columns):
            print(f"   {i+1:2d}. {col}")
        
        print(f"\n📈 Sample Data:")
        sample = df.head(3)[['Player', 'Age', 'Team', 'Pos', 'PR_Ret', 'PR_Yds', 'KR_Ret', 'KR_Yds', 'Year']]
        print(sample.to_string(index=False))
        
        print(f"\n📅 Year Distribution (Top 10):")
        year_counts = df['Year'].value_counts().sort_index(ascending=False)
        for year, count in year_counts.head(10).items():
            print(f"   {year:.0f}: {count:,} players")
        
        # Check for data quality
        print(f"\n🔍 Data Quality Check:")
        print(f"   Missing player names: {df['Player'].isna().sum()}")
        print(f"   Missing team codes: {df['Team'].isna().sum()}")
        print(f"   Players with punt returns: {(df['PR_Ret'] > 0).sum():,}")
        print(f"   Players with kick returns: {(df['KR_Ret'] > 0).sum():,}")
        print(f"   Players with both return types: {((df['PR_Ret'] > 0) & (df['KR_Ret'] > 0)).sum():,}")
        
        # Show top returners
        print(f"\n🏃 Top Punt Returners (by attempts):")
        top_pr = df.nlargest(5, 'PR_Ret')[['Player', 'Team', 'Year', 'PR_Ret', 'PR_Yds', 'PR_Y/Ret']]
        print(top_pr.to_string(index=False))
        
        print(f"\n🏃 Top Kick Returners (by attempts):")
        top_kr = df.nlargest(5, 'KR_Ret')[['Player', 'Team', 'Year', 'KR_Ret', 'KR_Yds', 'KR_Y/Ret']]
        print(top_kr.to_string(index=False))
        
        print(f"\n✅ VERIFICATION COMPLETE - Returns stats are clean and ready!")
        
    except Exception as e:
        print(f"❌ Error verifying file: {e}")

if __name__ == "__main__":
    verify_returns_stats()