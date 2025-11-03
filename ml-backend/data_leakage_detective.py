#!/usr/bin/env python3
"""
DATA LEAKAGE DETECTIVE - Find Why We Have 100% Accuracy
=======================================================

Investigating potential data leakage issues in our betting model:
1. Check target correlations with features  
2. Examine feature importance
3. Look for circular dependencies
4. Test on truly unseen data
"""

import pandas as pd
import numpy as np
import glob
import warnings
warnings.filterwarnings('ignore')

def investigate_data_leakage():
    print("🕵️ DATA LEAKAGE INVESTIGATION")
    print("=" * 50)
    
    # Load the same data the model used
    data_files = glob.glob('nfl_data/*_clean.csv')
    print(f"Loading {len(data_files)} data files...")
    
    all_dataframes = []
    for file_path in data_files:
        try:
            df = pd.read_csv(file_path, low_memory=False)
            file_name = file_path.split('\\')[-1]
            
            if 'passing' in file_name:
                df['nfl_category'] = 'passing'
            elif 'rushing' in file_name:
                df['nfl_category'] = 'rushing'
            elif 'receiving' in file_name:
                df['nfl_category'] = 'receiving'
            elif 'defense' in file_name:
                df['nfl_category'] = 'defense'
            elif 'scoring' in file_name:
                df['nfl_category'] = 'scoring'
            elif 'returns' in file_name:
                df['nfl_category'] = 'returns'
            elif 'scrimmage' in file_name:
                df['nfl_category'] = 'scrimmage'
            
            all_dataframes.append(df)
            print(f"  {file_name}: {len(df)} records")
        except Exception as e:
            print(f"  Error loading {file_path}: {e}")
    
    # Combine data
    df = pd.concat(all_dataframes, ignore_index=True, sort=False)
    print(f"\nTotal records: {len(df):,}")
    
    # Create the same targets the model used
    print("\n🎯 ANALYZING TARGETS...")
    
    # Fill missing age values
    if 'Age' in df.columns:
        df['Age'] = df['Age'].fillna(df['Age'].median())
    
    if 'G' in df.columns:
        df['G'] = df['G'].fillna(1)
    
    # Create targets exactly like the model
    targets = {}
    
    # Prime performer target
    if 'Age' in df.columns:
        targets['prime_performer'] = ((df['Age'] >= 25) & (df['Age'] <= 29)).astype(int)
    
    # Veteran reliability target  
    if 'Age' in df.columns:
        targets['veteran_reliability'] = ((df['Age'] >= 28) & (df['Age'] <= 32)).astype(int)
    
    # High volume target
    if 'G' in df.columns:
        game_threshold = df['G'].quantile(0.65)
        targets['high_volume'] = (df['G'] >= game_threshold).astype(int)
    
    # Analyze each target
    for target_name, target_values in targets.items():
        print(f"\n📊 TARGET: {target_name}")
        print(f"   Hit rate: {target_values.mean():.1%}")
        print(f"   Total positives: {target_values.sum():,}")
        print(f"   Total negatives: {(1-target_values).sum():,}")
        
        # Check if target is perfectly predictable from Age/G
        if target_name == 'prime_performer' and 'Age' in df.columns:
            # This target is PURELY based on Age - that's the problem!
            print(f"   🚨 PROBLEM: Target is purely Age >= 25 & Age <= 29")
            print(f"   🚨 If Age is a feature, this is 100% predictable!")
            
        elif target_name == 'veteran_reliability' and 'Age' in df.columns:
            print(f"   🚨 PROBLEM: Target is purely Age >= 28 & Age <= 32") 
            print(f"   🚨 If Age is a feature, this is 100% predictable!")
            
        elif target_name == 'high_volume' and 'G' in df.columns:
            print(f"   🚨 PROBLEM: Target is purely G >= {game_threshold:.1f}")
            print(f"   🚨 If G (Games) is a feature, this is 100% predictable!")
    
    # Check feature correlations
    print(f"\n🔍 CHECKING FEATURE CORRELATIONS...")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    features = df[numeric_cols].copy()
    
    # Check if Age and G are in features
    problematic_features = []
    if 'Age' in features.columns:
        problematic_features.append('Age')
        print(f"   🚨 Age is in features - causes perfect prediction for age-based targets!")
        
    if 'G' in features.columns:
        problematic_features.append('G')
        print(f"   🚨 G (Games) is in features - causes perfect prediction for game-based targets!")
    
    # Test correlation
    for target_name, target_values in targets.items():
        df[target_name] = target_values
        
        if target_name in ['prime_performer', 'veteran_reliability'] and 'Age' in features.columns:
            correlation = df['Age'].corr(df[target_name])
            print(f"   Age correlation with {target_name}: {correlation:.3f}")
            
        if target_name == 'high_volume' and 'G' in features.columns:
            correlation = df['G'].corr(df[target_name])
            print(f"   G correlation with {target_name}: {correlation:.3f}")
    
    print(f"\n💡 SOLUTION:")
    print(f"   1. Remove Age from features when predicting age-based targets")
    print(f"   2. Remove G from features when predicting game-based targets") 
    print(f"   3. Create targets based on PERFORMANCE, not demographics")
    print(f"   4. Use lagged/historical features instead of current period features")
    
    return df, targets, problematic_features

if __name__ == "__main__":
    df, targets, problems = investigate_data_leakage()
    
    print(f"\n🎯 SUMMARY:")
    print(f"   Problematic features found: {problems}")
    print(f"   Root cause: Targets are perfectly predictable from input features")
    print(f"   This is classic data leakage - not real predictive power!")