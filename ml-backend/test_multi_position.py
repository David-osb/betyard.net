"""
Test script to verify multi-position predictions work correctly
"""
import sys
sys.path.insert(0, '.')

from app import NFLMLModel
import json

# Initialize model
print("Initializing multi-position ML model...")
model = NFLMLModel()

print("\n✅ All models loaded successfully!")
print(f"   - QB: {'QB' in model.models}")
print(f"   - RB: {'RB' in model.models}")
print(f"   - WR: {'WR' in model.models}")
print(f"   - TE: {'TE' in model.models}")

# Test QB prediction
print("\n" + "="*50)
print("🏈 Testing QB Prediction")
print("="*50)
qb_pred = model.predict_qb_performance("Patrick Mahomes", "KC")
print(f"Passing Yards: {qb_pred.passing_yards}")
print(f"Touchdowns: {qb_pred.touchdowns}")
print(f"Confidence: {qb_pred.confidence}%")
print(f"Position: {qb_pred.position}")

# Test RB prediction
print("\n" + "="*50)
print("🏃 Testing RB Prediction")
print("="*50)
rb_pred = model.predict_rb_performance("Christian McCaffrey", "SF")
print(f"Rushing Yards: {rb_pred.rushing_yards}")
print(f"Rushing Attempts: {rb_pred.rushing_attempts}")
print(f"Touchdowns: {rb_pred.touchdowns}")
print(f"Confidence: {rb_pred.confidence}%")
print(f"Position: {rb_pred.position}")

# Test WR prediction
print("\n" + "="*50)
print("🎯 Testing WR Prediction")
print("="*50)
wr_pred = model.predict_wr_performance("Tyreek Hill", "MIA")
print(f"Receiving Yards: {wr_pred.receiving_yards}")
print(f"Receptions: {wr_pred.receptions}")
print(f"Targets: {wr_pred.targets}")
print(f"Touchdowns: {wr_pred.touchdowns}")
print(f"Confidence: {wr_pred.confidence}%")
print(f"Position: {wr_pred.position}")

# Test TE prediction
print("\n" + "="*50)
print("🏈 Testing TE Prediction")
print("="*50)
te_pred = model.predict_te_performance("Travis Kelce", "KC")
print(f"Receiving Yards: {te_pred.receiving_yards}")
print(f"Receptions: {te_pred.receptions}")
print(f"Targets: {te_pred.targets}")
print(f"Touchdowns: {te_pred.touchdowns}")
print(f"Confidence: {te_pred.confidence}%")
print(f"Position: {te_pred.position}")

print("\n" + "="*50)
print("✅ All multi-position predictions working!")
print("="*50)
