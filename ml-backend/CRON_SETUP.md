"""
Render Cron Job Configuration
This script runs the weekly data refresh automatically on Render

Add this to render.yaml:
services:
  - type: cron
    name: weekly-data-refresh
    env: python
    schedule: "0 2 * * 2"  # Every Tuesday at 2 AM UTC
    buildCommand: "pip install -r requirements.txt"
    startCommand: "python ml-backend/weekly_data_refresh.py"
"""

# Or run manually with:
# python weekly_data_refresh.py

print("See render.yaml for cron job configuration")
print("Manual run: python weekly_data_refresh.py")
