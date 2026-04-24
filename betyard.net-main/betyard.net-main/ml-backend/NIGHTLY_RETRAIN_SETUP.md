# Nightly NBA Retrain Setup

## Script

Run this script nightly:

- `ml-backend/nightly_nba_retrain.py`

It will:
1. Refresh NBA player data (`fetch_nba_stats.py`)
2. Retrain ensemble NBA prop models (`train_nba_ensemble_models.py`)
3. Write a timestamped log file to `ml-backend/logs/`

## Cloud nightly run (works when your PC is off)

GitHub Actions workflow added:

- `.github/workflows/nightly-nba-retrain.yml`

Schedule:

- `0 2 * * *` (2:00 AM UTC daily)

It runs retraining in GitHub-hosted runners, uploads run artifacts, and commits updated NBA model/data files back to the repo when changes are detected.

## Manual run

From `ml-backend`:

```powershell
c:/Coding/.venv/Scripts/python.exe nightly_nba_retrain.py
```

## Windows Task Scheduler (recommended on your machine)

### One-click setup (recommended)

Run this once in PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File c:\Coding\betyard.net-main\betyard.net-main\ml-backend\create_nightly_retrain_task.ps1 -Hour 2 -Minute 0 -Force
```

Optional flags:

- `-SkipFetch` to keep retraining without fetching new NBA data
- `-RunNow` to start the task immediately after creating it

Example:

```powershell
powershell -ExecutionPolicy Bypass -File c:\Coding\betyard.net-main\betyard.net-main\ml-backend\create_nightly_retrain_task.ps1 -Hour 2 -Minute 0 -SkipFetch -Force -RunNow
```

### Manual setup

1. Open Task Scheduler
2. Create Task
3. General:
   - Name: `BetYard NBA Nightly Retrain`
4. Triggers:
   - Daily, every 1 day, set your preferred time (example: 2:00 AM)
5. Actions:
   - Program/script:
     - `c:\Coding\.venv\Scripts\python.exe`
   - Add arguments:
     - `c:\Coding\betyard.net-main\betyard.net-main\ml-backend\nightly_nba_retrain.py`
   - Start in:
     - `c:\Coding\betyard.net-main\betyard.net-main\ml-backend`
6. Settings:
   - Enable "Run task as soon as possible after a scheduled start is missed"

## Optional environment flag

If you already refresh data separately, skip fetch step:

- Set env var `NBA_SKIP_FETCH=1` for the scheduled task.

## Success artifacts

After a successful run, verify:

- `ml-backend/nba_models/*.joblib`
- `ml-backend/nba_ensemble_metrics.json`
- `ml-backend/nba_prop_predictions_ml.json`
- `ml-backend/logs/nightly_nba_retrain_*.log`
