"""
Weekly Data Refresh Script
Automatically updates all NFL data every Tuesday
- Fetches latest ESPN player game logs
- Recalculates team ratings
- Updates for current week

Run this script via cron job: 0 2 * * 2 (Every Tuesday at 2 AM)
Or manually: python weekly_data_refresh.py
"""

import subprocess
import sys
import os
from datetime import datetime
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_command(command, description):
    """Run a shell command and log results"""
    logger.info(f"Starting: {description}")
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            logger.info(f"✅ SUCCESS: {description}")
            if result.stdout:
                logger.info(result.stdout)
            return True
        else:
            logger.error(f"❌ FAILED: {description}")
            logger.error(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"⏱️ TIMEOUT: {description} took too long")
        return False
    except Exception as e:
        logger.error(f"❌ ERROR in {description}: {e}")
        return False


def weekly_refresh():
    """Run all weekly data refresh tasks"""
    
    logger.info("=" * 60)
    logger.info("WEEKLY NFL DATA REFRESH")
    logger.info(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    tasks = [
        {
            'command': f'python "{os.path.join(script_dir, "fetch_espn_stats.py")}"',
            'description': 'Fetch latest ESPN player game logs',
            'required': True
        },
        {
            'command': f'python "{os.path.join(script_dir, "calculate_team_ratings.py")}"',
            'description': 'Recalculate team offensive/defensive ratings',
            'required': True
        }
    ]
    
    results = []
    
    for task in tasks:
        success = run_command(task['command'], task['description'])
        results.append({
            'task': task['description'],
            'success': success,
            'required': task.get('required', False)
        })
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("REFRESH SUMMARY")
    logger.info("=" * 60)
    
    total = len(results)
    succeeded = sum(1 for r in results if r['success'])
    failed = total - succeeded
    
    for result in results:
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        logger.info(f"{status} - {result['task']}")
    
    logger.info(f"\nTotal: {succeeded}/{total} tasks succeeded")
    
    # Check if any required tasks failed
    required_failures = [r for r in results if not r['success'] and r['required']]
    
    if required_failures:
        logger.error(f"\n❌ {len(required_failures)} REQUIRED TASKS FAILED!")
        logger.error("Data may be incomplete or stale")
        return False
    else:
        logger.info("\n✅ ALL REQUIRED TASKS COMPLETED SUCCESSFULLY")
        logger.info("NFL data is up to date!")
        return True


if __name__ == '__main__':
    success = weekly_refresh()
    sys.exit(0 if success else 1)
