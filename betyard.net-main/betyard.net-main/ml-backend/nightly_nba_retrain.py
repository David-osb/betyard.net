"""
Nightly NBA retrain runner.

What it does:
1. Refreshes NBA player data
2. Trains ensemble NBA prop models
3. Emits a timestamped run log in ml-backend/logs

Exit code is non-zero if any required step fails.
"""

import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent
LOG_DIR = ROOT / "logs"


def build_logger() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = LOG_DIR / f"nightly_nba_retrain_{ts}.log"

    logger = logging.getLogger("nightly_nba_retrain")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    fh = logging.FileHandler(logfile, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)

    logger.info("Log file: %s", logfile)
    return logger


def run_step(logger: logging.Logger, script_name: str, description: str, timeout_sec: int = 1800) -> bool:
    script_path = ROOT / script_name
    logger.info("Starting: %s", description)

    if not script_path.exists():
        logger.error("Missing script: %s", script_path)
        return False

    cmd = [sys.executable, str(script_path)]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired:
        logger.error("TIMEOUT: %s exceeded %s seconds", description, timeout_sec)
        return False
    except Exception as exc:
        logger.error("ERROR: %s failed with exception: %s", description, exc)
        return False

    if proc.stdout:
        logger.info("%s stdout:\n%s", description, proc.stdout.strip())
    if proc.stderr:
        logger.warning("%s stderr:\n%s", description, proc.stderr.strip())

    if proc.returncode != 0:
        logger.error("FAILED: %s (exit=%s)", description, proc.returncode)
        return False

    logger.info("SUCCESS: %s", description)
    return True


def main() -> int:
    logger = build_logger()
    logger.info("=" * 70)
    logger.info("NIGHTLY NBA RETRAIN START")
    logger.info("Working directory: %s", ROOT)
    logger.info("Python executable: %s", sys.executable)
    logger.info("=" * 70)

    # Optional guard: skip data fetch if disabled explicitly.
    skip_fetch = os.getenv("NBA_SKIP_FETCH", "0") == "1"

    tasks = []
    if not skip_fetch:
        tasks.append(("fetch_nba_stats.py", "Refresh NBA player data", True, 1800))

    tasks.append(("train_nba_ensemble_models.py", "Train NBA ensemble prop models", True, 3600))

    results = []
    for script_name, description, required, timeout_sec in tasks:
        ok = run_step(logger, script_name, description, timeout_sec)
        results.append({
            "script": script_name,
            "description": description,
            "required": required,
            "ok": ok,
        })

    logger.info("\n" + "=" * 70)
    logger.info("NIGHTLY NBA RETRAIN SUMMARY")
    logger.info("=" * 70)

    for result in results:
        status = "PASS" if result["ok"] else "FAIL"
        logger.info("%s - %s", status, result["description"])

    required_failures = [r for r in results if r["required"] and not r["ok"]]
    if required_failures:
        logger.error("Required failures: %s", len(required_failures))
        logger.error("Nightly retrain did not complete successfully.")
        return 1

    logger.info("Nightly retrain completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
