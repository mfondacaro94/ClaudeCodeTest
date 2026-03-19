"""Daily pipeline orchestrator: scrape -> features -> train -> evaluate.

Analogous to UFC weekly_update.py, but runs daily since MLB has games every day.

Usage:
    python scripts/daily_update.py                # Full pipeline
    python scripts/daily_update.py --no-retrain    # Data only
    python scripts/daily_update.py --dry-run       # Preview
"""

import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.helpers import get_logger

logger = get_logger("daily_update")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable


def run_step(name: str, script: str, dry_run: bool = False):
    """Run a pipeline step."""
    logger.info(f"{'[DRY RUN] ' if dry_run else ''}Step: {name}")
    if dry_run:
        return True

    full_path = PROJECT_ROOT / script
    result = subprocess.run(
        [PYTHON, str(full_path)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        logger.error(f"Step '{name}' failed:\n{result.stderr[-500:]}")
        return False

    logger.info(f"Step '{name}' completed successfully")
    return True


def main():
    parser = argparse.ArgumentParser(description="MLB daily update pipeline")
    parser.add_argument("--no-retrain", action="store_true", help="Skip model retraining")
    parser.add_argument("--dry-run", action="store_true", help="Preview steps without executing")
    parser.add_argument("--odds-only", action="store_true", help="Only fetch live odds")
    args = parser.parse_args()

    start = datetime.now()
    logger.info(f"=== Daily Update Started: {start.isoformat()} ===")

    if args.odds_only:
        run_step("Fetch live odds", "scraper/scrape_odds.py", args.dry_run)
        logger.info("Odds-only update complete.")
        return

    steps = [
        ("Scrape game results", "scraper/scrape_games.py"),
        ("Scrape team stats", "scraper/scrape_teams.py"),
        ("Scrape pitcher stats", "scraper/scrape_pitchers.py"),
        ("Feature engineering", "models/feature_engineering.py"),
    ]

    if not args.no_retrain:
        steps.extend([
            ("Train win/loss model", "models/train.py"),
            ("Train totals model", "models/train_totals.py"),
            ("Evaluate models", "models/evaluate.py"),
        ])

    steps.append(("Fetch live odds", "scraper/scrape_odds.py"))

    for name, script in steps:
        success = run_step(name, script, args.dry_run)
        if not success and not args.dry_run:
            logger.error(f"Pipeline failed at step: {name}")
            return

    elapsed = (datetime.now() - start).total_seconds()
    logger.info(f"=== Daily Update Complete: {elapsed:.0f}s ===")


if __name__ == "__main__":
    main()
