"""Full pipeline orchestrator: download -> merge -> features -> train -> evaluate."""

import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.helpers import get_logger

logger = get_logger("pipeline")


def run_step(name: str, script: str):
    """Run a pipeline step as a subprocess."""
    logger.info(f"=== {name} ===")
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / script)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        logger.error(f"{name} failed:\n{result.stderr}")
        raise RuntimeError(f"{name} failed")
    logger.info(f"{name} completed successfully")


def main():
    steps = [
        ("Download tennis-data.co.uk", "scraper/download_tennis_data_uk.py"),
        ("Download Sackmann data", "scraper/download_sackmann.py"),
        ("Merge data sources", "scraper/merge_sources.py"),
        ("Feature engineering", "models/feature_engineering.py"),
        ("Train models", "models/train.py"),
        ("Evaluate models", "models/evaluate.py"),
    ]

    for name, script in steps:
        run_step(name, script)

    logger.info("Pipeline complete!")


if __name__ == "__main__":
    main()
