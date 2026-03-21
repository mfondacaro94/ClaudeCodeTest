"""Run all scrapers sequentially to avoid rate-limit conflicts.

Order: games (re-scrape with new schema) -> injuries (2015-2022) -> batters (missing players) -> pitchers (verify)
"""
import subprocess
import sys
import time
from pathlib import Path

SCRAPER_DIR = Path(__file__).resolve().parent.parent / "scraper"

scrapers = [
    ("Games", SCRAPER_DIR / "scrape_games.py"),
    ("Injuries", SCRAPER_DIR / "scrape_injuries.py"),
    ("Batters", SCRAPER_DIR / "scrape_batters.py"),
    ("Pitchers", SCRAPER_DIR / "scrape_pitchers.py"),
]

for name, script in scrapers:
    print(f"\n{'='*60}")
    print(f"  Starting: {name} scraper")
    print(f"{'='*60}\n", flush=True)

    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(SCRAPER_DIR.parent),
    )

    if result.returncode != 0:
        print(f"\n*** {name} scraper failed (exit code {result.returncode}) ***")
    else:
        print(f"\n*** {name} scraper completed successfully ***")

    # Brief pause between scrapers to let rate limits cool down
    time.sleep(10)

print(f"\n{'='*60}")
print("  All scrapers finished")
print(f"{'='*60}")
