"""Overnight scrape runner — park factors, then pitcher game logs.

Runs sequentially to avoid baseball-reference rate limits.
Umpire data is pre-downloaded from Retrosheet (no scraping needed).

Expected runtime: ~11-12 hours total
- Park factors: 11 pages (~1 min)
- Pitcher game logs: ~6,800 pages (~11 hrs)

All scrapers save progress incrementally. Safe to interrupt and resume.
"""
import subprocess
import sys
import time
from pathlib import Path

SCRAPER_DIR = Path(__file__).resolve().parent.parent / "scraper"

scrapers = [
    ("Park Factors", SCRAPER_DIR / "scrape_park_factors.py"),
    ("Pitcher Game Logs (ALL pitchers)", SCRAPER_DIR / "scrape_pitcher_gamelogs.py"),
]

for name, script in scrapers:
    print(f"\n{'='*60}")
    print(f"  Starting: {name}")
    print(f"  Script: {script}")
    print(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n", flush=True)

    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(SCRAPER_DIR.parent),
    )

    if result.returncode != 0:
        print(f"\n*** {name} failed (exit code {result.returncode}) ***")
        print("Continuing to next scraper...\n")
    else:
        print(f"\n*** {name} completed successfully ***")

    # Brief pause between scrapers
    time.sleep(10)

print(f"\n{'='*60}")
print(f"  All overnight scrapers finished")
print(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*60}")
