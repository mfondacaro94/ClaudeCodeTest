#!/bin/bash
# Wait for baseball-reference to lift rate limit, then resume scraping.
# Checks every 30 minutes. Logs to logs/resume_check.log.

cd "$(dirname "$0")/.."
LOG="logs/resume_check.log"

echo "$(date): Waiting for baseball-reference rate limit to lift..." >> "$LOG"

while true; do
    STATUS=$(curl -s -o /dev/null -w "%{http_code}" "https://www.baseball-reference.com/teams/NYY/2024.shtml")
    echo "$(date): BR status = $STATUS" >> "$LOG"

    if [ "$STATUS" = "200" ]; then
        echo "$(date): Rate limit lifted! Starting scraper..." >> "$LOG"
        # Wait 5 min extra to be safe
        sleep 300
        python3 scraper/scrape_pitcher_gamelogs.py >> logs/overnight_scrape.log 2>&1
        echo "$(date): Scraper finished." >> "$LOG"
        exit 0
    fi

    # Check every 30 minutes
    sleep 1800
done
