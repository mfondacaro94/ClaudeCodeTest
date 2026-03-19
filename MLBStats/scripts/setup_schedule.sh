#!/bin/bash
# Install a macOS launchd job to run the daily update at 5:00 AM every day.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PLIST_NAME="com.mlb-predictor.daily-update"
PLIST_PATH="$HOME/Library/LaunchAgents/${PLIST_NAME}.plist"
PYTHON_PATH="$(which python3)"
LOG_PATH="${PROJECT_DIR}/logs/daily_update.log"

echo "Setting up MLB Predictor daily update..."
echo "  Project: $PROJECT_DIR"
echo "  Python:  $PYTHON_PATH"
echo "  Log:     $LOG_PATH"
echo ""

cat > "$PLIST_PATH" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>${PLIST_NAME}</string>
    <key>ProgramArguments</key>
    <array>
        <string>${PYTHON_PATH}</string>
        <string>${PROJECT_DIR}/scripts/daily_update.py</string>
    </array>
    <key>WorkingDirectory</key>
    <string>${PROJECT_DIR}</string>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>5</integer>
        <key>Minute</key>
        <integer>0</integer>
    </dict>
    <key>StandardOutPath</key>
    <string>${LOG_PATH}</string>
    <key>StandardErrorPath</key>
    <string>${LOG_PATH}</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin</string>
    </dict>
</dict>
</plist>
EOF

launchctl load "$PLIST_PATH"
echo "Installed and loaded: $PLIST_PATH"
echo ""
echo "Manage with:"
echo "  launchctl list | grep mlb"
echo "  launchctl start $PLIST_NAME"
echo "  tail -f $LOG_PATH"
echo "  launchctl unload $PLIST_PATH"
