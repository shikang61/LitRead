#!/usr/bin/env bash
# Install LitRead as an always-on launchd service on a Mac mini (or any Mac).
# Auto-starts at login and restarts if it crashes. Run this from the repo on the
# Mac mini:  bash deploy/install-macmini.sh
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY="$REPO/.venv/bin/python"
PORT="${PORT:-7860}"
LABEL="com.litread.app"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"

if [ ! -x "$PY" ]; then
  echo "No venv python at $PY"
  echo "Create it first:"
  echo "  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
  exit 1
fi
if [ ! -f "$REPO/.env" ]; then
  echo "WARNING: no $REPO/.env — set OPENAI_API_KEY and/or XAI_API_KEY there before use."
fi

mkdir -p "$HOME/Library/LaunchAgents" "$REPO/deploy/logs"

cat > "$PLIST" <<PLISTEOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>$LABEL</string>
    <key>ProgramArguments</key>
    <array>
        <string>$PY</string>
        <string>$REPO/app.py</string>
    </array>
    <key>WorkingDirectory</key><string>$REPO</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PORT</key><string>$PORT</string>
        <key>LITREAD_CACHE_DIR</key><string>$REPO/cache</string>
    </dict>
    <key>RunAtLoad</key><true/>
    <key>KeepAlive</key><true/>
    <key>StandardOutPath</key><string>$REPO/deploy/logs/litread.log</string>
    <key>StandardErrorPath</key><string>$REPO/deploy/logs/litread.err.log</string>
</dict>
</plist>
PLISTEOF

# Reload cleanly (bootout/bootstrap is the modern API; fall back to load/unload).
launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST" 2>/dev/null || launchctl load "$PLIST"

echo "Installed $LABEL"
echo "  Local:  http://localhost:$PORT"
echo "  Logs:   $REPO/deploy/logs/litread.log"
echo "Manage:"
echo "  Restart: launchctl kickstart -k gui/$(id -u)/$LABEL"
echo "  Stop:    launchctl bootout gui/$(id -u)/$LABEL"
echo "Next: expose it — bash deploy/expose-tailscale.sh   (or expose-cloudflare.sh)"
