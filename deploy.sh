#!/usr/bin/env bash
# deploy.sh — the one script to deploy LitRead on the Mac mini.
# Installs (or refreshes) the always-on launchd service, pulls the latest code,
# syncs deps, restarts the service, and exposes it over Tailscale.
# Run from inside the repo on the Mac mini:  bash deploy.sh
#
# One-time prereqs (see deploy/README.md):
#   - uv venv + uv pip install -r requirements.txt   (creates .venv)
#   - cp .env.example .env  and fill in API keys
#   - tailscale up  (+ enable MagicDNS/HTTPS) for the public-ish HTTPS URL
set -euo pipefail

REPO="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO"
PY="$REPO/.venv/bin/python"
PORT="${PORT:-7860}"
LABEL="com.litread.app"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
USER_NAME="$(id -un)"

if [ ! -x "$PY" ]; then
  echo "ERROR: no venv python at $PY"
  echo "Create it first:  uv venv && uv pip install -r requirements.txt"
  exit 1
fi
if [ ! -f "$REPO/.env" ]; then
  echo "WARNING: no $REPO/.env — set OPENAI_API_KEY and/or XAI_API_KEY there before use."
fi

echo "==> git pull"
git pull --ff-only

echo "==> install dependencies"
if command -v uv >/dev/null 2>&1; then
  uv pip install --python "$PY" -r requirements.txt
else
  "$PY" -m pip install -q -r requirements.txt
fi

# Write/refresh the launchd LaunchAgent. Idempotent: rewriting every deploy keeps
# the config correct. NOTE: do NOT add StandardOutPath/StandardErrorPath — on macOS
# Sequoia those keys make the agent fail to spawn (EX_CONFIG 78).
echo "==> writing $PLIST"
mkdir -p "$HOME/Library/LaunchAgents"
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
        <key>PATH</key><string>/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
        <key>PYTHONUNBUFFERED</key><string>1</string>
        <key>HOME</key><string>$HOME</string>
        <key>USER</key><string>$USER_NAME</string>
    </dict>
    <key>RunAtLoad</key><true/>
    <key>KeepAlive</key><true/>
</dict>
</plist>
PLISTEOF

# Free the port from any stray terminal-launched process, then (re)load the service.
lsof -ti ":$PORT" | xargs kill -9 2>/dev/null || true
echo "==> (re)loading service"
launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
sleep 1
launchctl bootstrap "gui/$(id -u)" "$PLIST"

# Expose over Tailscale (same port, so this survives every redeploy).
if command -v tailscale >/dev/null 2>&1; then
  echo "==> tailscale serve"
  tailscale serve --bg "http://127.0.0.1:$PORT" 2>/dev/null || true
fi

# Wait for the app to come up.
echo -n "Waiting for app to start"
for _ in $(seq 1 20); do
  if curl -s -o /dev/null -w "%{http_code}" "http://localhost:$PORT/" 2>/dev/null | grep -q "200"; then
    echo " OK"
    break
  fi
  echo -n "."
  sleep 1
done

echo
echo "Done."
echo "  Local:     http://localhost:$PORT"
if command -v tailscale >/dev/null 2>&1; then
  TSNAME=$(tailscale status --json 2>/dev/null | python3 -c \
    "import sys,json; d=json.load(sys.stdin); print(d.get('Self',{}).get('DNSName','').rstrip('.'))" 2>/dev/null || true)
  [ -n "$TSNAME" ] && echo "  Tailscale: https://$TSNAME"
fi
echo
echo "Manage:  launchctl bootout gui/$(id -u)/$LABEL    # stop"
echo "         bash deploy.sh                          # redeploy after a code change"
