#!/usr/bin/env bash
# Update an already-installed LitRead service: pull latest, sync deps with uv,
# and restart the launchd service so the new code is loaded.
# Run this from the repo on the Mac mini:  bash deploy/update-macmini.sh
#
# The Tailscale tunnel needs NO refresh — it proxies to the same
# local port, so it serves the new version as soon as the service restarts.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
PY="$REPO/.venv/bin/python"
PORT="${PORT:-7860}"
LABEL="com.litread.app"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Install: curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi
if [ ! -x "$PY" ]; then
  echo "No venv python at $PY — run deploy/install-macmini.sh first."
  exit 1
fi

echo "==> git stash, pull, restore"
git stash
git pull --ff-only
git stash pop

echo "==> uv pip install -r requirements.txt"
uv pip install --python "$PY" -r requirements.txt

echo "==> restart $LABEL"
launchctl kickstart -k "gui/$(id -u)/$LABEL"

echo
echo "Updated. Local: http://localhost:$PORT"
echo "Tunnel keeps serving automatically (same port)."
echo "PWA icon/manifest changes: uninstall + reinstall the app on each client"
echo "(the icon is cached at install time)."
