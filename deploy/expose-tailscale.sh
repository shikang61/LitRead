#!/usr/bin/env bash
# Expose LitRead over your private Tailscale network with HTTPS.
# Prereqs (one-time):
#   brew install tailscale            # or the Tailscale.app from the App Store
#   sudo tailscale up                 # log in
#   Enable HTTPS for your tailnet:  https://login.tailscale.com/admin/dns  (MagicDNS + HTTPS)
set -euo pipefail
PORT="${PORT:-7860}"

if ! command -v tailscale >/dev/null 2>&1; then
  echo "tailscale not found. Install: brew install tailscale  (then: sudo tailscale up)"
  exit 1
fi

# Proxy https://<machine>.<tailnet>.ts.net  ->  the local app. --bg keeps it running.
tailscale serve --bg "http://127.0.0.1:$PORT"

echo
echo "Serving LitRead on your tailnet. Your URL:"
tailscale serve status
echo
echo "Open that https URL on any device signed into your Tailscale account,"
echo "then use the browser's 'Install / Add to Home Screen' to get the PWA."
echo "Stop sharing:  tailscale serve --https=443 off"
