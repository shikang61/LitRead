#!/usr/bin/env bash
# Expose LitRead via a Cloudflare Tunnel.
#
# Quick mode (default): a temporary public https://*.trycloudflare.com URL — zero
# setup, but the URL changes each run and is public. Good for a quick test.
#
# Named/stable mode (needs a domain on Cloudflare): see deploy/README.md.
#
# Prereqs:  brew install cloudflared
set -euo pipefail
PORT="${PORT:-7860}"

if ! command -v cloudflared >/dev/null 2>&1; then
  echo "cloudflared not found. Install: brew install cloudflared"
  exit 1
fi

echo "Starting a quick Cloudflare tunnel to http://localhost:$PORT ..."
echo "(Ctrl-C to stop. For a permanent URL, set up a named tunnel — see deploy/README.md.)"
cloudflared tunnel --url "http://localhost:$PORT"
