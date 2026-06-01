# Deploying LitRead on a Mac mini (always-on personal server)

Run LitRead as a background service on your Mac mini and reach it from any device,
privately, over HTTPS. **Tailscale is the recommended path** (private, no domain,
HTTPS, free); Cloudflare Tunnel is an alternative if you want a public URL / custom
domain.

## 1. One-time prep on the Mac mini

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh       # install uv (skip if present)
git clone <your-repo> LitRead && cd LitRead
uv venv                                               # creates .venv
uv pip install -r requirements.txt
cp .env.example .env                                  # then edit .env:
#   OPENAI_API_KEY=...        (and/or)
#   XAI_API_KEY=...
```

Recommended: in **System Settings → Users & Groups → Automatically log in as** your
user, so the service comes back after a reboot/power cut (a LaunchAgent runs once
you're logged in).

## 2. Install the always-on service

```bash
bash deploy/install-macmini.sh
```

This writes a `launchd` LaunchAgent (`~/Library/LaunchAgents/com.litread.app.plist`)
that:
- starts `app.py` at login and **restarts it if it crashes** (`KeepAlive`),
- serves on `http://localhost:7860`,
- keeps the cache on local disk (`./cache`) so carousels/summaries + page images
  persist and you don't re-spend tokens,
- logs to `deploy/logs/`.

Manage it:
```bash
launchctl kickstart -k gui/$(id -u)/com.litread.app   # restart (after a git pull)
launchctl bootout   gui/$(id -u)/com.litread.app       # stop / uninstall
tail -f deploy/logs/litread.log                        # watch logs
```

## 3. Expose it

### Option A — Tailscale (recommended: private + HTTPS)
```bash
brew install tailscale          # or install Tailscale.app
sudo tailscale up               # log in (do this on the mini AND your phone/laptop)
# Enable MagicDNS + HTTPS once: https://login.tailscale.com/admin/dns
bash deploy/expose-tailscale.sh
```
You get a stable URL like `https://mac-mini.your-tailnet.ts.net`. Only devices signed
into *your* Tailscale account can reach it. Open it on your phone/laptop → browser
**Install / Add to Home Screen** → LitRead behaves like an installed app.

### Option B — Cloudflare Tunnel (public URL / custom domain)
```bash
brew install cloudflared
bash deploy/expose-cloudflare.sh        # quick, temporary https://*.trycloudflare.com URL
```
For a **permanent** URL on your own domain (domain must be on Cloudflare):
```bash
cloudflared tunnel login
cloudflared tunnel create litread
# map a hostname to the tunnel:
cloudflared tunnel route dns litread litread.yourdomain.com
# ~/.cloudflared/config.yml:
#   tunnel: <TUNNEL_ID>
#   credentials-file: /Users/<you>/.cloudflared/<TUNNEL_ID>.json
#   ingress:
#     - hostname: litread.yourdomain.com
#       service: http://localhost:7860
#     - service: http_status:404
cloudflared tunnel run litread        # (or install as a service: sudo cloudflared service install)
```
Note: this is **public** — anyone with the URL can use it (and spend your API tokens).
Put it behind **Cloudflare Access** (email/SSO gate) if you go this route.

## 4. Install LitRead as a desktop app (icon on your laptop)

LitRead ships a PWA manifest (`static/manifest.webmanifest`, `display: standalone`,
icons), so once it's served over **HTTPS** (the Tailscale/Cloudflare step above) you
can install it as a standalone app with its own icon — no download or wrapper needed.

On your laptop, open the HTTPS URL (e.g. `https://<mac-mini>.<tailnet>.ts.net`), then:
- **Chrome / Edge** — click the **Install** icon in the address bar (or ⋮ menu →
  "Install LitRead…"). Creates **LitRead.app** in Applications/Launchpad that opens in
  its own window.
- **Safari 17+ (macOS Sonoma/Sequoia)** — **File → Add to Dock**. Lands in the
  Dock/Launchpad as an app.

It launches like a native app (own window + LitRead icon) and can be pinned to the Dock.

Requirements for the Install option to appear / work:
1. Served over **HTTPS** — Tailscale Serve or the Cloudflare tunnel provides this; a plain
   `http://<ip>:7860` will NOT offer "Install".
2. The installed app is a thin client pointing at the mini — it only works while the
   **mini is on, the service is running, and your laptop is on Tailscale**. Otherwise the
   window shows a connection error.

## Updating after a code change
One command (pull + uv sync + restart):
```bash
bash deploy/update-macmini.sh
```
Or manually:
```bash
git pull
uv pip install -r requirements.txt           # if deps changed
launchctl kickstart -k gui/$(id -u)/com.litread.app
```
The tunnel needs no refresh — it proxies to the same port, so it serves the new
version as soon as the service restarts. PWA icon/manifest changes still require
**uninstall + reinstall** of the app on each client (icon cached at install time).

## Notes
- The app already binds `0.0.0.0` and reads `PORT`, so both tunnels just point at
  `localhost:7860`.
- API keys live only in `.env` on the mini — never in the tunnel or the image.
- Tailscale gives HTTPS, which (with the existing PWA manifest) lets every device
  install LitRead as an app.
