# Deploying LitRead on a Mac mini (always-on personal server)

Run LitRead as a background service on your Mac mini and reach it from any device,
privately, over HTTPS via Tailscale.

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

## 3. Expose it via Tailscale (private + HTTPS)

```bash
bash deploy/expose-tailscale.sh
```

You get a stable URL like `https://mac-mini.your-tailnet.ts.net`. Only devices signed
into *your* Tailscale account can reach it. Open it on your phone/laptop → browser
**Install / Add to Home Screen** → LitRead behaves like an installed app.

## 4. Install LitRead as a desktop app (icon on your laptop)

LitRead ships a PWA manifest (`static/manifest.webmanifest`, `display: standalone`,
icons), so once it's served over HTTPS you can install it as a standalone app with its
own icon — no download or wrapper needed.

On your laptop, open the HTTPS Tailscale URL, then:
- **Chrome / Edge** — click the **Install** icon in the address bar (or ⋮ menu →
  "Install LitRead…"). Creates **LitRead.app** in Applications/Launchpad.
- **Safari 17+ (macOS Sonoma/Sequoia)** — **File → Add to Dock**.

It launches like a native app (own window + LitRead icon) and can be pinned to the Dock.

> The installed app is a thin client pointing at the mini — it only works while the
> mini is on, the service is running, and your device is on Tailscale.

## Updating after a code change

One command (pull + uv sync + restart):
```bash
bash deploy/update-macmini.sh
```

The Tailscale tunnel needs no refresh — it proxies to the same port, so it serves the
new version as soon as the service restarts. PWA icon/manifest changes still require
**uninstall + reinstall** of the app on each client (icon cached at install time).
