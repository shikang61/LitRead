# Deploying LitRead on a Mac mini (always-on personal server)

Run LitRead as a background service on your Mac mini and reach it from any device,
privately, over HTTPS via Tailscale. **One script does it all: `deploy.sh`.**

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

Set up Tailscale once so you get a stable private HTTPS URL:

```bash
brew install tailscale            # or install the Tailscale.app
sudo tailscale up                 # log in (also sign in on your phone/laptop)
# Enable MagicDNS + HTTPS for your tailnet once:
#   https://login.tailscale.com/admin/dns
```

Recommended: in **System Settings → Users & Groups → Automatically log in as** your
user, so the service comes back after a reboot/power cut (a LaunchAgent runs once
you're logged in).

## 2. Deploy

```bash
bash deploy.sh
```

The **same command** does first-time install and every later update. It:

- pulls the latest code (`git pull`) and syncs deps,
- writes/refreshes a `launchd` LaunchAgent (`~/Library/LaunchAgents/com.litread.app.plist`)
  that starts `app.py` at login and **restarts it if it crashes** (`KeepAlive`),
- serves on `http://localhost:7860`,
- keeps the cache on local disk (`./cache`) so carousels/summaries + page images
  persist and you don't re-spend tokens,
- exposes it over Tailscale at a stable HTTPS URL like
  `https://mac-mini.your-tailnet.ts.net`.

Manage it:

```bash
bash deploy.sh                                   # redeploy after a code change
launchctl bootout gui/$(id -u)/com.litread.app   # stop / uninstall the service
```

> The service writes no log files (those `launchd` keys break the agent on macOS
> Sequoia). To debug a crash, run it in the foreground:
> `cd LitRead && .venv/bin/python app.py`.

## 3. Install LitRead as a desktop / home-screen app

LitRead ships a PWA manifest (`static/manifest.webmanifest`, `display: standalone`,
icons), so once it's served over HTTPS you can install it as a standalone app with its
own icon — no download or wrapper needed.

On your phone/laptop, open the HTTPS Tailscale URL, then:

- **Chrome / Edge** — click the **Install** icon in the address bar (or ⋮ menu →
  "Install LitRead…"). Creates **LitRead.app** in Applications/Launchpad.
- **Safari 17+ (macOS Sonoma/Sequoia)** — **File → Add to Dock**.
- **iOS / Android** — browser menu → **Add to Home Screen**.

It launches like a native app (own window + LitRead icon) and can be pinned to the Dock.

> The installed app is a thin client pointing at the mini — it only works while the
> mini is on, the service is running, and your device is on Tailscale. PWA
> icon/manifest changes require **uninstall + reinstall** on each client (the icon is
> cached at install time).
