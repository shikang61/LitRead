# 🚀 Deploying ArXiv Carousel online

This guide covers running the app on a hosted server so you don't have to launch `python app.py`
on your laptop every time. Options are ordered from easiest → most flexible.

| Option                      | Cost                     | Setup time | Sleeps?            | Best for                          |
|-----------------------------|--------------------------|------------|--------------------|-----------------------------------|
| 1. Hugging Face Spaces      | Free (CPU basic) / paid  | ~5 min     | No (always on)     | Quickest path, no infra at all    |
| 2. Render                   | Free / $7+/mo            | ~10 min    | Free tier sleeps   | Simple deploy from a GitHub repo  |
| 3. Railway                  | $5/mo trial credit       | ~10 min    | No                 | Usage-based, easy GitHub link     |
| 4. Fly.io                   | Free (3 small machines)  | ~15 min    | No                 | Docker, global region pinning     |
| 5. VPS (Hetzner / DO / EC2) | $4–$12/mo                | ~30 min    | No                 | Full control, custom domain       |
| 6. Google Cloud Run         | Pay per request          | ~20 min    | Scale-to-zero      | Bursty traffic, no fixed cost     |

> **TL;DR — for a personal one-user app, use Hugging Face Spaces (Option 1).** It's free,
> always-on, and built specifically for Gradio apps.

---

## Pre-flight checklist (any host)

1. The repo must contain `app.py`, `requirements.txt`, and a `.env.example` (no real `.env`).
2. **Never commit `.env`** — confirm `.env` is in `.gitignore`. Real keys go into the host's
   secrets/environment-variables UI.
3. Make sure `app.py` ends with `build_ui().launch(...)`. For hosted servers add
   `server_name="0.0.0.0"` and respect the host's `PORT` env var (see per-host notes below).
4. Push everything to a public or private GitHub repo — most hosts pull from there.

---

## 1. Hugging Face Spaces — recommended

Free tier gives an always-on CPU container. Native Gradio support.

### Steps

1. Create an account at <https://huggingface.co/join> if you don't have one.
2. Click **"+ New Space"** → name it (e.g. `arxiv-carousel`), pick **Gradio** as the SDK,
   visibility = Public or Private.
3. Clone the Space repo locally:
   ```bash
   git clone https://huggingface.co/spaces/<your-username>/arxiv-carousel hf-space
   cd hf-space
   cp /path/to/your/repo/{app.py,requirements.txt} .
   ```
4. Add a tiny `README.md` at the top of the Space with the required front-matter:
   ```yaml
   ---
   title: ArXiv Carousel
   emoji: 📚
   colorFrom: indigo
   colorTo: blue
   sdk: gradio
   sdk_version: "6.0.0"     # match the gradio version in requirements.txt
   app_file: app.py
   pinned: false
   ---
   ```
   (Spaces uses this to pick the runtime.)
5. Push:
   ```bash
   git add . && git commit -m "Initial deploy" && git push
   ```
6. In the Space UI, go to **Settings → Variables and secrets** and add:
   - `OPENAI_API_KEY` (mark as Secret)
   - `XAI_API_KEY`     (mark as Secret)
7. Wait ~2 minutes for the first build. The app will be live at
   `https://huggingface.co/spaces/<your-username>/arxiv-carousel`.

### Notes for Spaces

- The free CPU tier has limited memory; if PyTorch / sentence-transformers ship transitively and
  OOM you, trim `requirements.txt` to the runtime essentials only (see "Minimal requirements"
  below).
- Spaces auto-injects an env var `PORT` and runs `python app.py`. Your existing `launch()` call
  works as-is.
- Updates: just `git push` to the Space repo — auto-rebuild kicks in.

---

## 2. Render

Free tier sleeps after 15 min of inactivity (~30s cold start). Paid `$7/mo` Starter is always-on.

### Steps

1. Push your repo to GitHub.
2. Sign in to <https://render.com> → **New +** → **Web Service** → connect the GitHub repo.
3. Configure:
   - **Runtime:** Python 3
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `python app.py`
   - **Instance type:** Free (or Starter for always-on)
4. **Environment → Add Environment Variable**: `OPENAI_API_KEY`, `XAI_API_KEY`.
5. Edit `app.py`'s `launch()` to read Render's port:
   ```python
   build_ui().launch(
       server_name="0.0.0.0",
       server_port=int(os.environ.get("PORT", 7860)),
       theme=...,
       css=CSS,
       head=MATHJAX_HEAD,
   )
   ```
6. Click **Deploy**. URL appears under the service page.

---

## 3. Railway

```bash
npm install -g @railway/cli
railway login
railway init        # in your repo dir
railway up          # builds + deploys
railway variables set OPENAI_API_KEY=sk-... XAI_API_KEY=xai-...
```

Railway auto-detects Python from `requirements.txt`. Same `launch()` tweak as Render
(`server_name="0.0.0.0"`, `server_port=PORT`).

---

## 4. Fly.io (Docker)

Persistent free-tier VMs. Requires a `Dockerfile`.

### `Dockerfile`

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

ENV PORT=7860
EXPOSE 7860
CMD ["python", "app.py"]
```

### `fly.toml` (generated by `fly launch`)

```toml
app = "arxiv-carousel"
primary_region = "iad"

[http_service]
  internal_port = 7860
  force_https = true
  auto_stop_machines = "stop"
  auto_start_machines = true
  min_machines_running = 0

[env]
  PORT = "7860"
```

### Steps

```bash
brew install flyctl    # or: curl -L https://fly.io/install.sh | sh
fly auth login
fly launch --no-deploy        # generates fly.toml
fly secrets set OPENAI_API_KEY=sk-... XAI_API_KEY=xai-...
fly deploy
```

Open `https://<app-name>.fly.dev`.

---

## 5. VPS (Hetzner / DigitalOcean / EC2)

Full control. Pair with `systemd` for auto-restart and `caddy` for HTTPS + custom domain.

### One-time server setup

```bash
# On a fresh Ubuntu 22.04 box, as root:
apt update && apt install -y python3-venv git caddy

# App user
adduser --system --group --home /opt/arxiv-carousel app
su - app
git clone https://github.com/<you>/arxiv-carousel.git
cd arxiv-carousel
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Secrets — root-readable only
sudo tee /opt/arxiv-carousel/.env <<EOF
OPENAI_API_KEY=sk-...
XAI_API_KEY=xai-...
EOF
sudo chown app:app /opt/arxiv-carousel/.env
sudo chmod 600     /opt/arxiv-carousel/.env
```

### systemd unit — `/etc/systemd/system/arxiv-carousel.service`

```ini
[Unit]
Description=ArXiv Carousel (Gradio)
After=network.target

[Service]
User=app
Group=app
WorkingDirectory=/opt/arxiv-carousel
EnvironmentFile=/opt/arxiv-carousel/.env
ExecStart=/opt/arxiv-carousel/.venv/bin/python app.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now arxiv-carousel
sudo systemctl status arxiv-carousel
```

### Caddy reverse proxy + HTTPS — `/etc/caddy/Caddyfile`

```
arxiv.yourdomain.com {
    reverse_proxy 127.0.0.1:7860
}
```

```bash
sudo systemctl reload caddy
```

Point your domain's A record at the server's IP — Caddy auto-issues a Let's Encrypt cert.

### Updates

```bash
ssh app@server
cd /opt/arxiv-carousel
git pull
.venv/bin/pip install -r requirements.txt
sudo systemctl restart arxiv-carousel
```

---

## 6. Google Cloud Run (containerized, scale-to-zero)

Pay only when used. Cold start ~3–5s.

### Build & deploy

```bash
gcloud auth login
gcloud config set project <your-gcp-project>

# Build & push to Artifact Registry
gcloud builds submit --tag gcr.io/<project>/arxiv-carousel

# Deploy
gcloud run deploy arxiv-carousel \
  --image gcr.io/<project>/arxiv-carousel \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --set-secrets OPENAI_API_KEY=openai-key:latest,XAI_API_KEY=xai-key:latest \
  --port 7860 \
  --memory 1Gi
```

(Store keys via `gcloud secrets create openai-key --data-file=- < ...` first.)

---

## Required `app.py` tweak for hosted servers

Hosted platforms (Render, Railway, Fly, Cloud Run, VPS) expect the app to bind to `0.0.0.0` and
to honor a `$PORT` env var. Replace the bottom of `app.py`:

```python
if __name__ == "__main__":
    build_ui().launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        theme=gr.themes.Soft(
            text_size=gr.themes.sizes.text_lg,
            font=[
                gr.themes.GoogleFont("Inter"),
                "system-ui", "-apple-system", "Segoe UI",
                "Helvetica Neue", "Arial", "sans-serif",
            ],
        ),
        css=CSS,
        head=MATHJAX_HEAD,
    )
```

Hugging Face Spaces is the exception — it injects `server_name="0.0.0.0"` and a port for you, so
the unmodified `build_ui().launch(...)` works.

---

## Minimal `requirements.txt` (for resource-tight hosts)

The current `requirements.txt` carries a few leftover libs from the earlier RAG pipeline. If
you're memory-constrained on the free tier of any host, trim to the runtime essentials:

```text
gradio>=6.0.0
langchain-core>=0.3.0
langchain-openai>=0.2.0
arxiv>=2.1.0
pymupdf>=1.24.0
python-dotenv>=1.0.0
```

(Drop `faiss-cpu`, `sentence-transformers`, `torch`, `langchain-huggingface`,
`langchain-classic`, `langchain-text-splitters`, `langchain-community` — none are imported in
the current minimalist app.)

---

## Locking down access

If you don't want a public URL:

- **Hugging Face Spaces:** set the Space to **Private** (Space settings → Visibility).
- **Render/Railway:** put it behind their basic-auth proxy or a Cloudflare Access policy.
- **VPS + Caddy:** add basic auth in the Caddyfile:
  ```
  arxiv.yourdomain.com {
      basicauth {
          you JDJhJDEwJC...  # hash from `caddy hash-password`
      }
      reverse_proxy 127.0.0.1:7860
  }
  ```
- **Gradio built-in:** add `auth=("user", "pass")` to `launch()`. Simple but not secure for
  exposed deployments — combine with HTTPS.

---

## Cost-safety knobs

The app pays per request via your OpenAI / xAI keys. To cap exposure on a public deploy:

1. Lower `MAX_PAPER_CHARS` in `app.py` (currently 200,000) → caps tokens per request.
2. Enable hard usage limits at the provider:
   - OpenAI: <https://platform.openai.com/account/limits> → Monthly budget.
   - xAI:    <https://console.x.ai/> → API key spending limits.
3. Add Gradio queuing to throttle concurrent runs:
   ```python
   build_ui().queue(default_concurrency_limit=1, max_size=10).launch(...)
   ```
4. Add basic auth (see above) so only you / invited users can hit it.

---

## Smoke-testing a deploy

After deploying, paste this into the URL box:

```
https://arxiv.org/abs/2305.10601
```

Expected: ~5–15s end-to-end, 4–6 cards rendered, cost panel showing non-zero tokens.

If the page loads but generation fails with `Missing OPENAI_API_KEY`, your secrets aren't wired
in — check the host's env-var UI.

If you get HTTP 429 on every fetch, the host's outbound IP is rate-limited by arxiv; try a
different region or wait 5–10 minutes.
