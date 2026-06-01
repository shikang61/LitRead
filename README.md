---
title: LitRead
emoji: 📚
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: "6.0.0"
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
---

# 📚 LitRead

Paste an ArXiv link, get a captivating bite-sized summary rendered as a card grid. Powered by
OpenAI or Grok via LangChain, served through a minimalist Gradio web UI.

![screenshot placeholder](docs/screenshot.png)

## Features

- 🔗 **Paste any ArXiv URL or ID** — `https://arxiv.org/abs/2305.10601`, `2305.10601v3`, or `cs.LG/0701001`.
- 🃏 **Carousel cards** — one card per aspect (Motivation, Context, Proposed Solution, How It Works,
  Key Results, Comparison, Future Work) with a 1-line framing sentence + 2–4 punchy bullets.
- 🔎 **Annotated reader** — `Annotate Paper` renders each PDF page beside a notes column: the LLM
  boxes the high-signal paragraphs/equations/figures, highlights key phrases, and writes a one-line
  takeaway per box. Click a note to scroll to + flash its box. Skim the notes for the gist, then
  dive into the boxes you care about.
- ✍️ **Layman language** — captivating, jargon-light tone with key technical terms bolded.
- ➗ **Math rendering** — LaTeX via MathJax (`$inline$`, `$$display$$`).
- 🔄 **Live timers** — fetch + generation elapsed time tick every 0.3s.
- 🪙 **API cost panel** — input/output token counts and estimated USD cost per run.
- 📥 **Export as PNG** — one-click client-side capture of the rendered card grid (html2canvas).
- 📚 **Export to Zotero** — auto-generated `.bib` download; one-click `Open PDF` / `ArXiv Page` links.
- 🛡 **HTTP 429 retry** — auto-retries arxiv rate-limit responses with countdown.
- 🌐 **Provider-flexible** — Grok (`grok-4.3` default) or OpenAI (`gpt-5.4`) via the same
  `ChatOpenAI` client (Grok uses the OpenAI-compatible xAI base URL).
- 🚦 **Bounded concurrency** — Gradio queue caps in-flight runs (`default_concurrency_limit=2`).
- 🔐 **`.env` config** — no API-key input box in the UI; keys live in `.env` or shell env vars.

## Tech stack

| Layer            | Library                              |
|------------------|--------------------------------------|
| Web UI           | `gradio` (6.x)                       |
| LLM orchestration| `langchain-core`, `langchain-openai` |
| ArXiv fetch      | `arxiv` (Python client)              |
| PDF text extract | `pymupdf` (`fitz`)                   |
| Env loader       | `python-dotenv`                      |
| Math rendering   | MathJax 3 (CDN, client-side)         |
| PNG export       | html2canvas (CDN, client-side)       |

## Quick start

```bash
# 1. Clone & enter directory
git clone <repo-url> arxiv-carousel
cd arxiv-carousel

# 2. Create a venv (Python 3.11+ recommended)
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure API keys
cp .env.example .env
#   then edit .env and fill in OPENAI_API_KEY and/or XAI_API_KEY

# 5. Launch
python app.py
```

By default the app binds to `http://127.0.0.1:7860`.

## Configuration

### `.env`

```env
OPENAI_API_KEY=sk-...
XAI_API_KEY=xai-...
```

You only need the key for the provider you intend to use. Either variable can also be set as
a shell environment variable; real env vars take precedence over `.env`.

### Model IDs and pricing

Top of `app.py`:

```python
OPENAI_MODEL = "gpt-5.4"
GROK_MODEL   = "grok-4.3"
GROK_BASE_URL = "https://api.x.ai/v1"

PRICING = {
    "gpt-5.4":  {"input": 5.00, "output": 15.00},   # USD per 1M tokens
    "grok-4.3": {"input": 5.00, "output": 15.00},
}
```

Edit these constants to match real model names and the pricing tier you're billed at. If you
change a model ID, also update the matching entry in `PRICING` so the cost panel stays accurate.

### Limits

```python
MAX_PAPER_CHARS    = 200_000  # paper text fed to the LLM (~50k tokens)
MAX_FETCH_RETRIES  = 4        # ArXiv API retry budget on HTTP 429
```

Raise `MAX_PAPER_CHARS` if you want longer papers to be summarized in full (and you trust the
model's context window + your wallet).

## Project layout

```
.
├── app.py            # Carousel pipeline + Gradio UI, callbacks, CSS, MathJax loader
├── core.py           # Shared helpers: providers, ArXiv ID, fetch_pdf, LLM factory, cache, JSON parser
├── annotate.py       # Annotated reader: block extraction, page render, annotation LLM call, HTML
├── tests/            # pytest suite (core + annotate; network smoke gated by RUN_NET_SMOKE)
├── requirements.txt  # Pinned minimums for runtime deps
├── .env.example      # Template — copy to .env and fill in keys
├── .env              # (gitignored) your real keys
├── .gitignore
├── claude.md         # Original project brief
├── README.md         # You are here
└── DEPLOY.md         # Online-deployment recipes
```

## Code walkthrough

`app.py` is split into clearly-marked sections; jump to any of these via `grep`:

| Section header                                    | Purpose                                                                  |
|---------------------------------------------------|--------------------------------------------------------------------------|
| `Config`                                          | Model IDs, pricing, paper-size cap, retry budget                         |
| `System prompt`                                   | The carousel JSON schema and rules sent to the LLM                       |
| `Helpers`                                         | URL parser, ArXiv loader, LLM factory, cost estimator, BibTeX builder    |
| `HTML renderers`                                  | Paper-header, action row, cards, hook, cost panel, error/placeholder     |
| `Main callback — generate_carousel`              | The streaming generator wired to the Generate button                     |
| `UI`                                              | Gradio Blocks layout + CSS + MathJax + html2canvas PNG-export loader     |

### Pipeline at a glance

```
┌────────────┐  regex   ┌──────────┐  arxiv API  ┌──────────┐  pymupdf  ┌──────────┐
│ User input │ ───────► │ ArXiv ID │ ──────────► │ PDF bytes│ ───────► │ Text +  │
└────────────┘          └──────────┘             └──────────┘           │ metadata │
                                                                        └────┬─────┘
                                                                             │
                                              ┌──────────────────────────────┘
                                              ▼
                                    ┌────────────────────┐
                                    │  ChatOpenAI.stream │  ◄── system prompt requests strict JSON
                                    │  (background thread)│
                                    └─────────┬──────────┘
                                              │
                          chunks via queue.Queue, polled at 0.3s
                                              │
                                              ▼
                                      ┌───────────────┐
                                      │ parse JSON +  │  tolerant of code fences and
                                      │ render cards  │  invalid `\` escapes
                                      └──────┬────────┘
                                             │
                                             ▼
                            ┌─────────────────────────────────┐
                            │ gr.HTML output  ◄── MathJax     │
                            │  (card grid, hook, header)      │
                            └─────────────────────────────────┘
```

### Streaming + live timer

The fetch and the LLM stream both run on background `threading.Thread` workers. The main Gradio
generator polls them via a `queue.Queue` with a 0.3–0.5s timeout, yielding a status update on
every poll. This means:

- The timer ticks even during the synchronous PDF download and the LLM's wait-for-first-token gap.
- HTTP 429 retries show a per-second countdown.
- Token usage updates as `usage_metadata` arrives on the final stream chunk.

### JSON parsing

LLMs sometimes emit invalid `\` escapes (e.g. LaTeX `\(`, `\$`). `parse_carousel_json()` first
tries `json.loads()`; on failure it doubles up invalid backslashes (`\(` → `\\(`) and retries.
This preserves LaTeX while still parsing.

### Math (LaTeX) rendering

MathJax 3 is loaded from CDN in the page `<head>`. A small DOM observer re-typesets the `#output`
container whenever Gradio re-renders the card HTML.

### PNG export

`html2canvas` is loaded from CDN alongside MathJax. The `📥 Export as PNG` button calls
`exportCardsToPng()`, which hides the action row, snapshots `#output` at 2× scale, and triggers a
browser download of `arxiv-carousel.png`. Fully client-side — no server round-trip.

### BibTeX / Zotero export

`make_bibtex()` builds an arXiv-style `@article{…}` entry from the paper metadata, base64-encodes
it into a `data:application/x-bibtex` link, and exposes it as a one-click `.bib` download. Zotero's
"Import from clipboard / file" recognises the format and auto-fills all fields.

## Troubleshooting

| Symptom                                                            | Likely cause / fix                                                                 |
|--------------------------------------------------------------------|-------------------------------------------------------------------------------------|
| `Failed to fetch paper … HTTP 429`                                 | ArXiv rate-limited your IP. Wait 5–10 minutes. Retry budget is `MAX_FETCH_RETRIES`. |
| `Missing OPENAI_API_KEY` / `XAI_API_KEY`                           | Key missing from `.env` and shell env. Re-check `cat .env`.                         |
| `404 model_not_found` from provider                                | `gpt-5.4` / `grok-4.3` are placeholders. Change `OPENAI_MODEL` / `GROK_MODEL`.       |
| `PyTorch was not found … Disabling PyTorch`                        | Stray transitive dep. Harmless — the app doesn't use PyTorch.                       |
| Math doesn't render                                                | MathJax CDN blocked. Allow `cdn.jsdelivr.net` or self-host MathJax.                  |
| Cost shows `$0.0000` after a real run                              | Provider didn't return `usage_metadata`. Check `stream_usage=True` is accepted.     |

## Customization recipes

### Add a third provider

```python
PROVIDERS["Anthropic"] = {
    "model": "claude-sonnet-4-6",
    "base_url": None,  # claude API uses anthropic SDK, not OpenAI-compat;
                       #   wire via langchain-anthropic instead
    "env_key": "ANTHROPIC_API_KEY",
}
```

Note: Anthropic's API is not OpenAI-compatible. You'd swap `ChatOpenAI` for
`ChatAnthropic` (`langchain-anthropic`) in `make_llm()` based on the provider.

### Change the carousel aspects

Edit `CAROUSEL_PROMPT`. Remove or rename aspects; LLM follows whatever schema you ask for. The
renderer accepts arbitrary `{emoji, title, body, bullets}` slides.

### Self-host MathJax (offline / air-gapped)

Replace the CDN `<script>` in `MATHJAX_HEAD` with a path to a local copy of `tex-mml-chtml.js`.
Gradio serves static files via its built-in handler if you place them under `./static/`.

## License

MIT — do what you want; no warranty.

## Acknowledgements

- ArXiv for free open-access papers.
- LangChain for the streaming/orchestration plumbing.
- Gradio for the UI substrate.
- MathJax for in-browser LaTeX.
