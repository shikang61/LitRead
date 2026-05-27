"""
ArXiv AI Assistant — paste an ArXiv link, get a card-grid summary.

Pipeline:
    URL  -> regex extract ArXiv ID
         -> arxiv.Client fetches metadata + pymupdf parses PDF text
         -> Full paper text + JSON system prompt -> ChatOpenAI (or Grok)
         -> Parse JSON -> render styled HTML card grid in Gradio
"""

import base64
import html
import json
import os
import queue
import re
import threading
import time
import warnings
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
load_dotenv(override=False)

warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain_community.*")
warnings.filterwarnings("ignore", message=".*langchain-community.*sunset.*")

# Custom UA — arxiv aggressively rate-limits the default Python-urllib UA.
_opener = urllib.request.build_opener()
_opener.addheaders = [
    ("User-Agent", "ArxivCarousel/1.0 (research summarizer; mailto:user@example.com)"),
]
urllib.request.install_opener(_opener)

import arxiv
import fitz  # pymupdf
import gradio as gr
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
OPENAI_MODEL = "gpt-5.4"
GROK_MODEL = "grok-4.3"
GROK_BASE_URL = "https://api.x.ai/v1"

MAX_PAPER_CHARS = 200_000
MAX_FETCH_RETRIES = 4

# Per-1M-token pricing in USD. Update with real numbers when you have them.
# Cost = (in_tok * input + out_tok * output) / 1_000_000
PRICING: Dict[str, Dict[str, float]] = {
    "gpt-5.4":  {"input": 5.00, "output": 15.00},
    "grok-4.3": {"input": 5.00, "output": 15.00},
}

PROVIDERS = {
    "Grok": {
        "model": GROK_MODEL,
        "base_url": GROK_BASE_URL,
        "env_key": "XAI_API_KEY",
    },
    "OpenAI": {
        "model": OPENAI_MODEL,
        "base_url": None,
        "env_key": "OPENAI_API_KEY",
    },
}

# -----------------------------------------------------------------------------
# System prompt — strict JSON output.
# -----------------------------------------------------------------------------
CAROUSEL_PROMPT = """You are a captivating science communicator.

You will receive the full text of a research paper. Produce a JSON object describing a
"bite-sized carousel" summary — a series of compact cards.

OUTPUT FORMAT (strict JSON, no markdown fences, no commentary):
{
  "hook": "<one-line teaser that makes the reader want to read the paper>",
  "slides": [
    {
      "emoji": "🎯",
      "title": "Motivation",
      "body": "1 short sentence framing the card.",
      "bullets": ["punchy point 1", "punchy point 2", "punchy point 3"]
    },
    ...
  ]
}

ASPECTS to cover (skip any absent from the paper):
  🎯 Motivation, 🌍 Context, 💡 Proposed Solution,
  ⚙️ How It Works, 🏆 Key Results, 📊 Comparison to Existing Methods, 🔮 Future Work

RULES:
- Use ONLY information from the paper text. Never invent facts.
- Each `body`: ONE short sentence (max ~20 words) — a framing line, not a full explanation.
- Each `bullets`: 2–4 punchy bullets (max ~12 words each). Use bullets to carry the substance.
- Wrap KEY technical terms in markdown bold: `**term**`. Bold sparingly (1–3 per bullet/body) —
  only the most important concept, method name, or metric. Do not bold whole phrases.
- Layman language. Define jargon briefly when unavoidable. Captivating tone.
- `bullets` is optional — omit if a single short sentence in `body` covers the slide cleanly.
- Output ONLY the JSON object — no prose before/after, no markdown fences.
- You MAY use LaTeX for math: `$x = y^2$` for inline, `$$E = mc^2$$` for display. Keep equations short.
  When emitting LaTeX in JSON strings, escape backslashes as `\\\\` (e.g. `\\\\frac{a}{b}`).
"""


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
ARXIV_ID_RE = re.compile(
    r"(?:arxiv\.org/(?:abs|pdf|html)/)?"
    r"(?P<id>\d{4}\.\d{4,5}(?:v\d+)?|[a-z\-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?)",
    re.IGNORECASE,
)


def extract_arxiv_id(text: str) -> Optional[str]:
    if not text:
        return None
    text = text.strip().rstrip("/")
    if text.lower().endswith(".pdf"):
        text = text[:-4]
    m = ARXIV_ID_RE.search(text)
    return m.group("id") if m else None


def load_arxiv_paper(arxiv_id: str) -> List[Document]:
    client = arxiv.Client(page_size=1, delay_seconds=3.0, num_retries=3)
    results = list(client.results(arxiv.Search(id_list=[arxiv_id])))
    if not results:
        return []
    result = results[0]

    with urllib.request.urlopen(result.pdf_url) as resp:
        pdf_bytes = resp.read()
    with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf:
        page_count = pdf.page_count
        text = "\n\n".join(page.get_text() for page in pdf)

    names = [a.name for a in result.authors]
    if not names:
        authors_display = "Unknown"
    elif len(names) == 1:
        authors_display = names[0]
    else:
        authors_display = f"{names[0]} et al."

    metadata = {
        "Title": result.title or "Unknown",
        "Authors": authors_display,
        "AuthorsAll": [a.name for a in result.authors],
        "Published": result.published.date().isoformat() if result.published else "",
        "Year": result.published.year if result.published else 0,
        "Pages": page_count,
        "entry_id": result.entry_id,
        "AbsURL": result.entry_id,
        "PdfURL": result.pdf_url,
        "Summary": (result.summary or "").strip(),
    }
    return [Document(page_content=text, metadata=metadata)]


def make_llm(provider: str, api_key: str, streaming: bool = True) -> ChatOpenAI:
    cfg = PROVIDERS[provider]
    kwargs: Dict[str, Any] = {
        "model": cfg["model"],
        "api_key": api_key,
        "streaming": streaming,
        "stream_usage": True,        # langchain-openai >=0.2: usage on final chunk
        "temperature": 0.3,          # tighter sampling → more reliable JSON
    }
    # Request strict JSON output when supported (OpenAI; Grok ignores or errors).
    if cfg["base_url"] is None:
        kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
    if cfg["base_url"]:
        kwargs["base_url"] = cfg["base_url"]

    try:
        return ChatOpenAI(**kwargs)
    except TypeError:
        # Older langchain-openai doesn't accept stream_usage; retry without.
        kwargs.pop("stream_usage", None)
        return ChatOpenAI(**kwargs)


def estimate_cost(model: str, in_tok: int, out_tok: int) -> float:
    p = PRICING.get(model)
    if not p:
        return 0.0
    return (in_tok * p["input"] + out_tok * p["output"]) / 1_000_000


# ---- JSON parser tolerant of code fences, surrounding prose, bad escapes ---
# Valid JSON string escapes are \\ \" \/ \b \f \n \r \t \uXXXX. Anything else
# (e.g. \( \$ \% from LaTeX) is illegal. LLMs sometimes emit those — strip the
# stray backslash before retrying.
_INVALID_ESC_RE = re.compile(r'\\([^\\"/bfnrtu])')


def parse_carousel_json(text: str) -> Dict[str, Any]:
    """Tolerant JSON parser. Tries several common LLM-mistake fixes before giving up."""
    s = text.strip()
    fenced = re.match(r"^```(?:json)?\s*\n(.*?)\n```\s*$", s, re.DOTALL)
    if fenced:
        s = fenced.group(1)
    first, last = s.find("{"), s.rfind("}")
    if first != -1 and last != -1 and last > first:
        s = s[first : last + 1]

    def _fix_escapes(t: str) -> str:
        # Double up invalid backslash escapes (\(, \$, \%, ...) so JSON parses
        # AND the backslash survives in the value (needed for LaTeX).
        return _INVALID_ESC_RE.sub(r"\\\\\1", t)

    def _strip_trailing_commas(t: str) -> str:
        return re.sub(r",(\s*[\]}])", r"\1", t)

    candidates = [
        s,
        _fix_escapes(s),
        _strip_trailing_commas(s),
        _strip_trailing_commas(_fix_escapes(s)),
    ]
    last_err: Optional[Exception] = None
    for cand in candidates:
        try:
            return json.loads(cand, strict=False)
        except json.JSONDecodeError as e:
            last_err = e
    raise last_err if last_err else json.JSONDecodeError("no candidate", s, 0)


# ---- HTML renderers --------------------------------------------------------
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)


def escape_with_bold(text: str) -> str:
    """HTML-escape, then convert `**term**` → `<strong>term</strong>`."""
    escaped = html.escape(text)
    return _BOLD_RE.sub(r"<strong>\1</strong>", escaped)


def make_bibtex(arxiv_id: str, title: str, authors_all: List[str], year: int, abs_url: str) -> str:
    """Build an arXiv-style BibTeX entry. Zotero auto-imports from a .bib download."""
    first_surname = (authors_all[0].split()[-1] if authors_all else "anon").lower()
    safe = re.sub(r"\W", "", f"{first_surname}{year}{arxiv_id.replace('.', '')}")
    authors_joined = " and ".join(authors_all) if authors_all else "Unknown"
    return (
        f"@article{{{safe},\n"
        f"  title         = {{{title}}},\n"
        f"  author        = {{{authors_joined}}},\n"
        f"  year          = {{{year or ''}}},\n"
        f"  eprint        = {{{arxiv_id}}},\n"
        f"  archivePrefix = {{arXiv}},\n"
        f"  primaryClass  = {{cs.LG}},\n"
        f"  url           = {{{abs_url}}}\n"
        f"}}\n"
    )


def render_actions_html(pdf_url: str, abs_url: str, bibtex: str, arxiv_id: str) -> str:
    bib_b64 = base64.b64encode(bibtex.encode("utf-8")).decode("ascii")
    bib_name = f"{arxiv_id.replace('/', '_')}.bib"
    return (
        '<div class="action-row">'
        '<button type="button" class="action-btn" onclick="exportCardsToPng()">'
        '<span class="action-icon">📥</span> Export as PNG'
        '</button>'
        f'<a class="action-btn" href="{html.escape(pdf_url)}" target="_blank" rel="noopener">'
        '<span class="action-icon">📄</span> Open PDF'
        '</a>'
        f'<a class="action-btn" href="{html.escape(abs_url)}" target="_blank" rel="noopener">'
        '<span class="action-icon">🔗</span> ArXiv Page'
        '</a>'
        f'<a class="action-btn" href="data:application/x-bibtex;base64,{bib_b64}" '
        f'download="{html.escape(bib_name)}">'
        '<span class="action-icon">📚</span> Export to Zotero (.bib)'
        '</a>'
        '</div>'
    )

def render_paper_header_html(
    title: str,
    authors: str,
    published: str,
    pages: int,
    truncated: bool,
) -> str:
    parts = [
        '<div class="paper-meta">',
        f'<h2 class="paper-title">{html.escape(title)}</h2>',
        f'<p><strong>Authors:</strong> {html.escape(authors)}</p>',
    ]
    if published:
        parts.append(f'<p><strong>Published:</strong> {html.escape(published)}</p>')
    if pages:
        parts.append(f'<p><strong>Pages scanned:</strong> {pages}</p>')
    if truncated:
        parts.append(
            f'<p class="warn">⚠️ Paper truncated to {MAX_PAPER_CHARS:,} chars.</p>'
        )
    parts.append("</div>")
    return "".join(parts)


def render_cards_html(
    header_html: str,
    hook: str,
    slides: List[Dict[str, Any]],
    actions_html: str = "",
) -> str:
    parts = [header_html]
    if actions_html:
        parts.append(actions_html)
    if hook:
        parts.append(f'<div class="hook">{html.escape(hook)}</div>')
    parts.append('<div class="card-grid">')
    for slide in slides:
        emoji = html.escape(str(slide.get("emoji", "🎠")))
        title = html.escape(str(slide.get("title", "Slide")))
        body = str(slide.get("body", "")).strip()
        bullets = slide.get("bullets") or []
        if not isinstance(bullets, list):
            bullets = []

        card_inner = [
            f'<div class="card-emoji">{emoji}</div>',
            f'<h3>{title}</h3>',
        ]
        if body:
            card_inner.append(f'<p>{escape_with_bold(body)}</p>')
        if bullets:
            items = "".join(
                f'<li>{escape_with_bold(str(b).strip())}</li>'
                for b in bullets
                if str(b).strip()
            )
            if items:
                card_inner.append(f'<ul class="card-bullets">{items}</ul>')

        parts.append(f'<div class="card">{"".join(card_inner)}</div>')
    parts.append("</div>")
    return "".join(parts)


def render_cost_html(model: str = "—", in_tok: int = 0, out_tok: int = 0) -> str:
    cost = estimate_cost(model, in_tok, out_tok)
    return (
        '<div class="cost-card">'
        '<div class="cost-title">🪙 API Usage</div>'
        f'<div class="cost-row"><span>Model</span><span>{html.escape(model or "—")}</span></div>'
        f'<div class="cost-row"><span>Input</span><span>{in_tok:,} tok</span></div>'
        f'<div class="cost-row"><span>Output</span><span>{out_tok:,} tok</span></div>'
        f'<div class="cost-row cost-total"><span>Cost</span><span>${cost:.4f}</span></div>'
        '</div>'
    )


def render_placeholder_html(msg: str) -> str:
    return f'<div class="placeholder">{html.escape(msg)}</div>'


def render_error_html(header_html: str, message: str, raw: str = "") -> str:
    block = (
        f'<div class="error-card"><strong>⚠️ {html.escape(message)}</strong>'
        + (f'<pre>{html.escape(raw[:4000])}</pre>' if raw else "")
        + "</div>"
    )
    return header_html + block


# -----------------------------------------------------------------------------
# Main callback — yields (status_md, output_html, cost_html).
# -----------------------------------------------------------------------------
def generate_carousel(url: str, provider: str):
    model_name = PROVIDERS[provider]["model"]
    initial_cost = render_cost_html(model_name)

    aid = extract_arxiv_id(url)
    if not aid:
        yield (
            "❌ **Invalid URL.** Could not find an ArXiv ID.",
            render_placeholder_html("Paste a valid ArXiv URL or ID."),
            initial_cost,
        )
        return

    env_key = PROVIDERS[provider]["env_key"]
    api_key = os.getenv(env_key, "").strip()
    if not api_key:
        yield (
            f"❌ **Missing `{env_key}` in environment.** Set it in `.env` or your shell.",
            render_placeholder_html(f"Set {env_key} in .env"),
            initial_cost,
        )
        return

    t_start = time.monotonic()

    # ---- Fetch with 429 retry + live elapsed timer (threaded) --------------
    docs: Optional[List[Document]] = None
    last_err: Optional[Exception] = None
    for attempt in range(MAX_FETCH_RETRIES):
        t_attempt = time.monotonic()
        holder: dict = {"docs": None, "error": None}

        def _do_fetch():
            try:
                holder["docs"] = load_arxiv_paper(aid)
            except Exception as e:
                holder["error"] = e

        worker = threading.Thread(target=_do_fetch, daemon=True)
        worker.start()
        while worker.is_alive():
            elapsed = time.monotonic() - t_attempt
            yield (
                f"⏳ Fetching paper `{aid}`… "
                f"(attempt {attempt + 1}/{MAX_FETCH_RETRIES}, {elapsed:.1f}s)",
                render_placeholder_html("Fetching paper…"),
                initial_cost,
            )
            worker.join(timeout=0.5)

        err = holder["error"]
        if err is None:
            docs = holder["docs"]
            break

        last_err = err
        if isinstance(err, urllib.error.HTTPError) and err.code == 429 and attempt < MAX_FETCH_RETRIES - 1:
            wait = (attempt + 1) * 5
            for s in range(wait, 0, -1):
                yield (
                    f"⏳ Rate-limited by arxiv (HTTP 429). "
                    f"Retrying in **{s}s** (attempt {attempt + 2}/{MAX_FETCH_RETRIES})…",
                    render_placeholder_html("Waiting on arxiv rate limit…"),
                    initial_cost,
                )
                time.sleep(1)
            continue

        msg = (
            f"HTTP {err.code}: {err.reason}"
            if isinstance(err, urllib.error.HTTPError)
            else f"{type(err).__name__}: {err}"
        )
        yield (
            f"❌ **Fetch failed** ({time.monotonic() - t_start:.1f}s): `{msg}`",
            render_placeholder_html(f"Fetch failed: {msg}"),
            initial_cost,
        )
        return

    if docs is None:
        yield (
            f"❌ **Fetch failed after {MAX_FETCH_RETRIES} retries:** `{last_err}`",
            render_placeholder_html("Fetch failed after retries."),
            initial_cost,
        )
        return
    if not docs:
        yield (
            f"❌ **Paper `{aid}` not found.**",
            render_placeholder_html(f"Paper {aid} not found."),
            initial_cost,
        )
        return

    fetch_dt = time.monotonic() - t_start

    meta = docs[0].metadata
    title = meta.get("Title", "Unknown")
    authors = meta.get("Authors", "Unknown")
    authors_all = meta.get("AuthorsAll") or []
    published = meta.get("Published", "")
    year = int(meta.get("Year", 0) or 0)
    pages = int(meta.get("Pages", 0) or 0)
    pdf_url = meta.get("PdfURL", "")
    abs_url = meta.get("AbsURL", "")
    paper_text = docs[0].page_content[:MAX_PAPER_CHARS]
    truncated = len(docs[0].page_content) > MAX_PAPER_CHARS
    header_html = render_paper_header_html(title, authors, published, pages, truncated)
    bibtex = make_bibtex(aid, title, authors_all, year, abs_url)
    actions_html = render_actions_html(pdf_url, abs_url, bibtex, aid)

    yield (
        f"✅ Fetched in {fetch_dt:.1f}s. Generating carousel…",
        header_html + render_placeholder_html("Generating slides…"),
        initial_cost,
    )

    try:
        llm = make_llm(provider, api_key, streaming=True)
    except Exception as e:
        yield (
            f"❌ **LLM init failed:** `{type(e).__name__}: {e}`",
            render_error_html(header_html, f"LLM init failed: {e}"),
            initial_cost,
        )
        return

    messages = [
        SystemMessage(content=CAROUSEL_PROMPT),
        HumanMessage(content=f"PAPER TEXT:\n\n{paper_text}"),
    ]

    # ---- Stream LLM tokens via background thread + live timer + token count
    t_gen = time.monotonic()
    chunk_q: queue.Queue = queue.Queue()

    def _stream_llm():
        try:
            for c in llm.stream(messages):
                chunk_q.put(("chunk", c))
        except Exception as exc:
            chunk_q.put(("error", exc))
        finally:
            chunk_q.put(("done", None))

    threading.Thread(target=_stream_llm, daemon=True).start()

    buffer = ""
    in_tok = 0
    out_tok = 0
    gen_error: Optional[Exception] = None

    while True:
        try:
            kind, val = chunk_q.get(timeout=0.3)
        except queue.Empty:
            elapsed = time.monotonic() - t_gen
            yield (
                f"✍️ Generating… {elapsed:.1f}s ({len(buffer):,} chars)",
                header_html + render_placeholder_html(f"Generating slides… ({elapsed:.1f}s)"),
                render_cost_html(model_name, in_tok, out_tok),
            )
            continue
        if kind == "chunk":
            content = getattr(val, "content", "") or ""
            buffer += content
            usage = getattr(val, "usage_metadata", None)
            if usage:
                in_tok = usage.get("input_tokens", in_tok) or in_tok
                out_tok = usage.get("output_tokens", out_tok) or out_tok
            elapsed = time.monotonic() - t_gen
            yield (
                f"✍️ Generating… {elapsed:.1f}s ({len(buffer):,} chars)",
                header_html + render_placeholder_html(f"Generating slides… ({elapsed:.1f}s)"),
                render_cost_html(model_name, in_tok, out_tok),
            )
        elif kind == "error":
            gen_error = val
            break
        elif kind == "done":
            break

    if gen_error is not None:
        elapsed = time.monotonic() - t_gen
        yield (
            f"❌ **Generation error** after {elapsed:.1f}s: "
            f"`{type(gen_error).__name__}: {gen_error}`",
            render_error_html(header_html, f"Generation error: {gen_error}", buffer),
            render_cost_html(model_name, in_tok, out_tok),
        )
        return

    # ---- Parse JSON, render cards ------------------------------------------
    try:
        data = parse_carousel_json(buffer)
    except Exception as e:
        elapsed = time.monotonic() - t_gen
        yield (
            f"❌ **Failed to parse JSON** after {elapsed:.1f}s: `{type(e).__name__}: {e}`",
            render_error_html(
                header_html,
                "Model did not return valid JSON. Raw output below.",
                buffer,
            ),
            render_cost_html(model_name, in_tok, out_tok),
        )
        return

    hook = str(data.get("hook", "")).strip()
    slides = data.get("slides") or []
    if not isinstance(slides, list):
        slides = []

    cards_html = render_cards_html(header_html, hook, slides, actions_html=actions_html)
    total = time.monotonic() - t_start
    gen_dt = time.monotonic() - t_gen

    yield (
        f"✅ Done — `{aid}` in {total:.1f}s "
        f"(fetch {fetch_dt:.1f}s + gen {gen_dt:.1f}s) · {len(slides)} cards",
        cards_html,
        render_cost_html(model_name, in_tok, out_tok),
    )


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
CSS = """
.gradio-container {
    max-width: 100% !important;
    width: 100% !important;
    margin: 0 auto !important;
    padding: 1em 2em !important;
}

/* Force a clean, highly-legible font across the whole app, including injected HTML. */
.gradio-container,
.gradio-container *,
#output, #output * {
    font-family: "Inter", system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
}

#topbar {
    align-items: flex-start;
    padding: 0.5em 0;
    gap: 1em;
    flex-wrap: nowrap !important;
}
#topbar .prose h1 { margin: 0; font-size: 1.6em; }

/* Right-side stack: model dropdown sits above the API-usage card, both same width. */
#right-stack {
    max-width: 320px;
    min-width: 240px;
    margin-left: auto;
    display: flex;
    flex-direction: column;
    gap: 0.6em;
}
#right-stack > * { width: 100%; }
#provider-slot { width: 100%; }
/* Allow flex children to shrink below their intrinsic content width. */
#topbar > * { min-width: 0; }

/* Make room for the dropdown caret on the right edge of the value box. */
#provider-slot input,
#provider-slot select,
#provider-slot .wrap-inner,
#provider-slot [role="combobox"] {
    padding-right: 32px !important;
}

/* ---- Cost panel (top-right sidebar chip) ---- */
.cost-card {
    background: #fff;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 0.7em 0.9em;
    font-size: 0.9em;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04);
}
.cost-title {
    font-weight: 600;
    color: #6366f1;
    margin-bottom: 0.4em;
    font-size: 0.95em;
}
.cost-row {
    display: flex;
    justify-content: space-between;
    gap: 1em;
    padding: 0.15em 0;
    color: #4b5563;
}
.cost-row.cost-total {
    border-top: 1px solid #e5e7eb;
    margin-top: 0.3em;
    padding-top: 0.4em;
    color: #111827;
    font-weight: 600;
}

/* ---- Centered Google-style URL bar ---- */
#center-stack {
    display: flex; flex-direction: column; align-items: center;
    margin: 0.5em auto 1em auto; max-width: 1100px; width: 100%;
}
#tagline { color: #555; margin-bottom: 1.4em; font-size: 1.5em; text-align: center; font-weight: 500; }
#url-box textarea {
    font-size: 1.15em !important;
    padding: 14px 18px !important;
    border-radius: 28px !important;
}
#url-box { width: 100%; }
#generate-row { justify-content: center; margin-top: 1em; gap: 0.6em; }

/* ---- Status + output ---- */
#status { max-width: 1100px; margin: 1.5em auto 0.5em auto; text-align: center; }
#output { max-width: 1100px; margin: 0 auto 3em auto; }

/* ---- Paper meta ---- */
.paper-meta { margin-bottom: 1em; }
.paper-title { margin: 0 0 0.4em 0; font-size: 1.4em; line-height: 1.25; }
.paper-meta p { margin: 0.15em 0; color: #4b5563; }
.warn { color: #b45309 !important; font-style: italic; }

/* ---- Action buttons (Export / PDF / Zotero) ---- */
.action-row {
    display: flex;
    gap: 0.6em;
    flex-wrap: wrap;
    margin: 1.2em 0 1em 0;
}
.action-btn {
    display: inline-flex;
    align-items: center;
    gap: 0.5em;
    padding: 0.6em 1.05em;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    background: #fff;
    color: #1f2937;
    font-size: 1em;
    font-weight: 500;
    cursor: pointer;
    text-decoration: none;
    transition: background 0.15s, border-color 0.15s, color 0.15s, transform 0.15s;
    font-family: inherit;
}
.action-btn:hover {
    background: #f3f4f6;
    border-color: #6366f1;
    color: #4f46e5;
    transform: translateY(-1px);
}
.action-btn:active { transform: translateY(0); }
.action-icon { font-size: 1.1em; line-height: 1; }

/* ---- Hook ---- */
.hook {
    font-size: 1.6em;
    line-height: 1.4;
    color: #1f2937;
    font-style: italic;
    font-weight: 500;
    margin: 1.5em 0 1.4em 0;
    padding: 1.1em 1.3em;
    background: #f9fafb;
    border-left: 4px solid #6366f1;
    border-radius: 0 10px 10px 0;
}

/* ---- Card grid ---- */
.card-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 16px;
    margin-top: 0.5em;
}
.card {
    background: #fff;
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    padding: 18px 18px 20px 18px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.04);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 18px rgba(0,0,0,0.08);
}
.card-emoji { font-size: 2.2em; line-height: 1; margin-bottom: 0.2em; }
.card h3 {
    margin: 0.1em 0 0.45em 0;
    font-size: 1.55em;
    font-weight: 700;
    color: #0f172a;
    letter-spacing: -0.01em;
}
.card p {
    margin: 0 0 0.4em 0;
    line-height: 1.55;
    color: #374151;
    font-size: 1.1em;
}
.card p strong, .card-bullets strong {
    color: #4f46e5;
    font-weight: 600;
}
.card-bullets {
    margin: 0.1em 0 0 0;
    padding-left: 0;
    list-style: none;
    color: #374151;
    font-size: 1.05em;
    line-height: 1.5;
}
.card-bullets li {
    position: relative;
    padding-left: 1em;        /* room for the bullet glyph */
    margin: 0.15em 0;
}
.card-bullets li::before {
    content: "○";
    color: #6366f1;
    position: absolute;
    left: 0;                  /* glyph hugs the left edge */
    top: 0;
    font-size: 0.85em;
    line-height: 1.7;
}

/* ---- Placeholder / error ---- */
.placeholder {
    color: #9ca3af;
    font-style: italic;
    text-align: center;
    padding: 2em 0;
}
.error-card {
    background: #fef2f2;
    border: 1px solid #fecaca;
    color: #991b1b;
    padding: 1em 1.2em;
    border-radius: 10px;
    margin-top: 1em;
}
.error-card pre {
    margin-top: 0.6em;
    white-space: pre-wrap;
    word-break: break-word;
    background: #fff;
    padding: 0.6em;
    border-radius: 6px;
    font-size: 0.85em;
    max-height: 300px;
    overflow: auto;
}
"""


MATHJAX_HEAD = r"""
<script>
window.MathJax = {
  tex: {
    inlineMath:  [['$', '$'], ['\\(', '\\)']],
    displayMath: [['$$', '$$'], ['\\[', '\\]']]
  },
  options: {
    skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
  }
};
</script>
<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<script src="https://cdn.jsdelivr.net/npm/html2canvas@1.4.1/dist/html2canvas.min.js"></script>
<script>
(function() {
  function setup() {
    var t = document.getElementById('output');
    if (!t) { setTimeout(setup, 200); return; }
    var pending = null;
    new MutationObserver(function() {
      if (pending) clearTimeout(pending);
      pending = setTimeout(function() {
        if (window.MathJax && window.MathJax.typesetPromise) {
          window.MathJax.typesetPromise([t]).catch(function() {});
        }
      }, 80);
    }).observe(t, { childList: true, subtree: true, characterData: true });
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', setup);
  } else {
    setup();
  }
})();

// ---- Export rendered cards to PNG via html2canvas -----------------------
window.exportCardsToPng = async function() {
  var target = document.getElementById('output');
  if (!target || !window.html2canvas) {
    alert('Export library not loaded yet — please retry in a moment.');
    return;
  }
  // Hide the action row during capture so the PNG is clean.
  var actions = target.querySelector('.action-row');
  var prev = actions ? actions.style.display : null;
  if (actions) actions.style.display = 'none';
  try {
    var canvas = await html2canvas(target, {
      scale: 2,
      backgroundColor: '#ffffff',
      useCORS: true,
      logging: false
    });
    var link = document.createElement('a');
    link.download = 'arxiv-carousel.png';
    link.href = canvas.toDataURL('image/png');
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  } catch (err) {
    console.error(err);
    alert('Export failed: ' + err.message);
  } finally {
    if (actions) actions.style.display = prev;
  }
};
</script>
"""


def build_ui() -> gr.Blocks:
    initial_cost = render_cost_html(list(PROVIDERS.values())[0]["model"])

    with gr.Blocks(title="ArXiv Carousel") as demo:
        # ---- Top bar: title left, [Model + API Usage] stacked top-right ----
        with gr.Row(elem_id="topbar"):
            with gr.Column(scale=4):
                gr.Markdown("# 📚 ArXiv Carousel")
            with gr.Column(scale=1, min_width=260, elem_id="right-stack"):
                provider = gr.Dropdown(
                    choices=list(PROVIDERS.keys()),
                    value=list(PROVIDERS.keys())[0],
                    label="Model",
                    container=True,
                    elem_id="provider-slot",
                    filterable=False,
                    allow_custom_value=False,
                    interactive=True,
                )
                cost = gr.HTML(value=initial_cost, elem_id="cost-panel")

        # ---- Centered Google-style URL bar + button ----
        with gr.Column(elem_id="center-stack"):
            gr.Markdown(
                "Paste an ArXiv link → get a captivating bite-sized summary.",
                elem_id="tagline",
            )
            url = gr.Textbox(
                placeholder="https://arxiv.org/abs/2305.10601",
                show_label=False,
                container=False,
                elem_id="url-box",
            )
            with gr.Row(elem_id="generate-row"):
                generate_btn = gr.Button("✨ Generate Carousel", variant="primary")

        status = gr.Markdown("_Ready._", elem_id="status")
        output = gr.HTML(value="", elem_id="output")

        # Update cost panel live when provider changes.
        def _on_provider_change(p):
            return render_cost_html(PROVIDERS[p]["model"])

        provider.change(_on_provider_change, inputs=[provider], outputs=[cost])

        for trigger in (url.submit, generate_btn.click):
            trigger(
                generate_carousel,
                inputs=[url, provider],
                outputs=[status, output, cost],
            )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    # Streaming generators require a queue. HF Spaces also expects 0.0.0.0 + $PORT.
    demo.queue(default_concurrency_limit=2, max_size=20)
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        ssr_mode=False,  # SSR breaks streaming generators on HF Spaces
        theme=gr.themes.Soft(
            text_size=gr.themes.sizes.text_lg,
            font=[
                gr.themes.GoogleFont("Inter"),
                "system-ui",
                "-apple-system",
                "Segoe UI",
                "Helvetica Neue",
                "Arial",
                "sans-serif",
            ],
        ),
        css=CSS,
        head=MATHJAX_HEAD,
    )
