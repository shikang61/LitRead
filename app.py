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
import os
import queue
import re
import threading
import time
import warnings
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
load_dotenv(override=False)

warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain_community.*")
warnings.filterwarnings("ignore", message=".*langchain-community.*sunset.*")

import fitz  # pymupdf
import gradio as gr
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

import core
from core import (
    CACHE_DIR,
    MODEL_CONTEXT,
    PROVIDERS,
    cache_get,
    cache_key,
    cache_put,
    estimate_cost,
    extract_arxiv_id,
    make_llm,
    parse_json,
    render_cost_html,
    strip_references,
)

import annotate

# -----------------------------------------------------------------------------
# Config (carousel-specific; shared config lives in core.py)
# -----------------------------------------------------------------------------
MAX_PAPER_CHARS = 200_000
MAX_FETCH_RETRIES = 4

# Bump this whenever CAROUSEL_PROMPT changes — invalidates cached carousels.
PROMPT_VERSION = "v4"

# -----------------------------------------------------------------------------
# System prompt — strict JSON output.
# -----------------------------------------------------------------------------
CAROUSEL_PROMPT = """You are a clear, plain-spoken explainer of research papers for a curious non-expert.

GOAL: After reading all cards, the reader should understand the paper's key learnings — the
problem, the core idea, how it works, the main result, and what's new about it — well enough
that opening the actual paper feels like a deep-dive, not a cold start. A reader who only sees
the cards should still walk away knowing what the paper did and why it matters.

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

ASPECTS to cover, in this order (skip any absent from the paper):
  🎯 Motivation, 🌍 Context, 🌟 Why It Matters,
  💡 Proposed Solution, 🏆 Key Results,
  📊 Comparison to Existing Methods, 🔮 Future Work

"💡 Proposed Solution" should cover both the high-level idea AND a short sketch of how it
works mechanically — the reader does not need a separate "How It Works" card.

"🌟 Why It Matters" should explain the paper's contribution and impact on the field —
what the community gains (new capability, new benchmark, new tool, settled question, opened
direction). Stay grounded in the paper's own framing of its contribution; do not project
future impact the authors did not claim.

ACCURACY (highest priority — overrides every other rule):
- Use ONLY information stated or directly supported by the paper text. Do NOT invent results,
  manufacture motivation, or fill gaps from prior knowledge. If the paper does not support it,
  do not write it.
- LIBERAL paraphrasing is encouraged — rewrite freely into plain, conversational English to
  maximise readability. You may restructure sentences, swap analogies for jargon, change voice,
  and reorder clauses. The goal is digestibility, not transcription.
- HARD CONSTRAINTS that paraphrasing must NOT cross:
  - Numbers, model names, dataset names, metric names, and named methods stay VERBATIM.
    Never round percentages, never swap units, never rename a model or dataset.
  - Hedges and scope qualifiers must survive in some form ("on small models", "for English
    only", "preliminary", "in our setting"). You may rephrase them, but never delete them.
  - Don't upgrade tentative findings into definitive ones. "Suggests" ≠ "proves".
  - Don't invent contributions, baselines, or impact the authors didn't claim.
- Useful analogies are allowed when they illuminate a concept — but flag them clearly as
  analogies (e.g. "think of it like…") so the reader doesn't mistake them for paper content.
- If an aspect (e.g. "Future Work") is not discussed in the paper, OMIT that slide entirely.
  Do not fabricate plausible-sounding content to fill a slot.

LANGUAGE (layman-first, jargon only when load-bearing):
- Write as if explaining to a curious friend over coffee, not in an academic abstract.
- **Active voice, always.** Subject does the verb. Rewrite passive constructions:
  ✅ "The model learns from its own mistakes"
  ❌ "Mistakes are learned from by the model"
  ✅ "**LoRA** cuts training cost by **3x**"
  ❌ "Training cost is cut by **3x** through the use of **LoRA**"
- Default to plain English. Replace jargon with everyday words whenever the meaning survives
  (e.g. "the model learns from its own mistakes" beats "self-supervised reward refinement").
- Keep technical terms ONLY when (a) they are the paper's central named contribution
  (method/model/dataset names), (b) removing them would distort the meaning, or
  (c) a reader looking the paper up needs them to find related work.
- When a technical term is unavoidable, define it inline in <=6 words on first use, e.g.
  "**RLHF** (training from human ratings)".
- Avoid: "novel", "leverage", "robust", "state-of-the-art", "paradigm", "framework" as filler.
  Use only when the paper itself emphasises them.

CARD STRUCTURE:
- Each `body`: ONE short sentence (max ~20 words) — a framing line, not a full explanation.
- Each `bullets`: 2–4 punchy bullets (max ~12 words each). Use bullets to carry the substance.
- `bullets` is optional — omit if a single short sentence in `body` covers the slide cleanly.

FORMATTING:
- **MANDATORY bold** — every `body` and every bullet MUST wrap at least 4 key terms in markdown bold
  using DOUBLE asterisks: `**term**`. Pick the most important concept, method name, dataset,
  model, or metric in each line. Do NOT bold whole phrases — single terms only.
  Example bullet: `"Beats **GPT-4** by **+15.2 pp** on **MMLU**"`.
- Output ONLY the JSON object — no prose before/after, no markdown fences.
- You MAY use LaTeX for math: `$x = y^2$` for inline, `$$E = mc^2$$` for display. Keep equations short.
  When emitting LaTeX in JSON strings, escape backslashes as `\\\\` (e.g. `\\\\frac{a}{b}`).
"""


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def load_arxiv_paper(arxiv_id: str) -> List[Document]:
    """Fetch the paper and return its full text as a single Document. The
    network fetch + metadata come from core.fetch_pdf; here we add the text body."""
    pdf_bytes, metadata = core.fetch_pdf(arxiv_id)
    if pdf_bytes is None:
        return []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf:
        # "text" extraction mode is fastest; skips layout/HTML reconstruction.
        text = "\n\n".join(page.get_text("text") for page in pdf)
    # References list eats ~10-25% of tokens and the LLM can't summarise from it.
    text = strip_references(text)
    return [Document(page_content=text, metadata=metadata)]


# ---- HTML renderers --------------------------------------------------------
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)


def escape_with_bold(text: str) -> str:
    """HTML-escape, then convert `**term**` → `<strong>term</strong>`."""
    parts: List[str] = []
    last = 0
    for m in _BOLD_RE.finditer(text):
        parts.append(html.escape(text[last:m.start()]))
        parts.append(f'<strong>{html.escape(m.group(1))}</strong>')
        last = m.end()
    parts.append(html.escape(text[last:]))
    return "".join(parts)


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
    parts.append("</div>")
    if truncated:
        parts.append(
            f'<div class="trunc-banner">'
            f'<span class="trunc-icon">⚠️</span>'
            f'<span><strong>Paper truncated to {MAX_PAPER_CHARS:,} characters</strong> '
            f'before sending to the model. Tail content (later sections) was not summarised.'
            f'</span></div>'
        )
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
        parts.append(f'<div class="hook">{escape_with_bold(hook)}</div>')
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


def render_placeholder_html(msg: str) -> str:
    return f'<div class="placeholder">{html.escape(msg)}</div>'


def render_error_html(header_html: str, message: str, raw: str = "") -> str:
    block = (
        f'<div class="error-card"><strong>⚠️ {html.escape(message)}</strong>'
        + (f'<pre>{html.escape(raw[:4000])}</pre>' if raw else "")
        + "</div>"
    )
    return header_html + block


def render_recent_push(aid: str, title: str) -> str:
    """Hidden <img> whose onerror fires once the DOM mounts the cards. Pushes
    (arxiv_id, title) into the browser's localStorage via window.LITREAD_RECENT
    so the recent-papers chips update without a server round-trip."""
    safe_t = (title or "").replace("\\", "\\\\").replace("'", "\\'")
    safe_a = (aid or "").replace("\\", "\\\\").replace("'", "\\'")
    return (
        f'<img src="data:," class="recent-push" alt="" '
        f'onerror="if(window.LITREAD_RECENT) window.LITREAD_RECENT.push(\''
        f'{safe_a}\',\'{safe_t}\')">'
    )


# -----------------------------------------------------------------------------
# Main callback — yields (status_md, output_html, cost_html).
# -----------------------------------------------------------------------------
def generate_carousel(url: str, provider: str, force: bool = False):
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

    # ---- Cache hit short-circuits both arxiv fetch AND the LLM call. -------
    ckey = cache_key(aid, provider, model_name, PROMPT_VERSION)
    cached = cache_get(ckey)
    if cached and not force:
        meta_c = cached.get("meta", {})
        title_c = meta_c.get("title", "Unknown")
        bibtex_c = make_bibtex(
            aid, title_c, meta_c.get("authors_all") or [],
            int(meta_c.get("year", 0) or 0), meta_c.get("abs_url", ""),
        )
        header_c = render_paper_header_html(
            title_c, meta_c.get("authors", "Unknown"),
            meta_c.get("published", ""), int(meta_c.get("pages", 0) or 0),
            bool(meta_c.get("truncated", False)),
        )
        actions_c = render_actions_html(
            meta_c.get("pdf_url", ""), meta_c.get("abs_url", ""), bibtex_c, aid,
        )
        cards_c = render_cards_html(
            header_c, cached.get("hook", ""), cached.get("slides", []),
            actions_html=actions_c,
        ) + render_recent_push(aid, title_c)
        yield (
            f"⚡ Cached — `{aid}` served instantly, 0 tokens used.",
            cards_c,
            render_cost_html(model_name, 0, 0),
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
        # arxiv lib raises its own HTTPError class, urllib raises another — duck-type the
        # status so both metadata-API and PDF-fetch transient failures hit the retry path.
        status = getattr(err, "code", None) or getattr(err, "status", None)
        retryable = status == 429 or (isinstance(status, int) and 500 <= status < 600)
        if retryable and attempt < MAX_FETCH_RETRIES - 1:
            wait = (attempt + 1) * 5
            label = "Rate-limited" if status == 429 else f"ArXiv {status}"
            for s in range(wait, 0, -1):
                yield (
                    f"⏳ {label} (HTTP {status}). "
                    f"Retrying in **{s}s** (attempt {attempt + 2}/{MAX_FETCH_RETRIES})…",
                    render_placeholder_html(f"Waiting on arxiv (HTTP {status})…"),
                    initial_cost,
                )
                time.sleep(1)
            continue

        msg = (
            f"HTTP {status}: {getattr(err, 'reason', '') or err}"
            if status
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
        data = parse_json(buffer)
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

    cards_html = (
        render_cards_html(header_html, hook, slides, actions_html=actions_html)
        + render_recent_push(aid, title)
    )
    total = time.monotonic() - t_start
    gen_dt = time.monotonic() - t_gen

    # Persist to cache so subsequent runs on this paper cost 0 tokens.
    cache_put(ckey, {
        "hook": hook,
        "slides": slides,
        "meta": {
            "title": title, "authors": authors, "authors_all": authors_all,
            "published": published, "year": year, "pages": pages,
            "pdf_url": pdf_url, "abs_url": abs_url, "truncated": truncated,
        },
    })

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
    font-size: 15px !important;   /* base text 1pt smaller; em-based content scales down */
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
#force-row { justify-content: center; margin-top: 0.5em; }
#force-toggle { color: #6b7280; font-size: 0.9em; }

/* ---- Status + output ---- */
#status {
    width: fit-content;
    max-width: 1100px;
    margin: 1.5em auto 0.5em auto;
    padding: 0.5em 1.2em 0.7em 1.2em;
    text-align: center;
}
#status .prose,
#status .prose * {
    margin: 0 !important;
    padding: 0 !important;
    line-height: 1.2 !important;
}
#output { max-width: 1100px; margin: 0 auto 3em auto; }

/* ---- Paper meta ---- */
.paper-meta { margin-bottom: 1em; }
.paper-title { margin: 0 0 0.4em 0; font-size: 1.4em; line-height: 1.25; }
.paper-meta p { margin: 0.15em 0; color: #4b5563; }
.warn { color: #b45309 !important; font-style: italic; }

/* ---- Truncation banner ---- */
.trunc-banner {
    display: flex; align-items: flex-start; gap: 0.6em;
    background: #fffbeb;
    border: 1px solid #fcd34d;
    color: #92400e;
    padding: 0.7em 0.95em;
    border-radius: 10px;
    margin: 0.6em 0 1em 0;
    font-size: 0.95em;
    line-height: 1.4;
}
.trunc-icon { font-size: 1.15em; line-height: 1; }

/* ---- Cost panel meter ---- */
.cost-meter {
    height: 4px;
    background: #eef2ff;
    border-radius: 3px;
    overflow: hidden;
    margin: 0.1em 0 0.4em 0;
}
.cost-bar {
    height: 100%;
    background: #6366f1;
    transition: width 0.3s ease;
}
.cost-bar-amber { background: #f59e0b; }
.cost-bar-red   { background: #ef4444; }

/* ---- Recent papers (collapsible left rail) ---- */
:root { --lr-gutter: 0px; }
#recent-papers {
    position: fixed;
    top: 116px;                 /* sit below the LitRead title row */
    left: 0;
    width: 240px;
    max-height: calc(100vh - 136px);
    overflow-y: auto;
    overflow-x: hidden;
    padding: 1em 0.9em 1em 1em;
    z-index: 20;
    /* Solid panel so the rail reads as a sidebar and never shows content through it. */
    background: #f9fafb;
    border-right: 1px solid #e9ebf0;
    border-radius: 0 12px 12px 0;
    box-shadow: 1px 0 4px rgba(0,0,0,0.03);
    transition: width 0.2s ease, padding 0.2s ease;
}
/* Collapse / expand chevron (glyph supplied by ::before per state). */
.recent-toggle {
    position: absolute;
    top: 0.55em; right: 0.5em;
    width: 26px; height: 26px;
    border: none; background: none; cursor: pointer;
    color: #9ca3af; font-size: 1.2em; line-height: 1;
    border-radius: 7px; padding: 0;
    display: flex; align-items: center; justify-content: center;
    transition: background 0.15s, color 0.15s;
}
.recent-toggle:hover { background: #eef2ff; color: #6366f1; }
body:not(.lr-rail-collapsed) .recent-toggle::before { content: '‹'; }
body.lr-rail-collapsed .recent-toggle::before { content: '›'; }
.recent-collapsed-icon { display: none; font-size: 1.35em; cursor: pointer; margin-top: 0.3em; text-align: center; }
body.lr-rail-collapsed .recent-collapsed-icon { display: block; }
.recent-header {
    font-size: 0.85em;
    font-weight: 600;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 0.6em;
    padding-left: 0.2em;
    padding-right: 1.8em;   /* clear the toggle button */
}
.recent-empty-msg {
    font-size: 0.8em;
    color: #9ca3af;
    padding: 0.4em 0.2em;
    line-height: 1.4;
}
.recent-list { display: flex; flex-direction: column; gap: 0.25em; }
.recent-item {
    text-align: left;
    background: none;
    border: 1px solid transparent;
    border-radius: 8px;
    padding: 0.55em 0.7em;
    cursor: pointer;
    font-family: inherit;
    width: 100%;
    transition: background 0.15s, border-color 0.15s;
}
.recent-item:hover { background: #eef2ff; border-color: #6366f1; }
.recent-title {
    font-size: 0.9em;
    color: #1f2937;
    font-weight: 500;
    line-height: 1.35;
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
}
.recent-aid {
    font-size: 0.72em;
    color: #9ca3af;
    margin-top: 0.2em;
    font-family: ui-monospace, "SF Mono", Menlo, monospace !important;
}
/* Collapsed: shrink to a thin strip, hide the body, centre the toggle. */
body.lr-rail-collapsed #recent-papers {
    width: 46px !important;
    padding: 0.5em 0 !important;
    background: #f3f4f6;
    border-right: 1px solid #e5e7eb;
    border-radius: 0 10px 10px 0;
    box-shadow: 1px 0 3px rgba(0,0,0,0.04);
}
body.lr-rail-collapsed .recent-body { display: none; }
/* In the thin strip the toggle flows at the top-centre (no fragile offsets). */
body.lr-rail-collapsed .recent-toggle { position: static; margin: 0 auto 0.2em auto; }

/* Reserve a left gutter for the rail and offset the centred content into it so
   the rail can never overlap the search bar/output. Viewport-based + shrink-to-
   fit, so it holds regardless of Gradio's own container padding. */
@media (min-width: 1100px) {
    :root { --lr-gutter: 256px; }                  /* 240 rail + 16 gap */
    body.lr-rail-collapsed { --lr-gutter: 64px; }  /* 48 strip + 16 gap */
    #center-stack, #status, #output, #reader {
        /* Truly centred on the page (equal auto margins). Shrinking the max-width
           by 2x the gutter guarantees the centred block's left edge stays clear
           of the rail, so it centres without ever overlapping. */
        margin-left: auto !important;
        margin-right: auto !important;
        max-width: min(1100px, calc(100vw - var(--lr-gutter) * 2 - 4em)) !important;
        transition: max-width 0.2s ease;
    }
}
/* Hide the rail on narrow screens (no recents UI); content uses full width. */
@media (max-width: 1099px) {
    #recent-papers { display: none; }
}
img.recent-push { width: 0; height: 0; }

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
.action-btn.is-active {
    background: #eef2ff;
    border-color: #6366f1;
    color: #4338ca;
}
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
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin-top: 0.5em;
}
@media (max-width: 820px) { .card-grid { grid-template-columns: repeat(2, 1fr); } }
@media (max-width: 540px) { .card-grid { grid-template-columns: 1fr; } }
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

/* ---- Annotated reader ---- */
#reader { max-width: 1100px; margin: 0 auto 3em auto; }
.lr-header { margin: 0.5em 0 1.2em 0; }
.lr-title { margin: 0 0 0.3em 0; font-size: 1.4em; line-height: 1.25; color: #0f172a; }
.lr-authors { margin: 0; color: #6b7280; }
.lr-row {
    display: flex;
    gap: 18px;
    align-items: stretch;   /* notes column matches the page image height */
    margin-bottom: 2.2em;
}
.lr-pagewrap { flex: 1 1 58%; min-width: 0; }
.lr-page-num {
    font-size: 0.75em; font-weight: 600; color: #9ca3af;
    text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.4em;
}
.lr-page { position: relative; display: block; }
.lr-page-img {
    width: 100%; display: block;
    border: 1px solid #e5e7eb; border-radius: 8px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.06);
}
/* Hidden bridge components the drag JS drives. */
#lr-selection, #lr-summarize { display: none !important; }
/* Transparent layer over each page that captures the rubber-band drag. */
.lr-drawlayer { position: absolute; inset: 0; cursor: crosshair; z-index: 2; }
.lr-rubber {
    position: absolute;
    border: 1.5px dashed #6366f1;
    background: rgba(99,102,241,0.12);
    pointer-events: none;
    z-index: 3;
}
/* Marker drawn at a saved selection. */
.lr-sel {
    position: absolute;
    outline: 2px solid #6366f1;
    outline-offset: 2px;
    border-radius: 4px;
    background: rgba(99,102,241,0.10);
    pointer-events: none;
    box-sizing: border-box;
    transition: box-shadow 0.2s ease, background 0.2s ease;
}
.lr-box-num {
    position: absolute;
    bottom: 100%; left: -4px;     /* sit above the marker, not over the content */
    margin-bottom: 4px;
    width: 20px; height: 20px;
    background: #6366f1; color: #fff;
    border-radius: 50%;
    font-size: 0.7em; font-weight: 700;
    display: flex; align-items: center; justify-content: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.25);
    z-index: 4;
}
.lr-sel.lr-flash {
    box-shadow: 0 0 0 4px rgba(99,102,241,0.45);
    background: rgba(99,102,241,0.22);
}
.lr-hint {
    font-size: 0.85em; color: #4338ca;
    background: #eef2ff; border: 1px solid #c7d2fe;
    border-radius: 8px; padding: 0.5em 0.8em; margin: 0 0 1.2em 0;
}
/* Summary cards stack in order beside the page. */
.lr-notes { flex: 1 1 42%; min-width: 0; display: flex; flex-direction: column; gap: 0.5em; }
.lr-note {
    box-sizing: border-box;
    display: flex; gap: 0.6em; align-items: flex-start;
    padding: 0.6em 0.8em;
    border: 1px solid #e5e7eb; border-radius: 10px;
    background: #fff; cursor: pointer;
    transition: background 0.15s, border-color 0.15s, transform 0.15s;
}
.lr-note:hover { background: #eef2ff; border-color: #6366f1; transform: translateX(2px); }
.lr-note-num {
    flex: 0 0 auto;
    width: 22px; height: 22px;
    background: #6366f1; color: #fff; border-radius: 50%;
    font-size: 0.72em; font-weight: 700;
    display: flex; align-items: center; justify-content: center;
    margin-top: 0.1em;
}
.lr-note-body { display: flex; flex-direction: column; gap: 0.25em; }
.lr-note-text { color: #374151; line-height: 1.45; font-size: 0.98em; }
.lr-note-empty { color: #9ca3af; font-style: italic; font-size: 0.9em; padding: 0.4em 0.2em; }
.lr-empty { color: #9ca3af; font-style: italic; text-align: center; padding: 2em 0; }
@media (max-width: 860px) {
    .lr-row { flex-direction: column; }
    .lr-pagewrap, .lr-notes { flex: 1 1 100%; width: 100%; }
}
"""


MATHJAX_HEAD = r"""
<link rel="icon" type="image/png" sizes="32x32" href="/gradio_api/file=static/favicon-32.png">
<link rel="apple-touch-icon" sizes="192x192" href="/gradio_api/file=static/icon-192.png">
<link rel="manifest" href="/gradio_api/file=static/manifest.webmanifest">
<meta name="theme-color" content="#6366f1">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-title" content="LitRead">
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

// ---- Recent papers: localStorage-backed chip row above URL input -------
// Pure client-side: the server-rendered <img class="recent-push"> fires
// onerror on mount and calls window.LITREAD_RECENT.push(aid, title).
window.LITREAD_RECENT = {
  KEY: 'litread.recent',
  MAX: 8,
  read: function() {
    try { return JSON.parse(localStorage.getItem(this.KEY) || '[]') || []; }
    catch (e) { return []; }
  },
  write: function(arr) {
    try { localStorage.setItem(this.KEY, JSON.stringify(arr.slice(0, this.MAX))); }
    catch (e) { /* quota or private-mode — silently ignore */ }
  },
  push: function(aid, title) {
    if (!aid) return;
    var arr = this.read().filter(function(e) { return e.aid !== aid; });
    arr.unshift({ aid: aid, title: title || aid, t: Date.now() });
    this.write(arr);
    this.render();
  },
  render: function() {
    var box = document.getElementById('recent-papers');
    if (!box) return;
    var arr = this.read();
    var esc = function(s) {
      return String(s || '').replace(/[<>&"']/g, function(c) {
        return ({'<':'&lt;','>':'&gt;','&':'&amp;','"':'&quot;',"'":'&#39;'})[c];
      });
    };
    var inner;
    if (!arr.length) {
      inner = '<div class="recent-header">Recent</div>' +
        '<div class="recent-empty-msg">Papers you generate appear here.</div>';
    } else {
      var lst = '<div class="recent-header">Recent</div><div class="recent-list">';
      arr.forEach(function(e) {
        var aSafe = e.aid.replace(/'/g, "\\'");
        lst += '<button type="button" class="recent-item" ' +
          'title="' + esc(e.title) + '\\n' + esc(e.aid) + '" ' +
          'onclick="window.LITREAD_RECENT.load(\'' + aSafe + '\')">' +
          '<div class="recent-title">' + esc(e.title || e.aid) + '</div>' +
          '<div class="recent-aid">' + esc(e.aid) + '</div>' +
          '</button>';
      });
      lst += '</div>';
      inner = lst;
    }
    box.innerHTML =
      '<button type="button" class="recent-toggle" aria-label="Toggle recent panel" ' +
        'title="Collapse / expand" onclick="window.LITREAD_RECENT.toggle()"></button>' +
      '<div class="recent-collapsed-icon" title="Expand" ' +
        'onclick="window.LITREAD_RECENT.toggle()">📚</div>' +
      '<div class="recent-body">' + inner + '</div>';
  },
  CKEY: 'litread.recent.collapsed',
  isCollapsed: function() {
    try { return localStorage.getItem(this.CKEY) === '1'; } catch (e) { return false; }
  },
  applyCollapsed: function() {
    document.body.classList.toggle('lr-rail-collapsed', this.isCollapsed());
  },
  toggle: function() {
    var next = !this.isCollapsed();
    try { localStorage.setItem(this.CKEY, next ? '1' : '0'); } catch (e) {}
    this.applyCollapsed();
  },
  load: function(aid) {
    var ta = document.querySelector('#url-box textarea');
    if (ta) {
      ta.value = 'https://arxiv.org/abs/' + aid;
      ta.dispatchEvent(new Event('input', { bubbles: true }));
    }
    var btn = document.querySelector('#generate-row button');
    if (btn) btn.click();
  }
};
function __mountRecent() {
  if (!document.getElementById('recent-papers')) {
    setTimeout(__mountRecent, 300); return;
  }
  window.LITREAD_RECENT.applyCollapsed();
  window.LITREAD_RECENT.render();
}
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', __mountRecent);
} else {
  __mountRecent();
}

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

// ---- Annotated reader: click a note to scroll to + flash its box --------
window.lrFlash = function(id) {
  var el = document.getElementById(id);
  if (!el) return;
  el.scrollIntoView({ behavior: 'smooth', block: 'center' });
  el.classList.add('lr-flash');
  setTimeout(function() { el.classList.remove('lr-flash'); }, 1500);
};

// ---- Interactive reader: drag a box on a page -> summarise that region ----
// On mouseup the drag rect (page-relative 0-1 fractions) + page index are written
// to the hidden #lr-selection textbox and the hidden #lr-summarize button is
// clicked, which runs the server-side summary and re-renders the reader.
(function() {
  function setSelectionAndSubmit(page, rect) {
    var box = document.getElementById('lr-selection');
    var ta = box ? (box.matches('textarea,input') ? box : box.querySelector('textarea, input')) : null;
    if (!ta) return;
    var setter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value');
    if (ta.tagName === 'TEXTAREA' && setter && setter.set) { setter.set.call(ta, JSON.stringify({ page: page, rect: rect })); }
    else { ta.value = JSON.stringify({ page: page, rect: rect }); }
    ta.dispatchEvent(new Event('input', { bubbles: true }));
    // gr.Button puts the elem_id on the <button> itself.
    var btn = document.getElementById('lr-summarize');
    if (btn && btn.tagName !== 'BUTTON') btn = btn.querySelector('button') || btn;
    if (btn) btn.click();
  }

  function attach(layer) {
    if (layer.__lrDrag) return;
    layer.__lrDrag = true;
    var start = null, rubber = null;
    layer.addEventListener('mousedown', function(e) {
      if (e.button !== 0) return;
      var r = layer.getBoundingClientRect();
      start = { x: e.clientX - r.left, y: e.clientY - r.top };
      rubber = document.createElement('div');
      rubber.className = 'lr-rubber';
      layer.appendChild(rubber);
      e.preventDefault();
    });
    layer.addEventListener('mousemove', function(e) {
      if (!start || !rubber) return;
      var r = layer.getBoundingClientRect();
      var x = e.clientX - r.left, y = e.clientY - r.top;
      rubber.style.left = Math.min(start.x, x) + 'px';
      rubber.style.top = Math.min(start.y, y) + 'px';
      rubber.style.width = Math.abs(x - start.x) + 'px';
      rubber.style.height = Math.abs(y - start.y) + 'px';
    });
    function overlapsExisting(x0, y0, x1, y1) {
      var page = layer.parentElement;
      if (!page) return null;
      var sels = page.querySelectorAll('.lr-sel');
      for (var i = 0; i < sels.length; i++) {
        var s = sels[i];
        var sx0 = parseFloat(s.style.left) / 100, sy0 = parseFloat(s.style.top) / 100;
        var sx1 = sx0 + parseFloat(s.style.width) / 100, sy1 = sy0 + parseFloat(s.style.height) / 100;
        if (!(x1 <= sx0 || x0 >= sx1 || y1 <= sy0 || y0 >= sy1)) return s;  // intersects
      }
      return null;
    }
    function finish(e) {
      if (!start) return;
      var r = layer.getBoundingClientRect();
      var x = e.clientX - r.left, y = e.clientY - r.top;
      var x0 = Math.min(start.x, x) / r.width, x1 = Math.max(start.x, x) / r.width;
      var y0 = Math.min(start.y, y) / r.height, y1 = Math.max(start.y, y) / r.height;
      if (rubber) { rubber.remove(); rubber = null; }
      start = null;
      if ((x1 - x0) <= 0.01 || (y1 - y0) <= 0.01) return;   // too small — ignore
      var hit = overlapsExisting(x0, y0, x1, y1);
      if (hit) { if (hit.id) window.lrFlash(hit.id); return; }  // already summarised — reject
      setSelectionAndSubmit(parseInt(layer.dataset.page, 10), [x0, y0, x1, y1]);
    }
    layer.addEventListener('mouseup', finish);
    layer.addEventListener('mouseleave', function(e) { if (start) finish(e); });
  }

  function hook() {
    var r = document.getElementById('reader');
    if (!r) { setTimeout(hook, 300); return; }
    var pending = null;
    function scan() { r.querySelectorAll('.lr-drawlayer').forEach(attach); }
    new MutationObserver(function() {
      if (pending) clearTimeout(pending);
      pending = setTimeout(scan, 100);
    }).observe(r, { childList: true, subtree: true });
    scan();
  }
  hook();
})();
</script>
"""


def build_ui() -> gr.Blocks:
    initial_cost = render_cost_html(list(PROVIDERS.values())[0]["model"])

    with gr.Blocks(title="LitRead") as demo:
        # Left rail — JS reads localStorage and renders a stacked list of
        # recent paper titles. Hidden on narrow screens via media query.
        gr.HTML(value="", elem_id="recent-papers")

        # ---- Top bar: title left, [Model + API Usage] stacked top-right ----
        with gr.Row(elem_id="topbar"):
            with gr.Column(scale=4):
                gr.Markdown("# 📚 LitRead")
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
                generate_btn = gr.Button("✨ Generate Carousel + Reader", variant="primary")
            with gr.Row(elem_id="force-row"):
                force = gr.Checkbox(
                    label="↻ Force regenerate (ignore cache)",
                    value=False,
                    elem_id="force-toggle",
                    container=False,
                )

        status = gr.Markdown("_Ready._", elem_id="status")
        output = gr.HTML(value="", elem_id="output")
        reader = gr.HTML(value="", elem_id="reader")

        # Interactive reader plumbing: per-session state + a hidden bridge the
        # drag JS drives (writes the selection JSON, clicks the summarise button).
        paper_state = gr.State(None)
        selection = gr.Textbox(elem_id="lr-selection", show_label=False, container=False)
        summarize_btn = gr.Button("summarize", elem_id="lr-summarize")

        # Update cost panel live when provider changes.
        def _on_provider_change(p):
            return render_cost_html(PROVIDERS[p]["model"])

        provider.change(_on_provider_change, inputs=[provider], outputs=[cost])

        for trigger in (url.submit, generate_btn.click):
            ev = trigger(
                generate_carousel,
                inputs=[url, provider, force],
                outputs=[status, output, cost],
            )
            # The same click also opens the interactive PDF reader below the cards.
            ev.then(
                annotate.load_reader,
                inputs=[url, provider, force],
                outputs=[reader, paper_state],
            )

        # Draw a box on the reader -> summarise that region (updates API Usage).
        summarize_btn.click(
            annotate.summarize_region,
            inputs=[paper_state, selection, provider],
            outputs=[reader, cost, paper_state],
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
        favicon_path="static/favicon-32.png",
        allowed_paths=["static", core.CACHE_DIR],
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
