"""Interactive PDF reader — the second LitRead mode.

The user opens a paper as plain page images, drags a bounding box over any region,
and gets an on-demand plain-English summary of the text under the box. Summaries
accumulate as a numbered running list beside the page; each leaves a numbered marker
on the page.

Pipeline:
    arxiv id -> core.fetch_pdf (bytes + metadata) -> render page PNGs -> session state
    drag box (JS) -> {page, rect} -> extract_region_text (fitz clip) -> LLM summary
                  -> append to state -> re-render reader + API-usage cost
"""

import html
import json
import os
import time
from typing import Any, Dict, List, Optional

import fitz  # pymupdf
from langchain_core.messages import HumanMessage, SystemMessage

import core
from core import PROVIDERS

PAGE_ZOOM = 2.0                 # rasterise pages at 2x for a crisp base image
MAX_FETCH_RETRIES = 4
MIN_SELECTION_FRAC = 0.01       # ignore drags smaller than 1% of the page
MAX_REGION_CHARS = 8000         # safety cap on text sent to the LLM per selection
PNG_DIR = os.path.join(core.CACHE_DIR, "pages")


REGION_PROMPT = """You are a clear, plain-spoken science communicator.

The user selected a region of a research paper. You are given the text from that region.
Explain what it is saying in plain, layman's terms — 2 to 4 short sentences, no jargon
(define any unavoidable term in a few words). If it is an equation, say in words what it
computes or represents. Stay grounded in the given text; do not invent results.

Output ONLY the explanation, as plain text (no markdown headers, no preamble)."""


# -----------------------------------------------------------------------------
# Page rasterisation
# -----------------------------------------------------------------------------
def render_page_pngs(pdf_bytes: bytes, arxiv_id: str, zoom: float = PAGE_ZOOM) -> List[str]:
    """Rasterise each page to a PNG under PNG_DIR; return the file paths.
    Re-uses existing files so repeat loads of the same paper skip re-rendering."""
    os.makedirs(PNG_DIR, exist_ok=True)
    safe = arxiv_id.replace("/", "_")
    paths: List[str] = []
    matrix = fitz.Matrix(zoom, zoom)
    with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf:
        for pno, page in enumerate(pdf):
            path = os.path.join(PNG_DIR, f"{safe}_p{pno}.png")
            if not os.path.exists(path):
                page.get_pixmap(matrix=matrix).save(path)
            paths.append(path)
    return paths


def _png_urls(paths: List[str]) -> List[str]:
    return [f"/gradio_api/file={p}" for p in paths]


# -----------------------------------------------------------------------------
# Region text extraction + selection parsing (pure)
# -----------------------------------------------------------------------------
def extract_region_text(pdf_bytes: bytes, page_index: int, rect: List[float]) -> str:
    """Return the text under a fractional rect [x0,y0,x1,y1] (each 0-1) on a page."""
    x0, y0, x1, y1 = rect
    with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf:
        if page_index < 0 or page_index >= pdf.page_count:
            return ""
        page = pdf[page_index]
        w, h = page.rect.width, page.rect.height
        clip = fitz.Rect(min(x0, x1) * w, min(y0, y1) * h, max(x0, x1) * w, max(y0, y1) * h)
        return page.get_text("text", clip=clip).strip()


def parse_selection(selection_json: str, n_pages: int) -> Optional[Dict[str, Any]]:
    """Validate the JSON a drag emits. Returns {page, rect:[x0,y0,x1,y1]} or None for
    malformed / out-of-range / too-small selections."""
    try:
        d = json.loads(selection_json)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(d, dict):
        return None
    page = d.get("page")
    rect = d.get("rect")
    if not isinstance(page, int) or page < 0 or page >= n_pages:
        return None
    if not (isinstance(rect, list) and len(rect) == 4):
        return None
    try:
        vals = [float(v) for v in rect]
    except (TypeError, ValueError):
        return None
    if any(v < -0.05 or v > 1.05 for v in vals):
        return None
    x0, x1 = sorted((max(0.0, min(1.0, vals[0])), max(0.0, min(1.0, vals[2]))))
    y0, y1 = sorted((max(0.0, min(1.0, vals[1])), max(0.0, min(1.0, vals[3]))))
    if (x1 - x0) < MIN_SELECTION_FRAC or (y1 - y0) < MIN_SELECTION_FRAC:
        return None
    return {"page": page, "rect": [x0, y0, x1, y1]}


def append_summary(
    state: Dict[str, Any],
    sel: Dict[str, Any],
    text: str,
    summary: str,
    in_tok: int,
    out_tok: int,
) -> Dict[str, Any]:
    """Return a new state with the summary appended and token totals updated."""
    sums = list(state.get("summaries", []))
    n = len(sums) + 1
    sums.append({
        "n": n, "page": sel["page"], "rect": sel["rect"],
        "text": text, "summary": summary,
    })
    new = dict(state)
    new["summaries"] = sums
    new["in_tok"] = state.get("in_tok", 0) + in_tok
    new["out_tok"] = state.get("out_tok", 0) + out_tok
    return new


# -----------------------------------------------------------------------------
# Rendering
# -----------------------------------------------------------------------------
def _pct(v: float) -> str:
    return f"{round(v * 100, 2)}%"


def _placeholder(msg: str = "Load a paper, then drag a box over any region to summarise it.") -> str:
    return f'<div class="lr-empty">{html.escape(msg)}</div>'


def render_reader(state: Optional[Dict[str, Any]]) -> str:
    """Page images (with a drag layer + selection markers) beside a panel of summaries."""
    if not state or not state.get("pages_meta"):
        return _placeholder()

    title = html.escape(str(state.get("title", "Unknown")))
    authors = html.escape(str(state.get("authors", "Unknown")))
    png_urls = _png_urls(state.get("pngs", []))
    summaries = state.get("summaries", [])
    by_page: Dict[int, List[Dict[str, Any]]] = {}
    for s in summaries:
        by_page.setdefault(s["page"], []).append(s)

    parts = [
        f'<div class="lr-header"><h2 class="lr-title">{title}</h2>'
        f'<p class="lr-authors">{authors}</p></div>',
        '<div class="lr-hint">Drag a box over any region of a page to get a plain-English summary.</div>',
    ]

    for pm in state["pages_meta"]:
        pno = pm["page"]
        url = png_urls[pno] if pno < len(png_urls) else ""
        page_sels = by_page.get(pno, [])

        markers = []
        for s in page_sels:
            x0, y0, x1, y1 = s["rect"]
            markers.append(
                f'<div class="lr-sel" id="lr-box-{s["n"]}" '
                f'style="left:{_pct(x0)};top:{_pct(y0)};width:{_pct(x1 - x0)};height:{_pct(y1 - y0)}">'
                f'<span class="lr-box-num">{s["n"]}</span></div>'
            )

        cards = []
        for s in page_sels:
            cards.append(
                f'<div class="lr-note" id="lr-note-{s["n"]}" data-target="lr-box-{s["n"]}" '
                f'onclick="lrFlash(\'lr-box-{s["n"]}\')">'
                f'<span class="lr-note-num">{s["n"]}</span>'
                f'<span class="lr-note-body">'
                f'<span class="lr-note-text">{html.escape(s["summary"])}</span>'
                f'</span></div>'
            )
        cards_html = "".join(cards) or '<div class="lr-note-empty">No selections on this page yet.</div>'

        parts.append(
            f'<div class="lr-row">'
            f'<div class="lr-pagewrap">'
            f'<div class="lr-page-num">p.{pno + 1}</div>'
            f'<div class="lr-page">'
            f'<img class="lr-page-img" src="{html.escape(url)}" loading="lazy" alt="page {pno + 1}">'
            f'<div class="lr-drawlayer" data-page="{pno}"></div>'
            f'{"".join(markers)}'
            f'</div></div>'
            f'<div class="lr-notes">{cards_html}</div>'
            f'</div>'
        )
    return "".join(parts)


# -----------------------------------------------------------------------------
# LLM summary
# -----------------------------------------------------------------------------
def _summarize_text(text: str, provider: str, api_key: str):
    llm = core.make_llm(provider, api_key, streaming=False)
    resp = llm.invoke([
        SystemMessage(content=REGION_PROMPT),
        HumanMessage(content=f"SELECTED TEXT:\n\n{text[:MAX_REGION_CHARS]}"),
    ])
    usage = getattr(resp, "usage_metadata", None) or {}
    summary = (getattr(resp, "content", "") or "").strip()
    return summary, usage.get("input_tokens", 0) or 0, usage.get("output_tokens", 0) or 0


# -----------------------------------------------------------------------------
# Gradio callbacks
# -----------------------------------------------------------------------------
def load_paper(url: str, provider: str, force: bool = False):
    """Fetch + rasterise a paper for interactive exploration. Generator yielding
    (status_md, reader_html, cost_html, state)."""
    model = PROVIDERS[provider]["model"]
    cost0 = core.render_cost_html(model)
    loading = _placeholder("Loading paper…")

    aid = core.extract_arxiv_id(url)
    if not aid:
        yield "❌ **Invalid URL.** Could not find an ArXiv ID.", _placeholder("Paste a valid ArXiv URL or ID."), cost0, None
        return

    t_start = time.monotonic()
    pdf_bytes: Optional[bytes] = None
    meta: Optional[Dict[str, Any]] = None
    for attempt in range(MAX_FETCH_RETRIES):
        yield f"⏳ Fetching paper `{aid}`… (attempt {attempt + 1}/{MAX_FETCH_RETRIES})", loading, cost0, None
        try:
            pdf_bytes, meta = core.fetch_pdf(aid)
            break
        except Exception as err:
            status = getattr(err, "code", None) or getattr(err, "status", None)
            retryable = status == 429 or (isinstance(status, int) and 500 <= status < 600)
            if retryable and attempt < MAX_FETCH_RETRIES - 1:
                wait = (attempt + 1) * 5
                yield f"⏳ ArXiv HTTP {status}. Retrying in {wait}s…", loading, cost0, None
                time.sleep(wait)
                continue
            yield f"❌ **Fetch failed:** `{type(err).__name__}: {err}`", _placeholder("Fetch failed."), cost0, None
            return

    if pdf_bytes is None:
        yield f"❌ **Paper `{aid}` not found.**", _placeholder("Paper not found."), cost0, None
        return

    yield f"✅ Fetched. Rendering pages for `{aid}`…", loading, cost0, None
    with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf:
        pages_meta = [
            {"page": i, "width": p.rect.width, "height": p.rect.height}
            for i, p in enumerate(pdf)
        ]
    pngs = render_page_pngs(pdf_bytes, aid)

    state = {
        "aid": aid,
        "pdf_bytes": pdf_bytes,
        "pages_meta": pages_meta,
        "pngs": pngs,
        "title": (meta or {}).get("Title", "Unknown"),
        "authors": (meta or {}).get("Authors", "Unknown"),
        "summaries": [],
        "in_tok": 0,
        "out_tok": 0,
        "model": model,
    }
    total = time.monotonic() - t_start
    yield (
        f"✅ Loaded `{aid}` in {total:.1f}s · {len(pages_meta)} pages · drag a box to summarise.",
        render_reader(state),
        cost0,
        state,
    )


def summarize_region(state: Optional[Dict[str, Any]], selection_json: str, provider: str):
    """Summarise the text under a drawn box. Returns (reader_html, cost_html, state)."""
    model = PROVIDERS[provider]["model"]
    if not state or not state.get("pdf_bytes"):
        return _placeholder(), core.render_cost_html(model), state

    def _out(st):
        return render_reader(st), core.render_cost_html(model, st.get("in_tok", 0), st.get("out_tok", 0)), st

    sel = parse_selection(selection_json, len(state["pages_meta"]))
    if sel is None:
        return _out(state)  # malformed/tiny selection — ignore

    text = extract_region_text(state["pdf_bytes"], sel["page"], sel["rect"])
    if not text:
        return _out(append_summary(state, sel, "", "No selectable text in that region.", 0, 0))

    env_key = PROVIDERS[provider]["env_key"]
    api_key = os.getenv(env_key, "").strip()
    if not api_key:
        return _out(append_summary(state, sel, text, f"⚠️ Set {env_key} in .env to summarise.", 0, 0))

    try:
        summary, in_tok, out_tok = _summarize_text(text, provider, api_key)
    except Exception as e:
        return _out(append_summary(state, sel, text, f"⚠️ Summary failed: {type(e).__name__}: {e}", 0, 0))

    return _out(append_summary(state, sel, text, summary, in_tok, out_tok))
