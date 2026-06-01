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
import re
import time
from typing import Any, Dict, List, Optional

import fitz  # pymupdf
from langchain_core.messages import HumanMessage, SystemMessage

import core
from core import PROVIDERS

PAGE_ZOOM = 2.0                 # rasterise pages at 2x for a crisp base image
MAX_FETCH_RETRIES = 4
SUMMARY_CACHE_VERSION = "v2"     # bump to invalidate cached region summaries
MIN_SELECTION_FRAC = 0.01       # ignore drags smaller than 1% of the page
MAX_REGION_CHARS = 8000         # safety cap on text sent to the LLM per selection
PNG_DIR = os.path.join(core.CACHE_DIR, "pages")


REGION_PROMPT = """You are a clear, plain-spoken science communicator.

The user selected a region of a research paper; you are given the text from that region.
Explain what it says in plain, layman's terms as 2 to 4 short bullet points. Put each
bullet on its own line starting with "- ", one idea per bullet, no jargon (define any
unavoidable term in a few words). **Bold the most important term(s)** in each bullet using
double asterisks, e.g. "- The model uses **contrastive learning** to ...". If it is an
equation, say in words what it computes or represents. Stay grounded in the given text;
do not invent results.

Output ONLY the bullet lines (each starting with "- "), nothing else."""


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


def extract_regions_text(pdf_bytes: bytes, regions: List[Dict[str, Any]]) -> str:
    """Concatenate the text under each region (in order), blank-line separated."""
    parts = [extract_region_text(pdf_bytes, r["page"], r["rect"]) for r in regions]
    return "\n\n".join(p for p in parts if p).strip()


def _valid_region(page: Any, rect: Any, n_pages: int) -> Optional[Dict[str, Any]]:
    """Validate one {page, rect} → normalised dict, or None."""
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


def parse_selection(selection_json: str, n_pages: int) -> Optional[Dict[str, Any]]:
    """Validate a single-region drag {page, rect} → dict or None."""
    try:
        d = json.loads(selection_json)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(d, dict):
        return None
    return _valid_region(d.get("page"), d.get("rect"), n_pages)


def parse_regions(selection_json: str, n_pages: int) -> Optional[List[Dict[str, Any]]]:
    """Validate a {"regions":[{page,rect},...]} payload. Returns a non-empty list of
    normalised regions, or None if malformed / all-invalid."""
    try:
        d = json.loads(selection_json)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(d, dict) or not isinstance(d.get("regions"), list):
        return None
    out = []
    for r in d["regions"]:
        if isinstance(r, dict):
            v = _valid_region(r.get("page"), r.get("rect"), n_pages)
            if v:
                out.append(v)
    return out or None


def _summary_cache_key(aid: str, provider: str, model: str) -> str:
    return core.cache_key(aid, provider, model, "summaries-" + SUMMARY_CACHE_VERSION)


def _persist_summaries(state: Dict[str, Any], provider: str) -> None:
    """Best-effort save of a paper's region summaries so they survive reloads."""
    try:
        core.cache_put(
            _summary_cache_key(state["aid"], provider, state["model"]),
            {"summaries": state["summaries"]},
        )
    except Exception:
        pass


def append_summary(
    state: Dict[str, Any],
    regions: List[Dict[str, Any]],
    text: str,
    summary: str,
    in_tok: int,
    out_tok: int,
) -> Dict[str, Any]:
    """Return a new state with the summary (covering one or more regions) appended
    and token totals updated."""
    sums = list(state.get("summaries", []))
    n = len(sums) + 1
    sums.append({"n": n, "regions": regions, "text": text, "summary": summary})
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


def _bullets_html(summary: str) -> str:
    """Render the summary (LLM returns '- ' bullet lines) as a <ul>. Falls back to a
    single bullet for one-line messages (e.g. warnings)."""
    lines = [re.sub(r"^\s*[-*•·]\s*", "", ln).strip() for ln in (summary or "").splitlines()]
    items = [ln for ln in lines if ln]
    if not items:
        t = (summary or "").strip()
        items = [t] if t else []
    lis = "".join(f"<li>{core.escape_with_bold(it)}</li>" for it in items)
    return f'<ul class="lr-bullets">{lis}</ul>'


def render_reader(state: Optional[Dict[str, Any]]) -> str:
    """Page images (with a drag layer + selection markers) beside a panel of summaries."""
    if not state or not state.get("pages_meta"):
        return _placeholder()

    title = html.escape(str(state.get("title", "Unknown")))
    authors = html.escape(str(state.get("authors", "Unknown")))
    png_urls = _png_urls(state.get("pngs", []))
    summaries = state.get("summaries", [])

    def _regions_of(s):
        regs = s.get("regions")
        if isinstance(regs, list) and regs:
            return regs
        if "rect" in s:                      # legacy single-region entry
            return [{"page": s.get("page", 0), "rect": s["rect"]}]
        return []

    cards_by_page: Dict[int, List[Dict[str, Any]]] = {}        # first-region page -> summaries
    markers_by_page: Dict[int, List[Any]] = {}                 # page -> (summary, idx, region)
    for s in summaries:
        regs = _regions_of(s)
        if not regs:
            continue
        cards_by_page.setdefault(regs[0]["page"], []).append(s)
        for i, reg in enumerate(regs):
            markers_by_page.setdefault(reg["page"], []).append((s, i, reg))

    parts = [
        f'<div class="lr-header"><h2 class="lr-title">{title}</h2>'
        f'<p class="lr-authors">{authors}</p></div>',
        '<div class="lr-hint">Drag a box to summarise a region. Hold <b>Shift</b> and drag to '
        'add more regions (even across pages), then click <b>Summarise</b> to combine them.</div>',
    ]

    for pm in state["pages_meta"]:
        pno = pm["page"]
        url = png_urls[pno] if pno < len(png_urls) else ""

        markers = []
        for s, i, reg in markers_by_page.get(pno, []):
            x0, y0, x1, y1 = reg["rect"]
            mid = f'lr-box-{s["n"]}' if i == 0 else f'lr-box-{s["n"]}-{i}'
            markers.append(
                f'<div class="lr-sel" id="{mid}" '
                f'style="left:{_pct(x0)};top:{_pct(y0)};width:{_pct(x1 - x0)};height:{_pct(y1 - y0)}">'
                f'<span class="lr-box-num">{s["n"]}</span></div>'
            )

        cards = []
        for s in cards_by_page.get(pno, []):
            cards.append(
                f'<div class="lr-note" id="lr-note-{s["n"]}" data-target="lr-box-{s["n"]}" '
                f'onclick="lrFlash(\'lr-box-{s["n"]}\')">'
                f'<span class="lr-note-num">{s["n"]}</span>'
                f'<span class="lr-note-body">{_bullets_html(s["summary"])}</span>'
                f'<button class="lr-del" title="Remove this annotation" '
                f'onclick="event.stopPropagation(); window.lrDelete({s["n"]})">✕</button>'
                f'</div>'
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
def load_reader(url: str, provider: str, force: bool = False):
    """Fetch + rasterise a paper for interactive reading. Generator yielding
    (reader_html, state). Runs alongside the carousel on the same button, so it
    does not touch the shared status/cost panels (errors render in the reader)."""
    model = PROVIDERS[provider]["model"]
    aid = core.extract_arxiv_id(url)
    if not aid:
        yield _placeholder("Paste a valid ArXiv URL to load the PDF."), None
        return

    yield _placeholder(f"Loading PDF for {aid} — drag a box once it appears…"), None
    pdf_bytes: Optional[bytes] = None
    meta: Optional[Dict[str, Any]] = None
    for attempt in range(MAX_FETCH_RETRIES):
        try:
            pdf_bytes, meta = core.fetch_pdf(aid)
            break
        except Exception as err:
            status = getattr(err, "code", None) or getattr(err, "status", None)
            retryable = status == 429 or (isinstance(status, int) and 500 <= status < 600)
            if retryable and attempt < MAX_FETCH_RETRIES - 1:
                wait = (attempt + 1) * 5
                yield _placeholder(f"ArXiv HTTP {status} — retrying in {wait}s…"), None
                time.sleep(wait)
                continue
            yield _placeholder(f"Could not load PDF: {type(err).__name__}: {err}"), None
            return

    if pdf_bytes is None:
        yield _placeholder(f"Paper {aid} not found."), None
        return

    with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf:
        pages_meta = [
            {"page": i, "width": p.rect.width, "height": p.rect.height}
            for i, p in enumerate(pdf)
        ]
    pngs = render_page_pngs(pdf_bytes, aid)
    state = {
        "aid": aid, "pdf_bytes": pdf_bytes, "pages_meta": pages_meta, "pngs": pngs,
        "title": (meta or {}).get("Title", "Unknown"),
        "authors": (meta or {}).get("Authors", "Unknown"),
        "summaries": [], "in_tok": 0, "out_tok": 0, "model": model,
    }
    # Restore previously cached summaries for this paper (drag-summaries persist).
    cached = core.cache_get(_summary_cache_key(aid, provider, model))
    if cached and isinstance(cached.get("summaries"), list):
        state["summaries"] = cached["summaries"]
    yield render_reader(state), state


def _render_out(state: Optional[Dict[str, Any]], provider: str):
    """(reader_html, cost_html, state) for the current state."""
    model = PROVIDERS[provider]["model"]
    if not state:
        return _placeholder(), core.render_cost_html(model), state
    return (render_reader(state),
            core.render_cost_html(model, state.get("in_tok", 0), state.get("out_tok", 0)),
            state)


def delete_summary(state: Optional[Dict[str, Any]], n_json: str, provider: str):
    """Remove the summary numbered n_json, renumber the rest, re-render + re-cache."""
    if not state or not state.get("summaries"):
        return _render_out(state, provider)
    try:
        n = int(str(n_json).strip())
    except (TypeError, ValueError):
        return _render_out(state, provider)
    sums = [dict(s) for s in state["summaries"] if s["n"] != n]
    for i, s in enumerate(sums, 1):
        s["n"] = i
    new = dict(state)
    new["summaries"] = sums
    _persist_summaries(new, provider)
    return _render_out(new, provider)


def summarize_region(state: Optional[Dict[str, Any]], selection_json: str, provider: str):
    """Summarise the text under one or more drawn boxes (combined). Returns
    (reader_html, cost_html, state)."""
    model = PROVIDERS[provider]["model"]
    if not state or not state.get("pdf_bytes"):
        return _placeholder(), core.render_cost_html(model), state

    def _out(st):
        return _render_out(st, provider)

    regions = parse_regions(selection_json, len(state["pages_meta"]))
    if not regions:
        return _out(state)  # malformed/tiny selection — ignore

    text = extract_regions_text(state["pdf_bytes"], regions)
    if not text:
        new = append_summary(state, regions, "", "No selectable text in that region.", 0, 0)
        _persist_summaries(new, provider)
        return _out(new)

    env_key = PROVIDERS[provider]["env_key"]
    api_key = os.getenv(env_key, "").strip()
    if not api_key:
        # Don't cache a transient "set your key" message.
        return _out(append_summary(state, regions, text, f"⚠️ Set {env_key} in .env to summarise.", 0, 0))

    try:
        summary, in_tok, out_tok = _summarize_text(text, provider, api_key)
    except Exception as e:
        # Don't cache a transient error.
        return _out(append_summary(state, regions, text, f"⚠️ Summary failed: {type(e).__name__}: {e}", 0, 0))

    new = append_summary(state, regions, text, summary, in_tok, out_tok)
    _persist_summaries(new, provider)
    return _out(new)
