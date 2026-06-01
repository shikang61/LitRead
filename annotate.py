"""Annotated PDF reader — the second LitRead mode.

Pipeline:
    arxiv id -> core.fetch_pdf (bytes + metadata)
             -> extract_blocks (per-page block list with stable IDs + bboxes)
             -> render_page_pngs (each page rasterised to a PNG)
             -> build_manifest (compact block list) -> LLM -> JSON annotations
             -> parse_annotations -> layout_annotations -> resolve_highlights
             -> render_reader_html (page image + overlay boxes/highlights | notes)

The LLM references PDF *block IDs* (not verbatim quotes), so boxing a region is
an exact bbox lookup. Overlay geometry is expressed in page-percentages so it
scales with the responsive page image.
"""

import html
import os
import queue
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import fitz  # pymupdf
from langchain_core.messages import HumanMessage, SystemMessage

import core
from core import PROVIDERS

# Bump when ANNOTATE_PROMPT changes — invalidates cached annotations.
ANNOTATE_VERSION = "v2"

PAGE_ZOOM = 2.0                 # rasterise pages at 2x for a crisp overlay base
MAX_MANIFEST_CHARS = 120_000    # cap LLM input; long papers get tail-truncated
MAX_FETCH_RETRIES = 4
PNG_DIR = os.path.join(core.CACHE_DIR, "pages")

VALID_KINDS = {"section", "paragraph", "equation", "figure"}


# -----------------------------------------------------------------------------
# System prompt — strict JSON, selective key-bits-only annotations.
# -----------------------------------------------------------------------------
ANNOTATE_PROMPT = """You annotate a research paper so a reader can skim it section by section and
understand what every part is doing, then dive into the bits they care about.

You receive a list of the paper's text blocks, one per line, formatted:
    <block_id>: <block text>
Image/figure blocks appear as:
    <block_id>: [FIGURE]

Annotate THOROUGHLY — aim for broad coverage, roughly 8-15 annotations per page:

1. SECTION MAP (do this first): for EVERY section and major subsection (Abstract,
   Introduction, Related Work, Method, Experiments, Results, Discussion, Conclusion,
   appendices, etc.), emit ONE annotation of kind "section" on that section's heading or
   first block, whose note states the POINT of the whole section in one line — why it's
   there and what the reader gets from it. Do not skip a section.

2. KEY POINTS within each section: add "paragraph" / "equation" / "figure" annotations
   for the substantive content — the core claim, each method step, the key equations,
   important figures/tables, the main results and comparisons. Most content-bearing
   paragraphs should get a note. Only skip truly trivial blocks (page headers/footers,
   author lists, pure citation lines, acknowledgements).

For each annotation, write a one-line side-note (max ~16 words, plain English, active
voice) saying what that block contributes — what a skimmer should take away from it.

OUTPUT (strict JSON, no markdown fences, no commentary):
{
  "annotations": [
    {
      "block_id": "p2_b5",
      "kind": "section" | "paragraph" | "equation" | "figure",
      "note": "one-line plain-English takeaway for this block",
      "keywords": ["exact phrase from the block to highlight", "another"]
    }
  ]
}

RULES:
- "block_id" MUST be copied verbatim from the input. Never invent an ID.
- "keywords": 1-4 short phrases that appear VERBATIM in that block's text (so they can be
  located and highlighted). For [FIGURE] and most "section" blocks use an empty list. If
  unsure a phrase is present verbatim, omit it.
- "kind": "section" for a section/subsection purpose summary; "equation" for blocks that
  are mainly a formula; "figure" for [FIGURE] blocks; "paragraph" otherwise.
- Keep notes concrete and grounded in the block. Do not invent results or claims.
- Output ONLY the JSON object.
"""


# -----------------------------------------------------------------------------
# Block extraction
# -----------------------------------------------------------------------------
def extract_blocks(pdf_bytes: bytes) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return (pages_meta, blocks).

    pages_meta: [{page, width, height}] in PDF points.
    blocks:     [{id:'p{page}_b{idx}', page, bbox:(x0,y0,x1,y1), kind:'text'|'image', text}]
    """
    pages_meta: List[Dict[str, Any]] = []
    blocks: List[Dict[str, Any]] = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf:
        for pno, page in enumerate(pdf):
            d = page.get_text("dict")
            pages_meta.append({"page": pno, "width": d["width"], "height": d["height"]})
            for bidx, b in enumerate(d.get("blocks", [])):
                bbox = tuple(round(float(c), 2) for c in b["bbox"])
                if b.get("type", 0) == 0:
                    lines = [
                        "".join(span.get("text", "") for span in line.get("spans", []))
                        for line in b.get("lines", [])
                    ]
                    text = " ".join(t for t in lines).strip()
                    kind = "text"
                else:
                    text = ""
                    kind = "image"
                blocks.append({
                    "id": f"p{pno}_b{bidx}",
                    "page": pno,
                    "bbox": bbox,
                    "kind": kind,
                    "text": text,
                })
    return pages_meta, blocks


def render_page_pngs(pdf_bytes: bytes, arxiv_id: str, zoom: float = PAGE_ZOOM) -> List[str]:
    """Rasterise each page to a PNG under PNG_DIR; return the file paths.
    Re-uses existing files so repeat runs on the same paper skip re-rendering."""
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


def build_manifest(blocks: List[Dict[str, Any]], max_chars: int = MAX_MANIFEST_CHARS) -> str:
    """Compact block list fed to the LLM. Skips empty text blocks; marks images."""
    lines: List[str] = []
    total = 0
    for b in blocks:
        if b["kind"] == "image":
            entry = f'{b["id"]}: [FIGURE]'
        else:
            t = " ".join(b["text"].split())
            if not t:
                continue
            if len(t) > 600:
                t = t[:600] + "…"
            entry = f'{b["id"]}: {t}'
        if total + len(entry) > max_chars:
            break
        lines.append(entry)
        total += len(entry) + 1
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Coordinate math + annotation processing
# -----------------------------------------------------------------------------
def bbox_to_pct(bbox, page_w: float, page_h: float) -> Dict[str, float]:
    """PDF-point bbox -> page-relative percentages for CSS overlay placement."""
    x0, y0, x1, y1 = bbox
    pw = page_w or 1.0
    ph = page_h or 1.0
    return {
        "left": x0 / pw * 100.0,
        "top": y0 / ph * 100.0,
        "width": (x1 - x0) / pw * 100.0,
        "height": (y1 - y0) / ph * 100.0,
    }


def parse_annotations(raw: str, valid_ids) -> List[Dict[str, Any]]:
    """Parse the LLM's JSON and keep only well-formed entries that reference a
    real block ID and carry a note. Hallucinated IDs / blank notes are dropped."""
    data = core.parse_json(raw)
    anns = data.get("annotations") if isinstance(data, dict) else None
    if not isinstance(anns, list):
        return []
    out: List[Dict[str, Any]] = []
    for a in anns:
        if not isinstance(a, dict):
            continue
        bid = a.get("block_id")
        if bid not in valid_ids:
            continue
        note = str(a.get("note", "")).strip()
        if not note:
            continue
        kind = a.get("kind", "paragraph")
        if kind not in VALID_KINDS:
            kind = "paragraph"
        kws = a.get("keywords")
        if isinstance(kws, list):
            keywords = [str(k).strip() for k in kws if str(k).strip()]
        else:
            keywords = []
        out.append({"block_id": bid, "kind": kind, "note": note, "keywords": keywords})
    return out


def layout_annotations(
    annotations: List[Dict[str, Any]],
    blocks: List[Dict[str, Any]],
    pages_meta: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Attach page, percentage box geometry, and a reading-order number to each
    annotation. Sorted top-to-bottom within each page. highlights start empty."""
    bmap = {b["id"]: b for b in blocks}
    pmap = {p["page"]: p for p in pages_meta}
    enriched: List[Dict[str, Any]] = []
    for a in annotations:
        b = bmap.get(a["block_id"])
        if not b:
            continue
        pm = pmap.get(b["page"])
        if not pm:
            continue
        enriched.append({
            "block_id": a["block_id"],
            "kind": a["kind"],
            "note": a["note"],
            "keywords": a["keywords"],
            "page": b["page"],
            "_y0": b["bbox"][1],
            "box": bbox_to_pct(b["bbox"], pm["width"], pm["height"]),
            "highlights": [],
        })
    enriched.sort(key=lambda e: (e["page"], e["_y0"]))
    for i, e in enumerate(enriched, start=1):
        e["n"] = i
        e.pop("_y0", None)
    return enriched


def resolve_highlights(
    pdf: "fitz.Document",
    enriched: List[Dict[str, Any]],
    blocks: List[Dict[str, Any]],
) -> None:
    """For each annotation keyword, locate it within its block's region and store
    page-percentage highlight rects. Mutates `enriched` in place. A keyword that
    can't be found is silently skipped (box + note still render)."""
    bmap = {b["id"]: b for b in blocks}
    for e in enriched:
        b = bmap.get(e["block_id"])
        if not b:
            continue
        page = pdf[e["page"]]
        pw, ph = page.rect.width, page.rect.height
        clip = fitz.Rect(b["bbox"])
        hls: List[Dict[str, float]] = []
        for kw in e["keywords"]:
            try:
                rects = page.search_for(kw, clip=clip)
            except Exception:
                rects = []
            for r in rects:
                hls.append(bbox_to_pct((r.x0, r.y0, r.x1, r.y1), pw, ph))
        e["highlights"] = hls


# -----------------------------------------------------------------------------
# Rendering
# -----------------------------------------------------------------------------
def _pct(v: float) -> str:
    return f"{round(v, 1)}%"


def render_reader_html(
    pages_meta: List[Dict[str, Any]],
    png_urls: List[str],
    enriched: List[Dict[str, Any]],
) -> str:
    """Side-by-side reader: each page image with overlay boxes/highlights on the
    left, its annotation notes on the right."""
    by_page: Dict[int, List[Dict[str, Any]]] = {}
    for e in enriched:
        by_page.setdefault(e["page"], []).append(e)

    parts: List[str] = []
    if not enriched:
        parts.append(
            '<div class="lr-empty">Nothing flagged for annotation — '
            'the model found no high-signal blocks to box.</div>'
        )

    for pm in pages_meta:
        pno = pm["page"]
        url = png_urls[pno] if pno < len(png_urls) else ""
        page_anns = by_page.get(pno, [])

        overlay: List[str] = []
        for e in page_anns:
            box = e["box"]
            overlay.append(
                f'<div class="lr-box lr-box-{e["kind"]}" id="lr-box-{e["n"]}" '
                f'style="left:{_pct(box["left"])};top:{_pct(box["top"])};'
                f'width:{_pct(box["width"])};height:{_pct(box["height"])}">'
                f'<span class="lr-box-num">{e["n"]}</span></div>'
            )
            for h in e["highlights"]:
                overlay.append(
                    f'<div class="lr-hl" style="left:{_pct(h["left"])};top:{_pct(h["top"])};'
                    f'width:{_pct(h["width"])};height:{_pct(h["height"])}"></div>'
                )

        notes: List[str] = []
        for e in page_anns:
            top = e["box"]["top"]
            notes.append(
                f'<div class="lr-note" id="lr-note-{e["n"]}" data-target="lr-box-{e["n"]}" '
                f'data-top="{round(top, 2)}" style="top:{_pct(top)}" '
                f'onclick="lrFlash(\'lr-box-{e["n"]}\')">'
                f'<span class="lr-note-num">{e["n"]}</span>'
                f'<span class="lr-note-body">'
                f'<span class="lr-kind lr-kind-{e["kind"]}">{html.escape(e["kind"])}</span>'
                f'<span class="lr-note-text">{html.escape(e["note"])}</span>'
                f'</span></div>'
            )
        notes_html = "".join(notes) or '<div class="lr-note-empty">No notes on this page.</div>'

        parts.append(
            f'<div class="lr-row">'
            f'<div class="lr-pagewrap">'
            f'<div class="lr-page-num">p.{pno + 1}</div>'
            f'<div class="lr-page">'
            f'<img class="lr-page-img" src="{html.escape(url)}" loading="lazy" alt="page {pno + 1}">'
            f'{"".join(overlay)}'
            f'</div></div>'
            f'<div class="lr-notes">{notes_html}</div>'
            f'</div>'
        )
    return "".join(parts)


def _header_html(meta: Dict[str, Any]) -> str:
    title = html.escape(str(meta.get("Title", "Unknown")))
    authors = html.escape(str(meta.get("Authors", "Unknown")))
    return (
        f'<div class="lr-header"><h2 class="lr-title">{title}</h2>'
        f'<p class="lr-authors">{authors}</p></div>'
    )


def _png_urls(paths: List[str]) -> List[str]:
    return [f"/gradio_api/file={p}" for p in paths]


def _pngs_exist(paths: List[str]) -> bool:
    return bool(paths) and all(os.path.exists(p) for p in paths)


# -----------------------------------------------------------------------------
# Orchestration — generator yielding (status_md, reader_html) for Gradio.
# -----------------------------------------------------------------------------
def annotate_paper(url: str, provider: str, force: bool = False):
    model_name = PROVIDERS[provider]["model"]
    placeholder = '<div class="lr-empty">Annotated paper will appear here.</div>'

    aid = core.extract_arxiv_id(url)
    if not aid:
        yield "❌ **Invalid URL.** Could not find an ArXiv ID.", placeholder
        return

    ckey = core.cache_key(aid, provider, model_name, ANNOTATE_VERSION)
    cached = core.cache_get(ckey)
    if cached and not force and _pngs_exist(cached.get("pngs", [])):
        reader = _header_html(cached.get("meta", {})) + render_reader_html(
            cached["pages_meta"], _png_urls(cached["pngs"]), cached["annotations"]
        )
        yield f"⚡ Cached — `{aid}` served instantly, 0 tokens used.", reader
        return

    env_key = PROVIDERS[provider]["env_key"]
    api_key = os.getenv(env_key, "").strip()
    if not api_key:
        yield f"❌ **Missing `{env_key}` in environment.** Set it in `.env`.", placeholder
        return

    t_start = time.monotonic()

    # ---- Fetch PDF with simple 429/5xx backoff -----------------------------
    pdf_bytes: Optional[bytes] = None
    meta: Optional[Dict[str, Any]] = None
    for attempt in range(MAX_FETCH_RETRIES):
        yield (
            f"⏳ Fetching paper `{aid}`… (attempt {attempt + 1}/{MAX_FETCH_RETRIES})",
            placeholder,
        )
        try:
            pdf_bytes, meta = core.fetch_pdf(aid)
            break
        except Exception as err:
            status = getattr(err, "code", None) or getattr(err, "status", None)
            retryable = status == 429 or (isinstance(status, int) and 500 <= status < 600)
            if retryable and attempt < MAX_FETCH_RETRIES - 1:
                wait = (attempt + 1) * 5
                yield (
                    f"⏳ ArXiv HTTP {status}. Retrying in {wait}s "
                    f"(attempt {attempt + 2}/{MAX_FETCH_RETRIES})…",
                    placeholder,
                )
                time.sleep(wait)
                continue
            yield f"❌ **Fetch failed:** `{type(err).__name__}: {err}`", placeholder
            return

    if pdf_bytes is None:
        yield f"❌ **Paper `{aid}` not found.**", placeholder
        return

    header = _header_html(meta or {})

    yield f"✅ Fetched. Extracting layout for `{aid}`…", header + placeholder
    pages_meta, blocks = extract_blocks(pdf_bytes)
    pngs = render_page_pngs(pdf_bytes, aid)
    valid_ids = {b["id"] for b in blocks}
    manifest = build_manifest(blocks)

    try:
        llm = core.make_llm(provider, api_key, streaming=False)
    except Exception as e:
        yield f"❌ **LLM init failed:** `{type(e).__name__}: {e}`", header + placeholder
        return

    messages = [SystemMessage(content=ANNOTATE_PROMPT), HumanMessage(content=manifest)]

    # ---- LLM call on a worker thread + live elapsed timer ------------------
    t_gen = time.monotonic()
    result_q: queue.Queue = queue.Queue()

    def _invoke():
        try:
            result_q.put(("ok", llm.invoke(messages)))
        except Exception as exc:
            result_q.put(("error", exc))

    threading.Thread(target=_invoke, daemon=True).start()
    resp = None
    gen_error: Optional[Exception] = None
    while True:
        try:
            kind, val = result_q.get(timeout=0.5)
        except queue.Empty:
            elapsed = time.monotonic() - t_gen
            yield (
                f"🔎 Reading & selecting key blocks… {elapsed:.1f}s",
                header + '<div class="lr-empty">Selecting key blocks…</div>',
            )
            continue
        if kind == "ok":
            resp = val
        else:
            gen_error = val
        break

    if gen_error is not None:
        yield (
            f"❌ **Annotation failed:** `{type(gen_error).__name__}: {gen_error}`",
            header + placeholder,
        )
        return

    in_tok = out_tok = 0
    usage = getattr(resp, "usage_metadata", None) or {}
    if usage:
        in_tok = usage.get("input_tokens", 0) or 0
        out_tok = usage.get("output_tokens", 0) or 0

    cleaned = parse_annotations(getattr(resp, "content", "") or "", valid_ids)
    enriched = layout_annotations(cleaned, blocks, pages_meta)
    with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf:
        resolve_highlights(pdf, enriched, blocks)

    reader = header + render_reader_html(pages_meta, _png_urls(pngs), enriched)

    core.cache_put(ckey, {
        "meta": {"Title": (meta or {}).get("Title"), "Authors": (meta or {}).get("Authors")},
        "pages_meta": pages_meta,
        "annotations": enriched,
        "pngs": pngs,
    })

    total = time.monotonic() - t_start
    cost = core.estimate_cost(model_name, in_tok, out_tok)
    yield (
        f"✅ Done — `{aid}` in {total:.1f}s · {len(enriched)} annotations · "
        f"{in_tok:,}+{out_tok:,} tok · ${cost:.4f}",
        reader,
    )
