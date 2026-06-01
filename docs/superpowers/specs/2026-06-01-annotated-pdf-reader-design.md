# Annotated PDF Reader — Design

**Date:** 2026-06-01
**Status:** Approved (design), ready for implementation
**Branch:** `feat/annotate-paper`

## Goal

Add a second mode to LitRead: given an ArXiv link, produce an **in-app side-by-side
annotated reader** of the paper. For each PDF page, the page renders as an image with
boxes drawn on the high-signal regions and keyword highlights inside them; a notes
column on the right gives a one-line gloss of each boxed region. The reader can skim
the notes to get the gist "without the fluff", then click a note to jump to its box for
detail.

This is a next step after the existing carousel summary — the carousel gives the
paper's story; this gives a navigable, skimmable overlay on the actual paper.

## Decisions (locked with user)

1. **Output target: in-app side-by-side view.** Pages rendered as images in Gradio with
   an HTML notes column beside them. No downloadable annotated PDF in v1.
2. **Density: selective.** The LLM flags only high-signal paragraphs / equations /
   figures (~5–10 per page), skipping filler/boilerplate.
3. **Coordinate mapping: block-ID.** The LLM references PDF block IDs, not verbatim
   quotes. Boxing is an exact bbox lookup; no fuzzy quote matching.
4. **One whole-document LLM call**, not per-page — "selective" needs global importance
   judgment across the whole paper. Reuses the existing cache pattern.
5. **Overlay rendering**, not burned-in boxes — keeps boxes/notes interactive
   (click note → scroll to + flash its box).

## Architecture / data flow

```
URL ─ extract_arxiv_id ─ fetch PDF bytes (reuse 429/5xx retry+backoff)
   └─ fitz: per page → blocks{id,page,bbox,type,text} + render page PNG (2x)
   └─ build compact block manifest (strip refs, mark image blocks as [FIGURE])
   └─ LLM (reuse make_llm, JSON mode, new ANNOTATE_PROMPT) →
        {annotations:[{block_id, kind:paragraph|equation|figure, note, keywords[]}]}
   └─ map block_id → bbox; per keyword: page.search_for(kw, clip=bbox) → highlight quads
   └─ render HTML: [page <img> + overlay boxes/highlights/markers] | [notes column]
   └─ cache annotation JSON (key: arxiv_id|provider|model|ANNOTATE_VERSION)
```

## Components

- **`core.py`** (new, small): shared helpers moved out of `app.py` so both UIs import
  one copy and `annotate.py` avoids a circular import with `app.py`:
  `extract_arxiv_id`, `strip_references`, `make_llm`, `PROVIDERS`/model config,
  `cache_get/put/key` (parametrised by a prompt-version arg), tolerant JSON parser,
  and the arxiv fetch-with-retry loop factored into a reusable generator/function.
- **`annotate.py`** (new): the new feature's logic —
  - `extract_blocks(pdf_bytes)` → `(pages_meta, blocks)`: per-page size + block list
    with stable IDs (`p{page}_b{idx}`), bbox, kind (text/image), text.
  - `render_page_pngs(pdf_bytes, out_dir)` → list of PNG paths (2x zoom).
  - `build_manifest(blocks)` → compact text sent to the LLM.
  - `ANNOTATE_PROMPT` + `parse_annotations(raw)` → validated annotation list.
  - `resolve_highlights(pdf, annotations, blocks)` → highlight quads per annotation via
    `page.search_for(kw, clip=block_bbox)`.
  - `render_reader_html(pages_meta, png_urls, blocks, annotations)` → the side-by-side
    HTML (page image + absolutely-positioned overlay + notes column).
- **`app.py`**: add a second button **"🔎 Annotate Paper"** beside Generate (same URL
  bar + provider dropdown). New full-width reader output area. Carousel untouched.

## Coordinate model

`fitz` page coords are PDF points (origin top-left in PyMuPDF's `get_text`/`Rect`).
Page PNG rendered at zoom `Z` (e.g. 2.0). Overlay div placed over the `<img>` using
percentage coords derived from bbox / page size (so it scales with the responsive
image): `left% = bbox.x0 / page_width * 100`, etc. Highlight quads from `search_for`
are mapped the same way. Percentages avoid hard-coding the rendered pixel size.

## Error handling

- Fetch: reuse existing 429/5xx retry + backoff.
- LLM JSON: reuse tolerant parser.
- Hallucinated `block_id` not in the block map → skip that annotation.
- `search_for` keyword miss → skip the highlight; still show the box + note.
- Page render failure → show the page without overlay.
- Zero valid annotations → friendly "nothing flagged" message.

## Testing (TDD)

- **Unit:**
  - `extract_blocks` returns stable IDs + plausible bboxes on a fixture PDF.
  - bbox → percentage-coord conversion math.
  - keyword → quad mapping (incl. miss returns empty, not error).
  - `parse_annotations` tolerant of fences / bad escapes; drops unknown `block_id`s and
    malformed entries.
- **Integration:** build a tiny synthetic PDF in-memory with `fitz`, run
  extract → (mocked LLM annotations) → render, assert N boxes + notes present in HTML,
  no exception. No real network / API call in tests.

## Caching

- Annotation JSON cached by `arxiv_id|provider|model|ANNOTATE_VERSION` (same scheme as
  carousel, bump `ANNOTATE_VERSION` on prompt change).
- Page PNGs cached per `arxiv_id` under the cache dir; reused across runs.

## Known v1 limits

- Page images are not selectable text (accepted tradeoff of in-app view).
- Equations rendered as inline/vector text may box imperfectly; image-block figures box
  cleanly.
- No downloadable annotated PDF — revisit later via a widened-margin export path.

## Implementation order

1. `core.py`: extract shared helpers from `app.py`; keep `app.py` importing them
   (carousel still works). Verify: app imports + carousel path unchanged.
2. `annotate.py` block extraction + coord math + parser, with unit tests (TDD).
3. `annotate.py` render + highlight resolution, with tests on a synthetic PDF.
4. Wire the "Annotate Paper" button + reader output into `app.py`.
5. Manual smoke: run app, annotate a known paper, confirm boxes/notes/nav.
