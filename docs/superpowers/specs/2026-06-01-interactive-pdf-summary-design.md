# Interactive PDF Region Summary — Design

**Date:** 2026-06-01
**Status:** Approved (design), ready for implementation
**Branch:** `feat/annotate-paper`

## Goal

Replace the automatic annotation pass with an **interactive** mode: the "Annotate Paper"
button opens the paper as plain page images; the user **drags a bounding box** over any
region and gets an **on-demand layman summary** of the text in that box. Summaries
accumulate as a numbered running list in a right-hand panel, with a matching numbered
marker left on the page. The API Usage panel reports cumulative tokens/cost for the
session's summaries.

## Decisions (locked with user)

1. **Replace** the auto-annotation pass — the old LLM-picks-blocks pipeline is removed.
2. **Right-side panel, running list** — each selection adds a numbered card; markers
   persist on the page; list grows as you explore (session-scoped, not across reload).
3. **Selection round-trip:** JS rubber-band → hidden `Textbox` (JSON) → hidden `Button`
   click (the proven recent-papers pattern). Gradio's `Image.select` can't capture a drag.
4. **Box → text:** `fitz` `page.get_text("text", clip=rect)`.
5. **API Usage panel updates for this mode** (cumulative across the session's summaries).

## Architecture / data flow

```
"Annotate Paper" ─ load_paper(url, provider, force):
    fetch PDF (arXiv 429 retry) + render page PNGs
    init session gr.State {aid, pdf_bytes, pages_meta, pngs, title, authors,
                           summaries:[], in_tok:0, out_tok:0, model}
    render reader (plain page images + drag layer per page + empty panel)
  → yields (status, reader_html, cost_html, state)

draw box (JS) ─ #lr-selection Textbox = {"page":p, "rect":[x0,y0,x1,y1]}  (0-1 fractions)
             ─ #lr-summarize Button.click ─ summarize_region(state, selection, provider):
                 sel = parse_selection(...)             # validate, ignore tiny/bad
                 text = extract_region_text(pdf_bytes, page, rect)  # fitz clip
                 if text: summary, in, out = LLM(REGION_PROMPT, text)
                 else:    summary = "No selectable text in that region." (0 tokens)
                 state = append_summary(state, sel, text, summary, in, out)
                 re-render reader (markers from state) + panel + cost
             ─ outputs (reader_html, cost_html, state)
```

## Components

- **`annotate.py`** (repurposed):
  - `load_paper(url, provider, force)` — generator yielding (status, reader, cost, state).
  - `extract_region_text(pdf_bytes, page_index, rect)` — fitz clip → text (pure-ish).
  - `parse_selection(selection_json, n_pages)` — validate → dict or None (pure).
  - `append_summary(state, sel, text, summary, in_tok, out_tok)` — returns new state (pure).
  - `summarize_region(state, selection_json, provider)` — orchestration: parse → extract →
    LLM → append → render. Returns (reader_html, cost_html, state).
  - `REGION_PROMPT` — plain-text layman explanation (2-4 sentences; explains equations too).
  - `render_reader(state)` — page images + drag layer + numbered selection markers + panel.
  - Keep: `render_page_pngs`, `_png_urls`.
- **Removed (replaced):** `ANNOTATE_PROMPT`, `ANNOTATE_VERSION`, `VALID_KINDS`,
  `extract_blocks`, `build_manifest`, `parse_annotations`, `layout_annotations`,
  `resolve_highlights`, `bbox_to_pct`, old `render_reader_html`, old `annotate_paper`, and
  their tests.
- **`app.py`:** `paper_state = gr.State(None)`; hidden `selection` Textbox + `summarize_btn`
  (CSS-hidden via elem_id); wire `annotate_btn → load_paper` (outputs status, reader, cost,
  state) and `summarize_btn → summarize_region` (outputs reader, cost, state). New JS
  rubber-band capture; CSS for the draw layer, selection markers, and summary cards.

## State shape (gr.State)

```python
{
  "aid": str, "pdf_bytes": bytes, "pages_meta": [{page,width,height}],
  "pngs": [str], "title": str, "authors": str,
  "summaries": [{ "n": int, "page": int, "rect": [x0,y0,x1,y1], "text": str, "summary": str }],
  "in_tok": int, "out_tok": int, "model": str,
}
```

## Interaction details

- Drag smaller than ~1% of the page in either dimension → ignored (stray click).
- Each selection: numbered marker box on the page (from rect fractions → %), numbered card
  in that page's panel column (ordered). Click a card → scroll to + flash its marker.
- Box over a figure / no text → card says "No selectable text in that region." (no LLM call).
- Same Model dropdown + env key; API Usage panel = cumulative session tokens/cost.

## Error handling

- No state / paper not loaded → summarize is a no-op (render prompt to load first).
- Bad/empty/tiny selection JSON → ignored (state unchanged).
- LLM error → card shows the error text; cost unchanged.
- arXiv 429 retry reused on load.

## Testing (TDD)

- `extract_region_text`: synthetic fitz PDF — clip over known text returns it; off-text clip
  returns "".
- `parse_selection`: valid JSON → dict; out-of-range page, non-list rect, tiny box,
  malformed → None.
- `append_summary`: appends with incrementing `n`, accumulates tokens, preserves prior.
- Network smoke (gated) updated for the new load flow.

## Known v1 limits

- Page images aren't selectable text (unchanged tradeoff).
- Figure/image regions yield no text (no vision description yet).
- Summaries are session-scoped (not cached/persisted across reload).
