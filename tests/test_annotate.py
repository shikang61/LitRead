"""Tests for annotate.py — PDF block extraction, coordinate math, annotation
parsing/layout, highlight resolution, and reader HTML rendering.

Uses a tiny synthetic PDF built in-memory with fitz; no network/LLM calls.
"""
import json
import os

import fitz
import pytest

import annotate


def _make_pdf() -> bytes:
    """Two paragraphs on one 300x400 page."""
    doc = fitz.open()
    page = doc.new_page(width=300, height=400)
    page.insert_text((50, 100), "Contrastive learning beats the baselines")
    page.insert_text((50, 250), "A second important paragraph follows here")
    return doc.tobytes()


# ---- extract_blocks ---------------------------------------------------------
def test_extract_blocks_page_meta():
    pages_meta, _ = annotate.extract_blocks(_make_pdf())
    assert len(pages_meta) == 1
    assert pages_meta[0]["width"] == 300
    assert pages_meta[0]["height"] == 400
    assert pages_meta[0]["page"] == 0


def test_extract_blocks_stable_ids_and_text():
    _, blocks = annotate.extract_blocks(_make_pdf())
    assert len(blocks) >= 2
    text_blocks = [b for b in blocks if b["kind"] == "text" and b["text"]]
    assert any("Contrastive" in b["text"] for b in text_blocks)
    for b in blocks:
        assert b["id"].startswith("p0_b")
        assert b["page"] == 0
        assert len(b["bbox"]) == 4
        # bbox lies within the page
        x0, y0, x1, y1 = b["bbox"]
        assert 0 <= x0 <= x1 <= 300
        assert 0 <= y0 <= y1 <= 400


# ---- bbox_to_pct ------------------------------------------------------------
def test_bbox_to_pct():
    pct = annotate.bbox_to_pct((30, 40, 90, 240), 300, 400)
    assert pct["left"] == pytest.approx(10.0)
    assert pct["top"] == pytest.approx(10.0)
    assert pct["width"] == pytest.approx(20.0)
    assert pct["height"] == pytest.approx(50.0)


# ---- parse_annotations ------------------------------------------------------
VALID = {"p0_b0", "p0_b1"}


def test_parse_annotations_happy():
    raw = (
        '{"annotations": [{"block_id": "p0_b0", "kind": "equation", '
        '"note": "the key claim", "keywords": ["Contrastive"]}]}'
    )
    out = annotate.parse_annotations(raw, VALID)
    assert len(out) == 1
    assert out[0]["block_id"] == "p0_b0"
    assert out[0]["kind"] == "equation"
    assert out[0]["note"] == "the key claim"
    assert out[0]["keywords"] == ["Contrastive"]


def test_parse_annotations_drops_unknown_block_id():
    raw = '{"annotations": [{"block_id": "p9_b9", "note": "ghost"}]}'
    assert annotate.parse_annotations(raw, VALID) == []


def test_parse_annotations_drops_missing_note():
    raw = '{"annotations": [{"block_id": "p0_b0", "keywords": ["x"]}]}'
    assert annotate.parse_annotations(raw, VALID) == []


def test_parse_annotations_defaults_kind_and_coerces_keywords():
    raw = '{"annotations": [{"block_id": "p0_b0", "note": "n", "keywords": "notalist"}]}'
    out = annotate.parse_annotations(raw, VALID)
    assert out[0]["kind"] == "paragraph"
    assert out[0]["keywords"] == []


def test_parse_annotations_keeps_section_kind():
    raw = '{"annotations": [{"block_id": "p0_b0", "kind": "section", "note": "what this section does"}]}'
    out = annotate.parse_annotations(raw, VALID)
    assert out[0]["kind"] == "section"


def test_parse_annotations_tolerates_code_fence():
    raw = '```json\n{"annotations": [{"block_id": "p0_b1", "note": "n"}]}\n```'
    out = annotate.parse_annotations(raw, VALID)
    assert len(out) == 1


# ---- layout_annotations -----------------------------------------------------
def test_layout_orders_by_page_then_top_and_numbers():
    blocks = [
        {"id": "p0_b0", "page": 0, "bbox": (10, 300, 90, 340), "kind": "text", "text": "lower"},
        {"id": "p0_b1", "page": 0, "bbox": (10, 50, 90, 90), "kind": "text", "text": "upper"},
    ]
    pages_meta = [{"page": 0, "width": 100, "height": 400}]
    anns = [
        {"block_id": "p0_b0", "kind": "paragraph", "note": "lower", "keywords": []},
        {"block_id": "p0_b1", "kind": "paragraph", "note": "upper", "keywords": []},
    ]
    out = annotate.layout_annotations(anns, blocks, pages_meta)
    # numbered top-to-bottom: the upper block (y=50) becomes #1
    assert out[0]["note"] == "upper"
    assert out[0]["n"] == 1
    assert out[1]["note"] == "lower"
    assert out[1]["n"] == 2
    # box carries percentage geometry
    assert out[0]["box"]["top"] == pytest.approx(12.5)
    assert out[0]["page"] == 0
    assert out[0]["highlights"] == []


# ---- resolve_highlights (needs a real fitz page) ---------------------------
def test_resolve_highlights_found_and_missing():
    pdf_bytes = _make_pdf()
    pages_meta, blocks = annotate.extract_blocks(pdf_bytes)
    target = next(b for b in blocks if "Contrastive" in b["text"])
    enriched = annotate.layout_annotations(
        [
            {"block_id": target["id"], "kind": "paragraph", "note": "n",
             "keywords": ["Contrastive", "ZZZ-not-present"]},
        ],
        blocks,
        pages_meta,
    )
    with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf:
        annotate.resolve_highlights(pdf, enriched, blocks)
    # "Contrastive" found → at least one highlight; the bogus word adds none
    assert len(enriched[0]["highlights"]) >= 1
    for h in enriched[0]["highlights"]:
        assert set(h) == {"left", "top", "width", "height"}


# ---- render_reader_html -----------------------------------------------------
def test_render_reader_html_contains_image_box_and_note():
    pages_meta = [{"page": 0, "width": 300, "height": 400}]
    png_urls = ["/file=cache/pages/x_p0.png"]
    enriched = [
        {"n": 1, "block_id": "p0_b1", "page": 0, "kind": "equation",
         "note": "the <core> idea", "keywords": ["Contrastive"],
         "box": {"left": 16.0, "top": 12.5, "width": 26.0, "height": 10.0},
         "highlights": [{"left": 17.0, "top": 13.0, "width": 10.0, "height": 3.0}]},
    ]
    out = annotate.render_reader_html(pages_meta, png_urls, enriched)
    assert "/file=cache/pages/x_p0.png" in out
    assert "lr-box-1" in out            # box anchor id for click-to-nav
    assert "16.0%" in out               # box left as percentage
    assert "the &lt;core&gt; idea" in out  # note text HTML-escaped
    assert "lr-hl" in out               # keyword highlight element
    assert "equation" in out            # kind tag shown
    # note is anchored to its box's vertical position (top = box top)
    assert 'data-top="12.5"' in out
    assert "top:12.5%" in out


def test_render_reader_html_empty_message():
    pages_meta = [{"page": 0, "width": 300, "height": 400}]
    out = annotate.render_reader_html(pages_meta, ["/file=x.png"], [])
    assert "nothing" in out.lower() or "no " in out.lower()


# ---- full pipeline on a 2-page synthetic PDF (no network/LLM) --------------
def test_full_pipeline_multipage(tmp_path, monkeypatch):
    monkeypatch.setattr(annotate, "PNG_DIR", str(tmp_path / "pages"))
    doc = fitz.open()
    p0 = doc.new_page(width=300, height=400)
    p0.insert_text((40, 80), "Transformers replace recurrence entirely")
    p1 = doc.new_page(width=300, height=400)
    p1.insert_text((40, 120), "Self attention scales with sequence length")
    pdf_bytes = doc.tobytes()

    pages_meta, blocks = annotate.extract_blocks(pdf_bytes)
    assert len(pages_meta) == 2

    pngs = annotate.render_page_pngs(pdf_bytes, "synthetic")
    assert len(pngs) == 2
    assert all(os.path.exists(p) for p in pngs)

    manifest = annotate.build_manifest(blocks)
    assert "Transformers" in manifest and "p1_b" in manifest

    b0 = next(b for b in blocks if b["page"] == 0 and b["text"])
    b1 = next(b for b in blocks if b["page"] == 1 and b["text"])
    fake = json.dumps({"annotations": [
        {"block_id": b0["id"], "kind": "paragraph", "note": "page0 takeaway", "keywords": []},
        {"block_id": b1["id"], "kind": "figure", "note": "page1 takeaway", "keywords": []},
    ]})
    cleaned = annotate.parse_annotations(fake, {b["id"] for b in blocks})
    enriched = annotate.layout_annotations(cleaned, blocks, pages_meta)
    with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf:
        annotate.resolve_highlights(pdf, enriched, blocks)

    html = annotate.render_reader_html(pages_meta, annotate._png_urls(pngs), enriched)
    assert html.count('class="lr-row"') == 2   # one row per page
    assert "page0 takeaway" in html and "page1 takeaway" in html
