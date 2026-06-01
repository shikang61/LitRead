"""End-to-end pipeline smoke against a real arXiv PDF, minus the LLM call.

Network-gated: set RUN_NET_SMOKE=1 to run. Verifies fetch -> block extraction
-> page rasterisation -> manifest -> (faked annotations) -> layout ->
highlight resolution -> reader HTML, against a genuine paper.
"""
import json
import os
import time

import fitz
import pytest

import annotate
import core


def _fetch_with_retry(aid, tries=5):
    """arXiv's metadata API rate-limits aggressively; back off on 429/5xx."""
    last = None
    for i in range(tries):
        try:
            return core.fetch_pdf(aid)
        except Exception as err:
            last = err
            status = getattr(err, "code", None) or getattr(err, "status", None)
            if status == 429 or (isinstance(status, int) and 500 <= status < 600):
                time.sleep((i + 1) * 8)
                continue
            raise
    raise last

pytestmark = pytest.mark.skipif(
    not os.environ.get("RUN_NET_SMOKE"),
    reason="network smoke test; set RUN_NET_SMOKE=1 to run",
)


def test_real_paper_pipeline_without_llm(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(annotate, "PNG_DIR", str(tmp_path / "pages"))

    aid = "1706.03762"  # Attention Is All You Need
    pdf_bytes, meta = _fetch_with_retry(aid)
    assert pdf_bytes and meta and meta["Title"]

    pages_meta, blocks = annotate.extract_blocks(pdf_bytes)
    assert pages_meta and blocks

    pngs = annotate.render_page_pngs(pdf_bytes, aid)
    assert len(pngs) == len(pages_meta)
    assert all(os.path.exists(p) for p in pngs)

    manifest = annotate.build_manifest(blocks)
    assert "p0_b" in manifest

    # Fake an LLM response from two real text blocks; highlight their first word.
    text_blocks = [b for b in blocks if b["kind"] == "text" and len(b["text"]) > 40][:2]
    assert len(text_blocks) == 2
    fake = {"annotations": [
        {"block_id": tb["id"], "kind": "paragraph", "note": "key takeaway",
         "keywords": [tb["text"].split()[0]]}
        for tb in text_blocks
    ]}

    valid_ids = {b["id"] for b in blocks}
    cleaned = annotate.parse_annotations(json.dumps(fake), valid_ids)
    assert len(cleaned) == 2

    enriched = annotate.layout_annotations(cleaned, blocks, pages_meta)
    with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf:
        annotate.resolve_highlights(pdf, enriched, blocks)
    # at least one keyword should resolve to a highlight rect
    assert sum(len(e["highlights"]) for e in enriched) >= 1

    html = annotate.render_reader_html(pages_meta, annotate._png_urls(pngs), enriched)
    assert "lr-box-1" in html and "lr-box-2" in html
    assert "lr-page-img" in html
