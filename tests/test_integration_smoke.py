"""End-to-end pipeline smoke against a real arXiv PDF, minus the LLM call.

Network-gated: set RUN_NET_SMOKE=1 to run. Verifies fetch -> page rasterisation
-> region text extraction -> selection parse -> summary state -> reader HTML,
against a genuine paper.
"""
import os
import time

import pytest

import annotate
import core

pytestmark = pytest.mark.skipif(
    not os.environ.get("RUN_NET_SMOKE"),
    reason="network smoke test; set RUN_NET_SMOKE=1 to run",
)


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


def test_real_paper_region_pipeline_without_llm(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(annotate, "PNG_DIR", str(tmp_path / "pages"))

    aid = "1706.03762"  # Attention Is All You Need
    pdf_bytes, meta = _fetch_with_retry(aid)
    assert pdf_bytes and meta and meta["Title"]

    import fitz
    with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf:
        pages_meta = [{"page": i, "width": p.rect.width, "height": p.rect.height}
                      for i, p in enumerate(pdf)]
    pngs = annotate.render_page_pngs(pdf_bytes, aid)
    assert len(pngs) == len(pages_meta)
    assert all(os.path.exists(p) for p in pngs)

    # The top of page 0 (title/abstract) has text.
    text = annotate.extract_region_text(pdf_bytes, 0, [0.1, 0.05, 0.9, 0.5])
    assert len(text) > 20

    sel = annotate.parse_selection('{"page":0,"rect":[0.1,0.05,0.9,0.5]}', len(pages_meta))
    assert sel is not None

    state = {"title": meta["Title"], "authors": meta["Authors"], "pages_meta": pages_meta,
             "pngs": pngs, "summaries": [], "in_tok": 0, "out_tok": 0, "model": "grok-4.3"}
    state = annotate.append_summary(state, sel, text, "plain summary", 100, 40)
    html = annotate.render_reader(state)
    assert "lr-box-1" in html and "lr-drawlayer" in html
