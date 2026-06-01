"""Tests for annotate.py — interactive region summary.

Region text extraction, selection parsing, summary-list state, and reader HTML.
Uses a tiny synthetic PDF built in-memory with fitz; no network/LLM calls.
"""
import json

import fitz
import pytest

import annotate
import core


def _make_pdf() -> bytes:
    """One 300x400 page with text near the top third."""
    doc = fitz.open()
    page = doc.new_page(width=300, height=400)
    page.insert_text((40, 100), "Contrastive learning beats the baselines")
    return doc.tobytes()


# ---- extract_region_text ----------------------------------------------------
def test_extract_region_text_hits_text():
    # text baseline ~y=100 on a 400-tall page -> fraction ~0.25
    txt = annotate.extract_region_text(_make_pdf(), 0, [0.0, 0.18, 1.0, 0.32])
    assert "Contrastive" in txt


def test_extract_region_text_empty_region():
    txt = annotate.extract_region_text(_make_pdf(), 0, [0.0, 0.85, 0.3, 0.98])
    assert txt == ""


def test_extract_region_text_out_of_range_page():
    assert annotate.extract_region_text(_make_pdf(), 9, [0.0, 0.0, 1.0, 1.0]) == ""


# ---- parse_selection --------------------------------------------------------
def test_parse_selection_valid_and_sorted():
    out = annotate.parse_selection('{"page":0,"rect":[0.5,0.4,0.1,0.2]}', n_pages=1)
    assert out["page"] == 0
    # corners get sorted to [x0<x1, y0<y1]
    assert out["rect"] == [0.1, 0.2, 0.5, 0.4]


def test_parse_selection_page_out_of_range():
    assert annotate.parse_selection('{"page":5,"rect":[0.1,0.2,0.5,0.4]}', n_pages=1) is None


def test_parse_selection_too_small():
    assert annotate.parse_selection('{"page":0,"rect":[0.1,0.2,0.105,0.4]}', n_pages=1) is None


def test_parse_selection_malformed():
    assert annotate.parse_selection("not json", n_pages=1) is None
    assert annotate.parse_selection('{"page":0,"rect":[0.1,0.2]}', n_pages=1) is None
    assert annotate.parse_selection('{"page":0}', n_pages=1) is None


# ---- append_summary ---------------------------------------------------------
def test_append_summary_numbers_and_accumulates_tokens():
    state = {"summaries": [], "in_tok": 0, "out_tok": 0}
    s1 = annotate.append_summary(state, {"page": 0, "rect": [0, 0, 1, 1]}, "txt", "sum1", 10, 5)
    assert s1["summaries"][0]["n"] == 1
    assert (s1["in_tok"], s1["out_tok"]) == (10, 5)
    s2 = annotate.append_summary(s1, {"page": 1, "rect": [0, 0, 1, 1]}, "t2", "sum2", 7, 3)
    assert s2["summaries"][1]["n"] == 2
    assert (s2["in_tok"], s2["out_tok"]) == (17, 8)
    # original state untouched (no mutation)
    assert state["summaries"] == []
    assert s1["summaries"] == [s1["summaries"][0]]


# ---- render_reader ----------------------------------------------------------
def test_summary_cache_roundtrip(monkeypatch, tmp_path):
    monkeypatch.setattr(core, "CACHE_DIR", str(tmp_path))
    state = {"aid": "1234.5678", "model": "grok-4.3",
             "summaries": [{"n": 1, "page": 0, "rect": [0, 0, 1, 1], "text": "t", "summary": "s"}]}
    annotate._persist_summaries(state, "Grok")
    got = core.cache_get(annotate._summary_cache_key("1234.5678", "Grok", "grok-4.3"))
    assert got["summaries"] == state["summaries"]


def test_delete_summary_removes_renumbers_and_recaches(monkeypatch, tmp_path):
    monkeypatch.setattr(core, "CACHE_DIR", str(tmp_path))
    state = {"aid": "1", "model": "grok-4.3", "in_tok": 0, "out_tok": 0,
             "summaries": [
                 {"n": 1, "page": 0, "rect": [0, 0, 1, 1], "text": "a", "summary": "A"},
                 {"n": 2, "page": 0, "rect": [0, 0, 1, 1], "text": "b", "summary": "B"},
                 {"n": 3, "page": 1, "rect": [0, 0, 1, 1], "text": "c", "summary": "C"},
             ]}
    _, _, new = annotate.delete_summary(state, "2", "Grok")
    assert [s["summary"] for s in new["summaries"]] == ["A", "C"]
    assert [s["n"] for s in new["summaries"]] == [1, 2]      # renumbered
    got = core.cache_get(annotate._summary_cache_key("1", "Grok", "grok-4.3"))
    assert [s["summary"] for s in got["summaries"]] == ["A", "C"]


def test_render_reader_no_state_placeholder():
    assert "lr-empty" in annotate.render_reader(None)


def test_render_reader_has_page_drawlayer_and_summary():
    state = {
        "title": "My <Paper>", "authors": "A. Author",
        "pages_meta": [{"page": 0, "width": 300, "height": 400}],
        "pngs": ["cache/pages/x_p0.png"],
        "summaries": [{"n": 1, "page": 0, "rect": [0.1, 0.2, 0.5, 0.4],
                       "text": "t", "summary": "in plain terms it says X"}],
        "in_tok": 0, "out_tok": 0, "model": "grok-4.3",
    }
    out = annotate.render_reader(state)
    assert 'class="lr-drawlayer" data-page="0"' in out   # drag capture layer
    assert "lr-page-img" in out
    assert "lr-box-1" in out                              # selection marker
    assert "10.0%" in out                                 # marker left = x0 0.1
    assert "in plain terms it says X" in out              # summary card
    assert "My &lt;Paper&gt;" in out                      # title escaped
