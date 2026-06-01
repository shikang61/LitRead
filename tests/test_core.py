"""Tests for shared helpers in core.py (extracted from app.py)."""
import json

import pytest

import core


# ---- extract_arxiv_id -------------------------------------------------------
@pytest.mark.parametrize(
    "text,expected",
    [
        ("https://arxiv.org/abs/2305.10601", "2305.10601"),
        ("https://arxiv.org/pdf/2305.10601.pdf", "2305.10601"),
        ("https://arxiv.org/abs/2305.10601v3", "2305.10601v3"),
        ("2305.10601", "2305.10601"),
        ("  https://arxiv.org/abs/2305.10601/  ", "2305.10601"),
        ("not a paper", None),
        ("", None),
    ],
)
def test_extract_arxiv_id(text, expected):
    assert core.extract_arxiv_id(text) == expected


# ---- cache round-trip + versioned key ---------------------------------------
def test_cache_put_then_get_roundtrips(monkeypatch, tmp_path):
    monkeypatch.setattr(core, "CACHE_DIR", str(tmp_path))
    key = core.cache_key("2305.10601", "OpenAI", "gpt-5.4", "annot-v1")
    payload = {"annotations": [{"block_id": "p1_b2", "note": "hi"}]}
    core.cache_put(key, payload)
    assert core.cache_get(key) == payload


def test_cache_get_missing_returns_none(monkeypatch, tmp_path):
    monkeypatch.setattr(core, "CACHE_DIR", str(tmp_path))
    assert core.cache_get("does-not-exist") is None


def test_cache_key_changes_with_version():
    a = core.cache_key("2305.10601", "OpenAI", "gpt-5.4", "v1")
    b = core.cache_key("2305.10601", "OpenAI", "gpt-5.4", "v2")
    assert a != b


# ---- tolerant JSON parser ---------------------------------------------------
def test_parse_json_plain():
    assert core.parse_json('{"a": 1}') == {"a": 1}


def test_parse_json_strips_code_fence():
    raw = '```json\n{"a": 1}\n```'
    assert core.parse_json(raw) == {"a": 1}


def test_parse_json_tolerates_trailing_comma():
    assert core.parse_json('{"a": 1,}') == {"a": 1}


def test_parse_json_tolerates_bad_latex_escape():
    # LLMs emit invalid JSON escapes like \( from LaTeX; parser must recover
    # and keep the backslash in the value.
    out = core.parse_json(r'{"eq": "see \(x=y\) here"}')
    assert "eq" in out


def test_parse_json_repairs_truncated():
    # Streamed/cut-off output should still yield the parsed prefix.
    out = core.parse_json('{"a": 1, "b": "unterminated')
    assert out["a"] == 1
