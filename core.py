"""Shared helpers for LitRead — used by both the carousel (app.py) and the
annotated reader (annotate.py).

Holds the pieces both entrypoints need: provider/model config, ArXiv ID
parsing, the LLM factory, the carousel/annotation cache, a tolerant JSON
parser, and the network fetch that downloads a paper's PDF bytes + metadata.
Kept in its own module so annotate.py can reuse them without a circular import
against app.py (which imports annotate.py to wire its UI button).
"""

import hashlib
import html
import json
import os
import re
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

import arxiv
import fitz  # pymupdf
from langchain_openai import ChatOpenAI

# Custom UA — arxiv aggressively rate-limits the default Python-urllib UA.
_opener = urllib.request.build_opener()
_opener.addheaders = [
    ("User-Agent", "ArxivCarousel/1.0 (research summarizer; mailto:user@example.com)"),
]
urllib.request.install_opener(_opener)

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
OPENAI_MODEL = "gpt-5.4"
GROK_MODEL = "grok-4.3"
GROK_BASE_URL = "https://api.x.ai/v1"

# Per-1M-token pricing in USD.
PRICING: Dict[str, Dict[str, float]] = {
    "gpt-5.4":  {"input": 5.00, "output": 15.00},
    "grok-4.3": {"input": 5.00, "output": 15.00},
}

# Context-window sizes (input tokens). Used by the cost panel meter.
MODEL_CONTEXT: Dict[str, int] = {
    "gpt-5.4":  400_000,
    "grok-4.3": 256_000,
}

PROVIDERS = {
    "Grok": {
        "model": GROK_MODEL,
        "base_url": GROK_BASE_URL,
        "env_key": "XAI_API_KEY",
    },
    "OpenAI": {
        "model": OPENAI_MODEL,
        "base_url": None,
        "env_key": "OPENAI_API_KEY",
    },
}

# Cache dir for cached LLM outputs. Use /data on HF Spaces with persistent
# storage; falls back to local ./cache otherwise.
CACHE_DIR = os.environ.get("LITREAD_CACHE_DIR", "cache")


# -----------------------------------------------------------------------------
# ArXiv ID parsing
# -----------------------------------------------------------------------------
ARXIV_ID_RE = re.compile(
    r"(?:arxiv\.org/(?:abs|pdf|html)/)?"
    r"(?P<id>\d{4}\.\d{4,5}(?:v\d+)?|[a-z\-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?)",
    re.IGNORECASE,
)


def extract_arxiv_id(text: str) -> Optional[str]:
    if not text:
        return None
    text = text.strip().rstrip("/")
    if text.lower().endswith(".pdf"):
        text = text[:-4]
    m = ARXIV_ID_RE.search(text)
    return m.group("id") if m else None


_REFERENCES_HEADER_RE = re.compile(
    r"(?im)^\s*(references|bibliography|acknowledg(?:e?)ments)\s*$"
)


def strip_references(text: str) -> str:
    """Cut the references/bibliography tail. Preserves >=40% of doc to avoid
    chopping a mid-paper section header by accident."""
    matches = list(_REFERENCES_HEADER_RE.finditer(text))
    if not matches:
        return text
    cut = matches[-1].start()
    if cut < len(text) * 0.4:
        return text
    return text[:cut].rstrip()


# -----------------------------------------------------------------------------
# Network fetch — download PDF bytes + metadata for an ArXiv ID.
# -----------------------------------------------------------------------------
def fetch_pdf(arxiv_id: str) -> Tuple[Optional[bytes], Optional[Dict[str, Any]]]:
    """Fetch a paper's raw PDF bytes and metadata. Returns (None, None) if the
    ID resolves to no paper. Raises on network errors (caller handles retry)."""
    # delay_seconds=0.5 (down from arxiv default 3.0) + num_retries=2 keeps the
    # metadata round-trip fast; backoff for transient HTTP errors is the caller's job.
    client = arxiv.Client(page_size=1, delay_seconds=0.5, num_retries=2)
    results = list(client.results(arxiv.Search(id_list=[arxiv_id])))
    if not results:
        return None, None
    result = results[0]

    # Hard 30s ceiling — without timeout, urlopen hangs forever when arxiv's CDN stalls.
    with urllib.request.urlopen(result.pdf_url, timeout=30) as resp:
        pdf_bytes = resp.read()
    with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf:
        page_count = pdf.page_count

    names = [a.name for a in result.authors]
    if not names:
        authors_display = "Unknown"
    elif len(names) == 1:
        authors_display = names[0]
    else:
        authors_display = f"{names[0]} et al."

    metadata = {
        "Title": result.title or "Unknown",
        "Authors": authors_display,
        "AuthorsAll": [a.name for a in result.authors],
        "Published": result.published.date().isoformat() if result.published else "",
        "Year": result.published.year if result.published else 0,
        "Pages": page_count,
        "entry_id": result.entry_id,
        "AbsURL": result.entry_id,
        "PdfURL": result.pdf_url,
        "Summary": (result.summary or "").strip(),
    }
    return pdf_bytes, metadata


# -----------------------------------------------------------------------------
# LLM factory + cost
# -----------------------------------------------------------------------------
def make_llm(provider: str, api_key: str, streaming: bool = True) -> ChatOpenAI:
    cfg = PROVIDERS[provider]
    kwargs: Dict[str, Any] = {
        "model": cfg["model"],
        "api_key": api_key,
        "streaming": streaming,
        "stream_usage": True,        # langchain-openai >=0.2: usage on final chunk
        "temperature": 0.3,          # tighter sampling → more reliable JSON
    }
    # Request strict JSON output when supported (OpenAI; Grok ignores or errors).
    if cfg["base_url"] is None:
        kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
    if cfg["base_url"]:
        kwargs["base_url"] = cfg["base_url"]

    try:
        return ChatOpenAI(**kwargs)
    except TypeError:
        # Older langchain-openai doesn't accept stream_usage; retry without.
        kwargs.pop("stream_usage", None)
        return ChatOpenAI(**kwargs)


_BOLD_RE = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)


def escape_with_bold(text: str) -> str:
    """HTML-escape, then convert `**term**` → `<strong>term</strong>`."""
    parts: List[str] = []
    last = 0
    for m in _BOLD_RE.finditer(text):
        parts.append(html.escape(text[last:m.start()]))
        parts.append(f'<strong>{html.escape(m.group(1))}</strong>')
        last = m.end()
    parts.append(html.escape(text[last:]))
    return "".join(parts)


def estimate_cost(model: str, in_tok: int, out_tok: int) -> float:
    p = PRICING.get(model)
    if not p:
        return 0.0
    return (in_tok * p["input"] + out_tok * p["output"]) / 1_000_000


def render_cost_html(model: str = "—", in_tok: int = 0, out_tok: int = 0) -> str:
    """API-usage card (model, input/output tokens, estimated cost). Shared by the
    carousel and the interactive reader."""
    cost = estimate_cost(model, in_tok, out_tok)
    cap = MODEL_CONTEXT.get(model)
    if cap:
        pct = max(0, min(100, int(in_tok / cap * 100))) if in_tok else 0
        # Colour the bar amber >70% / red >90% so over-budget runs are visible.
        bar_class = "cost-bar"
        if pct >= 90:
            bar_class += " cost-bar-red"
        elif pct >= 70:
            bar_class += " cost-bar-amber"
        input_row = (
            f'<div class="cost-row"><span>Input</span>'
            f'<span>{in_tok:,} / {cap:,} tok</span></div>'
            f'<div class="cost-meter"><div class="{bar_class}" style="width:{pct}%"></div></div>'
        )
    else:
        input_row = f'<div class="cost-row"><span>Input</span><span>{in_tok:,} tok</span></div>'
    return (
        '<div class="cost-card">'
        '<div class="cost-title">🪙 API Usage</div>'
        f'<div class="cost-row"><span>Model</span><span>{html.escape(model or "—")}</span></div>'
        f'{input_row}'
        f'<div class="cost-row"><span>Output</span><span>{out_tok:,} tok</span></div>'
        f'<div class="cost-row cost-total"><span>Cost</span><span>${cost:.4f}</span></div>'
        '</div>'
    )


# -----------------------------------------------------------------------------
# Cache (arxiv_id + provider + model + version -> JSON payload)
# -----------------------------------------------------------------------------
def cache_key(arxiv_id: str, provider: str, model: str, version: str) -> str:
    raw = f"{arxiv_id}|{provider}|{model}|{version}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def cache_get(key: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(CACHE_DIR, f"{key}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def cache_put(key: str, payload: Dict[str, Any]) -> None:
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        path = os.path.join(CACHE_DIR, f"{key}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
    except Exception:
        pass  # cache is best-effort; a write failure must never break the run


# -----------------------------------------------------------------------------
# Tolerant JSON parser (handles code fences, surrounding prose, bad escapes,
# trailing commas, smart quotes, and truncated/streamed output).
# -----------------------------------------------------------------------------
# Valid JSON string escapes are \\ \" \/ \b \f \n \r \t \uXXXX. Anything else
# (e.g. \( \$ \% from LaTeX) is illegal. LLMs sometimes emit those.
_INVALID_ESC_RE = re.compile(r'\\([^\\"/bfnrtu])')


def _normalise_quotes(t: str) -> str:
    """Replace smart/curly quotes the LLM occasionally emits with ASCII."""
    return (
        t.replace("“", '"').replace("”", '"')
         .replace("‘", "'").replace("’", "'")
    )


def _repair_truncated(t: str) -> str:
    """Close unterminated strings/objects/arrays so json.loads parses the prefix."""
    s = t.rstrip()
    in_string = False
    escape = False
    stack: List[str] = []
    for ch in s:
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            stack.append("}")
        elif ch == "[":
            stack.append("]")
        elif ch in "]}" and stack and stack[-1] == ch:
            stack.pop()
    suffix = ""
    if in_string:
        suffix += '"'
    end = s
    while end.endswith(","):
        end = end[:-1]
    return end + suffix + "".join(reversed(stack))


def parse_json(text: str) -> Dict[str, Any]:
    """Tolerant JSON parser. Tries several common LLM-mistake fixes before giving up."""
    s = _normalise_quotes(text.strip())
    fenced = re.match(r"^```(?:json)?\s*\n(.*?)\n```\s*$", s, re.DOTALL)
    if fenced:
        s = fenced.group(1)
    first = s.find("{")
    last = s.rfind("}")
    if first != -1 and last > first:
        s = s[first : last + 1]
    elif first != -1:
        # No closing brace yet (streaming mid-output) — keep from first { onward.
        s = s[first:]

    def _fix_escapes(t: str) -> str:
        # Double up invalid backslash escapes (\(, \$, \%, ...) so JSON parses
        # AND the backslash survives in the value (needed for LaTeX).
        return _INVALID_ESC_RE.sub(r"\\\\\1", t)

    def _strip_trailing_commas(t: str) -> str:
        return re.sub(r",(\s*[\]}])", r"\1", t)

    candidates = [
        s,
        _fix_escapes(s),
        _strip_trailing_commas(s),
        _strip_trailing_commas(_fix_escapes(s)),
        _repair_truncated(s),
        _repair_truncated(_strip_trailing_commas(_fix_escapes(s))),
    ]
    last_err: Optional[Exception] = None
    for cand in candidates:
        try:
            return json.loads(cand, strict=False)
        except json.JSONDecodeError as e:
            last_err = e
    raise last_err if last_err else json.JSONDecodeError("no candidate", s, 0)
