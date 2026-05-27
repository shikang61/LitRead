"""
ArXiv AI Assistant — paste an ArXiv link, chat about the paper.

Pipeline:
    URL  -> regex extract ArXiv ID
         -> ArxivLoader fetches full paper text
         -> RecursiveCharacterTextSplitter chunks the doc
         -> HuggingFace local embeddings + FAISS in-memory vector store (RAG)
         -> ChatOpenAI (OpenAI or Grok via base_url override) answers
         -> Gradio Blocks UI with streaming chatbot + per-session state
"""

import os
import re
import warnings
from typing import List, Optional

# `langchain-community` is being sunset, but ArxivLoader and FAISS have no
# standalone replacement yet — silence the deprecation noise on import.
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain_community.*")
warnings.filterwarnings("ignore", message=".*langchain-community.*sunset.*")

import gradio as gr
from langchain_community.document_loaders import ArxivLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# -----------------------------------------------------------------------------
# Config — edit model IDs here if provider names change.
# -----------------------------------------------------------------------------
OPENAI_MODEL = "gpt-5"
GROK_MODEL = "grok-4"
GROK_BASE_URL = "https://api.x.ai/v1"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
RETRIEVER_K = 6  # top-k chunks per query

PROVIDERS = {
    "OpenAI (GPT-5)": {
        "model": OPENAI_MODEL,
        "base_url": None,
        "env_key": "OPENAI_API_KEY",
    },
    "Grok (Grok-4)": {
        "model": GROK_MODEL,
        "base_url": GROK_BASE_URL,
        "env_key": "XAI_API_KEY",
    },
}

# -----------------------------------------------------------------------------
# System prompt — captivating "carousel" science communicator.
# -----------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a captivating science communicator answering questions about a research paper.

You will receive retrieved excerpts from the paper as `context` and a user question.

RULES:
- Answer ONLY from the provided context. If the context does not contain the answer, say so plainly.
- Use simple, layman-friendly language. No heavy jargon. If a technical term is unavoidable, briefly define it.
- Be engaging. Use a "hook" that makes the reader want to dive into the paper itself.

FORMATTING — "bite-sized carousel":
When the user asks for a summary, overview, or open-ended explanation, structure the response as a
visually separated Markdown carousel. Cover these aspects (skip any not present in the context):
    🎯  Motivation
    🌍  Context of the Problem
    💡  Proposed Solution
    ⚙️  How It Works
    📊  Comparison to Existing Methods
    🔮  Future Work

Template per slide:
---
### 🎠 Slide N: <emoji> <Snappy Title>
<2–4 sentences in layman terms, captivating tone>
---

For narrow follow-up questions (e.g. "what was the learning rate?"), respond conversationally in
1–3 sentences — no carousel needed.

CONTEXT:
{context}
"""

CONTEXTUALIZE_PROMPT = """Given the chat history and the latest user question, rewrite the question
so it can be understood on its own without the chat history. If the question is already standalone,
return it unchanged. Do NOT answer the question."""


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
# Matches new-style IDs (2305.10601, 2305.10601v2) and old-style (cs.LG/0701001).
ARXIV_ID_RE = re.compile(
    r"(?:arxiv\.org/(?:abs|pdf|html)/)?"
    r"(?P<id>\d{4}\.\d{4,5}(?:v\d+)?|[a-z\-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?)",
    re.IGNORECASE,
)


def extract_arxiv_id(text: str) -> Optional[str]:
    """Pull an ArXiv ID out of a URL or bare string. Returns None if not found."""
    if not text:
        return None
    text = text.strip().rstrip("/")
    if text.lower().endswith(".pdf"):
        text = text[:-4]
    m = ARXIV_ID_RE.search(text)
    return m.group("id") if m else None


def build_vectorstore(docs: List[Document]) -> FAISS:
    """Chunk docs and index them with FAISS + local HF embeddings."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return FAISS.from_documents(chunks, embeddings)


def resolve_api_key(ui_key: str, env_var: str) -> str:
    """UI input wins; fall back to env var. Empty string if neither set."""
    return (ui_key or "").strip() or os.getenv(env_var, "").strip()


def make_llm(provider: str, api_key: str, streaming: bool = True) -> ChatOpenAI:
    cfg = PROVIDERS[provider]
    kwargs = {"model": cfg["model"], "api_key": api_key, "streaming": streaming}
    if cfg["base_url"]:
        kwargs["base_url"] = cfg["base_url"]
    return ChatOpenAI(**kwargs)


def build_chain(llm: ChatOpenAI, vs: FAISS):
    """LCEL retrieval chain with history-aware question rewriting.

    Each call:
      1. `history_aware_retriever` rewrites the user question using prior chat
         turns, then pulls top-k chunks from FAISS.
      2. `qa_chain` stuffs those chunks into the system prompt and streams the
         answer.
    """
    retriever = vs.as_retriever(search_kwargs={"k": RETRIEVER_K})

    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", CONTEXTUALIZE_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_prompt
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)

    return create_retrieval_chain(history_aware_retriever, qa_chain)


def history_to_lc_messages(history: List[dict]) -> List:
    """Gradio `type="messages"` list -> LangChain message objects."""
    msgs = []
    for turn in history:
        role = turn.get("role")
        content = turn.get("content", "")
        if not content:
            continue
        if role == "user":
            msgs.append(HumanMessage(content=content))
        elif role == "assistant":
            msgs.append(AIMessage(content=content))
    return msgs


# -----------------------------------------------------------------------------
# Gradio callbacks
# -----------------------------------------------------------------------------
def load_paper(url: str, provider: str, api_key: str):
    """Fetch paper + build chain. Returns (session_state, status_markdown, cleared_chat)."""
    aid = extract_arxiv_id(url)
    if not aid:
        return None, "❌ **Invalid URL.** Could not find an ArXiv ID in your input.", []

    key = resolve_api_key(api_key, PROVIDERS[provider]["env_key"])
    if not key:
        env = PROVIDERS[provider]["env_key"]
        return (
            None,
            f"❌ **Missing API key** for {provider}. Paste it in the sidebar or set `{env}` in your environment.",
            [],
        )

    try:
        docs = ArxivLoader(query=aid, load_max_docs=1).load()
    except Exception as e:
        return None, f"❌ **Failed to fetch paper** `{aid}`: `{type(e).__name__}: {e}`", []

    if not docs:
        return None, f"❌ **Paper `{aid}` returned no content.** Check the ID.", []

    meta = docs[0].metadata or {}
    title = meta.get("Title") or meta.get("title") or "Unknown"
    authors = meta.get("Authors") or meta.get("authors") or "Unknown"
    published = meta.get("Published") or meta.get("published") or ""

    try:
        vs = build_vectorstore(docs)
    except Exception as e:
        return None, f"❌ **Indexing failed**: `{type(e).__name__}: {e}`", []

    try:
        chain = build_chain(make_llm(provider, key, streaming=True), vs)
    except Exception as e:
        return None, f"❌ **LLM init failed**: `{type(e).__name__}: {e}`", []

    state = {"chain": chain, "arxiv_id": aid, "title": title}
    status = (
        f"✅ **Loaded `{aid}`**\n\n"
        f"**Title:** {title}\n\n"
        f"**Authors:** {authors}\n\n"
        + (f"**Published:** {published}\n\n" if published else "")
        + "_Ready to chat. Try: **'Give me a carousel summary'**._"
    )
    return state, status, []


def chat_respond(message: str, history: List[dict], state: Optional[dict]):
    """Generator: appends user turn + streaming assistant turn to `history`.

    Yields (cleared_input, updated_history) tuples. Gradio's `.then` chain
    pattern: first call clears the textbox, subsequent yields stream tokens
    into the assistant bubble.
    """
    if not message or not message.strip():
        yield "", history
        return

    history = (history or []) + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": ""},
    ]

    if not state or "chain" not in state:
        history[-1]["content"] = "⚠️ Load a paper first using the sidebar."
        yield "", history
        return

    lc_history = history_to_lc_messages(history[:-2])  # exclude current user msg + empty assistant

    try:
        partial = ""
        for chunk in state["chain"].stream({"input": message, "chat_history": lc_history}):
            # create_retrieval_chain yields dicts; the LLM answer arrives under "answer".
            token = chunk.get("answer")
            if token:
                partial += token
                history[-1]["content"] = partial
                yield "", history
        if not partial:
            history[-1]["content"] = "_(Empty response from model.)_"
            yield "", history
    except Exception as e:
        history[-1]["content"] = f"⚠️ **Generation error:** `{type(e).__name__}: {e}`"
        yield "", history


def clear_chat():
    """Wipe the chat panel but keep the loaded paper / chain alive in session state."""
    return []


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
def build_ui() -> gr.Blocks:
    with gr.Blocks(title="ArXiv AI Assistant") as demo:
        gr.Markdown(
            "# 📚 ArXiv AI Assistant\n"
            "Paste an ArXiv link, load the paper, then chat about it. "
            "Answers are grounded *only* in the paper text via RAG."
        )

        # Per-session state — holds the RAG chain so each user has their own paper.
        session = gr.State(value=None)

        with gr.Row():
            with gr.Column(scale=1, min_width=320):
                gr.Markdown("### ⚙️ Configuration")
                provider = gr.Dropdown(
                    choices=list(PROVIDERS.keys()),
                    value="OpenAI (GPT-5)",
                    label="LLM Provider",
                )
                api_key = gr.Textbox(
                    label="API Key",
                    type="password",
                    placeholder="sk-... (blank = use env var)",
                )
                url = gr.Textbox(
                    label="ArXiv URL or ID",
                    placeholder="https://arxiv.org/abs/2305.10601",
                )
                load_btn = gr.Button("📥 Load Paper", variant="primary")
                status = gr.Markdown("_No paper loaded yet._")

            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    height=560,
                    show_label=False,
                    avatar_images=(None, None),
                )
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Ask about the paper… (e.g. 'Give me a carousel summary')",
                        show_label=False,
                        scale=8,
                        container=False,
                    )
                    send = gr.Button("Send", variant="primary", scale=1, min_width=80)
                clear = gr.Button("🗑 Clear chat", size="sm")

        load_btn.click(
            load_paper,
            inputs=[url, provider, api_key],
            outputs=[session, status, chatbot],
        )

        # Both Enter-in-textbox and Send-click trigger streaming chat_respond.
        # chat_respond is a generator yielding (cleared_input, updated_history).
        for trigger in (msg.submit, send.click):
            trigger(
                chat_respond,
                inputs=[msg, chatbot, session],
                outputs=[msg, chatbot],
            )

        clear.click(clear_chat, outputs=[chatbot])

    return demo


if __name__ == "__main__":
    build_ui().launch(theme=gr.themes.Soft())
