"""
frontend/app.py — Streamlit UI for DocQA-Pro.

Tabs:
  📄 Upload    — Upload PDF/DOCX or index a URL
  💬 Q&A Chat  — Chat interface with source citation
  🔍 Debug     — FAISS vs BM25 vs Hybrid side-by-side
  📊 Evaluate  — RAGAS metrics dashboard
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocQA-Pro",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Dark background */
.stApp { background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%); }

/* Glassmorphism cards */
.glass-card {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 16px;
    padding: 20px;
    backdrop-filter: blur(10px);
    margin-bottom: 16px;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.glass-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(99,102,241,0.3);
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, rgba(99,102,241,0.25), rgba(139,92,246,0.15));
    border: 1px solid rgba(99,102,241,0.4);
    border-radius: 14px;
    padding: 18px;
    text-align: center;
}
.metric-value { font-size: 2.2rem; font-weight: 700; color: #a5b4fc; }
.metric-label { font-size: 0.8rem; color: rgba(255,255,255,0.6); text-transform: uppercase; letter-spacing: 1px; }

/* Chat bubbles */
.chat-user {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    border-radius: 18px 18px 4px 18px;
    padding: 12px 18px;
    margin: 8px 0 8px 20%;
    color: white;
    font-size: 0.95rem;
}
.chat-assistant {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 18px 18px 18px 4px;
    padding: 12px 18px;
    margin: 8px 20% 8px 0;
    color: rgba(255,255,255,0.9);
    font-size: 0.95rem;
}

/* Source chip */
.source-chip {
    display: inline-block;
    background: rgba(99,102,241,0.2);
    border: 1px solid rgba(99,102,241,0.4);
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.75rem;
    color: #a5b4fc;
    margin: 2px;
}

/* Status badge */
.badge-ok   { color: #34d399; font-weight: 600; }
.badge-warn { color: #fbbf24; font-weight: 600; }
.badge-err  { color: #f87171; font-weight: 600; }

/* Tabs */
[data-baseweb="tab"] { color: rgba(255,255,255,0.6) !important; }
[data-baseweb="tab"][aria-selected="true"] { color: #a5b4fc !important; border-bottom: 2px solid #a5b4fc !important; }

/* Sidebar */
[data-testid="stSidebar"] { background: rgba(15,12,41,0.95) !important; border-right: 1px solid rgba(255,255,255,0.1); }
</style>
""",
    unsafe_allow_html=True,
)

# ── Config ────────────────────────────────────────────────────────────────────
BACKEND_URL = "http://localhost:8000"


def api(method: str, path: str, **kwargs) -> Dict | None:
    """Generic API helper with error handling."""
    try:
        resp = httpx.request(method, f"{BACKEND_URL}{path}", timeout=120, **kwargs)
        resp.raise_for_status()
        return resp.json()
    except httpx.ConnectError:
        st.error("❌ Cannot reach backend. Is the FastAPI server running?")
        return None
    except Exception as exc:
        st.error(f"API error: {exc}")
        return None


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 DocQA-Pro")
    st.markdown("*Hybrid RAG · FAISS + BM25 · Flan-T5*")
    st.divider()

    health = api("GET", "/health")
    if health:
        faiss_ok = health.get("faiss_ready", False)
        bm25_ok = health.get("bm25_ready", False)
        n_sources = health.get("indexed_sources", 0)
        st.markdown(
            f"**FAISS** {'<span class=\"badge-ok\">●  Ready</span>' if faiss_ok else '<span class=\"badge-warn\">● Empty</span>'}",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"**BM25**  {'<span class=\"badge-ok\">●  Ready</span>' if bm25_ok else '<span class=\"badge-warn\">● Empty</span>'}",
            unsafe_allow_html=True,
        )
        st.markdown(f"**Sources indexed:** `{n_sources}`")
        st.markdown(f"**LLM:** `{health.get('llm_model','—')}`")
    else:
        st.warning("Backend offline")

    st.divider()
    if st.button("🗑️ Clear all indexes", use_container_width=True):
        result = api("DELETE", "/clear")
        if result:
            st.success("All indexes cleared!")
            st.rerun()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_upload, tab_chat, tab_debug, tab_eval = st.tabs(
    ["📄 Upload", "💬 Q&A Chat", "🔍 Retrieval Debug", "📊 Evaluation"]
)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Upload
# ══════════════════════════════════════════════════════════════════════════════
with tab_upload:
    st.markdown("## 📄 Document Ingestion")
    st.markdown("Upload **PDF** or **DOCX** files, or paste a web URL to index.")

    col_file, col_url = st.columns(2, gap="large")

    with col_file:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 📂 Upload File")
        uploaded = st.file_uploader(
            "Choose a PDF or DOCX file",
            type=["pdf", "docx", "doc"],
            label_visibility="collapsed",
        )
        if uploaded and st.button("⬆️ Index File", use_container_width=True, key="btn_upload"):
            with st.spinner(f"Processing *{uploaded.name}*…"):
                result = api(
                    "POST",
                    "/ingest/file",
                    files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                )
            if result:
                st.success(f"✅ Indexed **{result['chunks']}** chunks from `{uploaded.name}`")
                stats = result.get("stats", {})
                c1, c2, c3 = st.columns(3)
                c1.metric("Chunks", stats.get("count", "—"))
                c2.metric("Avg length", stats.get("avg_length", "—"))
                c3.metric("Total chars", stats.get("total_chars", "—"))
        st.markdown("</div>", unsafe_allow_html=True)

    with col_url:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🌐 Index URL")
        url_input = st.text_input("Web page URL", placeholder="https://example.com/article")
        if url_input and st.button("⬆️ Index URL", use_container_width=True, key="btn_url"):
            with st.spinner(f"Fetching `{url_input}`…"):
                result = api("POST", "/ingest/url", json={"url": url_input})
            if result:
                st.success(f"✅ Indexed **{result['chunks']}** chunks from URL")
        st.markdown("</div>", unsafe_allow_html=True)

    # Indexed sources table
    st.divider()
    st.markdown("### 📚 Indexed Sources")
    sources_data = api("GET", "/sources")
    if sources_data and sources_data.get("sources"):
        rows = []
        for s in sources_data["sources"]:
            rows.append({
                "Name": s.get("original_name", "—"),
                "Type": s.get("type", "—").upper(),
                "Chunks": s.get("chunks", 0),
                "Avg Length": s.get("stats", {}).get("avg_length", "—"),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("No documents indexed yet. Upload a file or add a URL above.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Q&A Chat
# ══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    st.markdown("## 💬 Q&A Chat")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Render history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-user">🙋 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-assistant">🤖 {msg["content"]}</div>', unsafe_allow_html=True)
            if msg.get("sources"):
                with st.expander("📎 Source chunks", expanded=False):
                    for i, src in enumerate(msg["sources"], 1):
                        st.markdown(
                            f'<span class="source-chip">📄 {src["source"]} '
                            f'{"p." + str(src["page"]) if src.get("page") else ""}</span>',
                            unsafe_allow_html=True,
                        )
                        st.markdown(f"> {src['content'][:250]}…")

    # Input
    col_q, col_btn = st.columns([5, 1])
    with col_q:
        question = st.text_input(
            "Ask a question", placeholder="What is the main topic of the document?",
            label_visibility="collapsed", key="chat_input"
        )
    with col_btn:
        ask = st.button("Ask ➤", use_container_width=True, key="btn_ask")

    if ask and question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.spinner("Thinking…"):
            result = api("POST", "/query", json={"question": question})
        if result:
            answer = result.get("answer", "No answer returned.")
            sources = result.get("sources", [])
            st.session_state.chat_history.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )
        st.rerun()

    if st.button("🗑️ Clear chat", key="btn_clear_chat"):
        st.session_state.chat_history = []
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Retrieval Debug
# ══════════════════════════════════════════════════════════════════════════════
with tab_debug:
    st.markdown("## 🔍 Retrieval Debug Panel")
    st.markdown("Compare FAISS (dense), BM25 (sparse), and Hybrid (RRF) results side-by-side.")

    debug_query = st.text_input(
        "Debug query",
        placeholder="Enter any question to inspect retrieval…",
        key="debug_query",
    )
    debug_k = st.slider("Top-K", 1, 10, 5, key="debug_k")

    if debug_query and st.button("🔎 Compare Retrievers", key="btn_compare"):
        with st.spinner("Running all retrievers…"):
            result = api(
                "POST",
                "/query",
                json={"question": debug_query, "k": debug_k, "include_comparison": True},
            )

        if result and "retrieval_comparison" in result:
            cmp = result["retrieval_comparison"]
            col_f, col_b, col_h = st.columns(3)

            for col, key, title, color in [
                (col_f, "faiss", "🔵 FAISS (Dense)", "#3b82f6"),
                (col_b, "bm25", "🟡 BM25 (Sparse)", "#f59e0b"),
                (col_h, "hybrid", "🟢 Hybrid (RRF)", "#10b981"),
            ]:
                with col:
                    st.markdown(f"**{title}**")
                    items = cmp.get(key, [])
                    for i, item in enumerate(items, 1):
                        st.markdown(
                            f"""<div class="glass-card">
                            <div style="color:{color};font-weight:600;font-size:0.8rem">
                                #{i} · score {item['score']:.4f}
                            </div>
                            <div style="color:rgba(255,255,255,0.5);font-size:0.75rem">
                                {item['source']}
                            </div>
                            <div style="margin-top:6px;font-size:0.88rem">
                                {item['content'][:220]}…
                            </div>
                            </div>""",
                            unsafe_allow_html=True,
                        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Evaluation
# ══════════════════════════════════════════════════════════════════════════════
with tab_eval:
    st.markdown("## 📊 RAGAS Evaluation Dashboard")

    eval_col, hist_col = st.columns([2, 3], gap="large")

    with eval_col:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### ▶️ Run Evaluation")
        st.markdown(
            "Provide Q&A pairs and the system will retrieve context, generate answers, "
            "then score with RAGAS."
        )

        default_pairs = json.dumps(
            [
                {"question": "What is retrieval-augmented generation?",
                 "ground_truth": "RAG combines retrieval with generation to answer questions from documents."},
                {"question": "What is BM25?",
                 "ground_truth": "BM25 is a probabilistic sparse retrieval algorithm based on term frequency."},
            ],
            indent=2,
        )
        qa_json = st.text_area(
            "Q&A pairs (JSON)", value=default_pairs, height=220, key="eval_qa"
        )
        run_name = st.text_input("Run name (optional)", placeholder="my-eval-run", key="eval_name")

        if st.button("🚀 Run RAGAS Evaluation", use_container_width=True, key="btn_eval"):
            try:
                pairs = json.loads(qa_json)
            except json.JSONDecodeError as exc:
                st.error(f"Invalid JSON: {exc}")
                pairs = None

            if pairs:
                with st.spinner("Running evaluation (this may take a few minutes)…"):
                    result = api(
                        "POST",
                        "/evaluate/build",
                        json={"qa_pairs": pairs, "run_name": run_name or None},
                    )
                if result and "metrics" in result:
                    st.success("✅ Evaluation complete!")
                    metrics = result["metrics"]
                    for metric, score in metrics.items():
                        st.markdown(
                            f'<div class="metric-card" style="margin-bottom:10px">'
                            f'<div class="metric-value">{score:.3f}</div>'
                            f'<div class="metric-label">{metric.replace("_", " ").title()}</div>'
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                elif result and "error" in result:
                    st.error(f"Evaluation error: {result['error']}")
        st.markdown("</div>", unsafe_allow_html=True)

    with hist_col:
        st.markdown("### 📈 Score History")
        history_data = api("GET", "/evaluate/history")

        if history_data and history_data.get("history"):
            history = history_data["history"]
            df_rows = []
            for run in history:
                row = {"Run": run.get("run_name", "—"), "Samples": run.get("num_samples", 0)}
                row.update(run.get("metrics", {}))
                df_rows.append(row)
            df = pd.DataFrame(df_rows)
            st.dataframe(df, use_container_width=True, hide_index=True)

            # Plot score history
            metric_cols = [c for c in df.columns if c not in ["Run", "Samples"]]
            if metric_cols:
                fig = go.Figure()
                colors = ["#6366f1", "#8b5cf6", "#10b981", "#f59e0b"]
                for i, metric in enumerate(metric_cols):
                    fig.add_trace(
                        go.Scatter(
                            x=df["Run"],
                            y=df[metric],
                            mode="lines+markers",
                            name=metric.replace("_", " ").title(),
                            line=dict(color=colors[i % len(colors)], width=2),
                            marker=dict(size=8),
                        )
                    )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="rgba(255,255,255,0.8)", family="Inter"),
                    legend=dict(orientation="h", y=-0.2),
                    yaxis=dict(range=[0, 1], gridcolor="rgba(255,255,255,0.1)"),
                    xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                    margin=dict(l=20, r=20, t=20, b=20),
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No evaluation history yet. Run your first evaluation on the left.")
