# ── Fix: Streamlit on Windows replaces sys.stderr with an object whose
# flush() raises OSError.  tqdm (called by transformers during weight
# loading) always calls sys.stderr.flush(), crashing the app.
# Patch it BEFORE anything imports tqdm / transformers.
import sys

_original_stderr_flush = getattr(sys.stderr, "flush", None)

def _safe_flush():
    try:
        if _original_stderr_flush:
            _original_stderr_flush()
    except OSError:
        pass

sys.stderr.flush = _safe_flush
# ── End fix ──────────────────────────────────────────────────────────────────

import streamlit as st
import pandas as pd
import json
import pathlib

try:
    from rag_pipeline import RAGPipeline
    from vector_db import VectorStore
    from config import EXPERIMENTAL_GROUPS, PAPER_RESULTS, PAPER_BENCHMARKS, LLMConfig
except ImportError:
    from src.rag_pipeline import RAGPipeline
    from src.vector_db import VectorStore
    from src.config import EXPERIMENTAL_GROUPS, PAPER_RESULTS, PAPER_BENCHMARKS, LLMConfig

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT_DIR / "data" / "processed" / "results.json"

st.set_page_config(
    page_title="Financial Analysis RAG — Ollama & Groq",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Financial Data Analysis System (LLM-RAG)")
st.caption("Reproduction & Benchmarking Dashboard — Wang, Ding & Zhu (2025) Applied and Computational Engineering")

# ── Provider banner ──────────────────────────────────────────────────────────
provider_label = "Ollama & Groq (Qwen)"
st.info(
    f"⚡ **Active LLM Provider:** `{provider_label}` &nbsp;|&nbsp; "
    f"**Base Model:** `llama3.2` &nbsp;|&nbsp; "
    f"**Enhanced Model:** `qwen/qwen3.8-27b` &nbsp;|&nbsp; "
    f"**API Endpoints:** `http://localhost:11434/v1` & `api.groq.com/openai/v1`"
)

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.header("Pipeline Configuration")
st.sidebar.markdown(f"**LLM Provider:** `{provider_label}`")
st.sidebar.markdown("**Ollama Endpoint:** `localhost:11434`")
st.sidebar.markdown("**Groq Endpoint:** `api.groq.com`")
st.sidebar.divider()

# Hardcoded so Streamlit never shows stale cached values
SIDEBAR_GROUPS = {
    "BG":  {"label": "BG: llama3.2 (RAG=False)",          "llm": "llama3.2",         "rag": False},
    "REG": {"label": "REG: llama3.2 (RAG=True)",          "llm": "llama3.2",         "rag": True},
    "VUG": {"label": "VUG: qwen/qwen3.8-27b (RAG=False)", "llm": "qwen/qwen3.8-27b", "rag": False},
    "FOG": {"label": "FOG: qwen/qwen3.8-27b (RAG=True)",  "llm": "qwen/qwen3.8-27b", "rag": True},
}

selected_group = st.sidebar.selectbox(
    "Experimental Group",
    options=list(SIDEBAR_GROUPS.keys()),
    format_func=lambda g: SIDEBAR_GROUPS[g]["label"]
)

top_k_val = st.sidebar.slider("Retrieval Top-K", min_value=1, max_value=10, value=5)

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_query, tab_results = st.tabs([
    "🔍 Interactive Query", "📊 Evaluation Metrics"
])

with tab_query:
    st.subheader("Query Financial Fundamentals")

    default_query = "What was the Operating Income (Loss) for INTEL CORP (INTC) in 2014 Q4?"
    user_query = st.text_input("Enter a financial question:", value=default_query)

    col1, col2 = st.columns(2)

    if st.button("Run Financial Query", type="primary"):
        grp = SIDEBAR_GROUPS[selected_group]

        with st.spinner("Processing query through RAG pipeline..."):
            pipeline = RAGPipeline(
                use_rag=grp["rag"],
                model_name=grp["llm"]
            )
            output = pipeline.query(user_query)

        with col1:
            st.markdown("### LLM Answer Output")
            st.info(output["answer"])
            st.json({
                "provider": provider_label,
                "model": output["model_used"],
                "rag_enabled": output["use_rag"],
                "group_code": selected_group
            })

        with col2:
            st.markdown("### Retrieved Context (FAISS)")
            if output["use_rag"] and output["retrieved_docs"]:
                docs_df = pd.DataFrame(output["retrieved_docs"])
                cols_to_show = ["company", "tickers", "period", "indicator", "amount", "score"]
                st.dataframe(docs_df[[c for c in cols_to_show if c in docs_df.columns]], use_container_width=True)
            elif not output["use_rag"]:
                st.warning("RAG is disabled for this experimental group. No records retrieved.")
            else:
                st.error("No relevant context found in vector store.")

with tab_results:
    st.subheader("Benchmark & Reproduction Results")

    if RESULTS_PATH.exists():
        results_data = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))

        # ── Top-level KPIs ───────────────────────────────────────────────
        reproduction = results_data.get("reproduction_results", {})
        fog_metrics = reproduction.get("FOG", {}).get("metrics", {})

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Target FOG Accuracy", "78.6%")
        col_m2.metric("Target FOG Recall", "89.2%")
        col_m3.metric("Target Latency Reduction", "34.8%")
        col_m4.metric("Achieved Retrieval Recall@5", f"{fog_metrics.get('overall_recall_at_5_pct', 0):.1f}%")

        st.divider()

        # ── Two-column comparison ────────────────────────────────────────
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### Paper Target Metrics (Table 15 — GPT Baseline)")
            targets = results_data.get("paper_targets", {})
            df_targets = pd.DataFrame([
                {"Configuration": k, "Accuracy (%)": v[0], "Recall (%)": v[1]}
                for k, v in targets.items()
            ])
            st.dataframe(df_targets, use_container_width=True)

        with col_b:
            st.markdown(f"#### Our Reproduction Metrics ({provider_label})")
            rows = []
            for group in ["BG", "REG", "VUG", "FOG"]:
                if group not in reproduction:
                    continue
                details = reproduction[group]
                metrics = details.get("metrics", {})
                # Read model from results.json config, guaranteed correct
                model = details.get("config", {}).get("llm", "unknown")
                rag = "Enabled" if details["config"]["rag"] else "Disabled"
                acc = metrics.get("overall_accuracy_pct")
                rec = metrics.get("recall_pct", metrics.get("entity_answer_recall_pct"))
                ent = metrics.get("entity_answer_recall_pct")
                rec5 = metrics.get("overall_recall_at_5_pct")
                lat = metrics.get("avg_latency_seconds")
                ram = metrics.get("peak_memory_mb")
                rows.append({
                    "Group": group,
                    "Model": model,
                    "RAG": rag,
                    "Accuracy": f"{acc:.1f}%" if isinstance(acc, (int, float)) else str(acc),
                    "Recall (%)": f"{rec:.1f}%" if isinstance(rec, (int, float)) else str(rec),
                    "Retrieval Recall@5": f"{rec5:.1f}%" if isinstance(rec5, (int, float)) else str(rec5),
                    "Avg Latency": f"{lat:.2f}s" if isinstance(lat, (int, float)) else str(lat),
                    "Peak RAM": f"{ram:.1f} MB" if isinstance(ram, (int, float)) else str(ram),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
            st.caption(
                "💡 **Metrics Explanation:** "
                "**Recall (%)** evaluates dynamic factual recall (did the model answer recall the target entity, period, metric, and numeric value). "
                "**Retrieval Recall@5** evaluates FAISS vector search (retrieving the correct 10-K filing in the top 5 candidates; 0.0% when RAG is disabled)."
            )
    else:
        st.info("No evaluation results found. Execute `python main.py` to run full evaluation.")