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
PROFILE_PATH = ROOT_DIR / "data" / "dataset_profile.json"
PREPROCESSING_REPORT = ROOT_DIR / "reports" / "preprocessing_steps.md"

st.set_page_config(
    page_title="NASDAQ Financial Analysis RAG (Grok xAI)",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Financial Data Analysis System (LLM-RAG)")
st.caption("Reproduction & Benchmarking Dashboard — Wang, Ding & Zhu (2025) Applied and Computational Engineering")

# Active Provider indicator
provider_label = "Grok (xAI)" if LLMConfig.provider == "grok" else LLMConfig.provider.title()
st.info(f"⚡ **Active LLM Provider:** `{provider_label}` &nbsp;|&nbsp; **Base Model:** `{LLMConfig.base_model}` &nbsp;|&nbsp; **Enhanced Model:** `{LLMConfig.enhanced_model}` &nbsp;|&nbsp; **API Endpoint:** `{LLMConfig.base_url or 'Default'}`")

# Sidebar - Mode Selection
st.sidebar.header("Pipeline Configuration")
st.sidebar.markdown(f"**LLM Provider:** `{provider_label}`")
st.sidebar.markdown(f"**Endpoint:** `{LLMConfig.base_url or 'api.x.ai/v1'}`")
st.sidebar.divider()
selected_group = st.sidebar.selectbox(
    "Experimental Group",
    options=list(EXPERIMENTAL_GROUPS.keys()),
    format_func=lambda g: f"{g}: {EXPERIMENTAL_GROUPS[g]['llm']} (RAG={EXPERIMENTAL_GROUPS[g]['rag']})"
)

top_k_val = st.sidebar.slider("Retrieval Top-K", min_value=1, max_value=10, value=5)

# Main Navigation Tabs
tab_query, tab_results, tab_dataset, tab_reports = st.tabs([
    "🔍 Interactive Query", "📊 Evaluation Metrics", "📁 Dataset Profile", "📑 Phase 1 Reports"
])

with tab_query:
    st.subheader("Query Financial Fundamentals")
    
    default_query = "What was the Operating Income (Loss) for INTEL CORP (INTC) in 2014 Q4?"
    user_query = st.text_input("Enter a financial question:", value=default_query)
    
    col1, col2 = st.columns(2)
    
    if st.button("Run Financial Query", type="primary"):
        group_cfg = EXPERIMENTAL_GROUPS[selected_group]
        
        with st.spinner("Processing query through RAG pipeline..."):
            pipeline = RAGPipeline(
                use_rag=group_cfg["rag"],
                model_name=group_cfg["llm"]
            )
            output = pipeline.query(user_query)
        
        with col1:
            st.markdown("### LLM Answer Output")
            st.info(output["answer"])
            st.json({
                "provider": provider_label,
                "model": output["model_used"],
                "rag_enabled": output["use_rag"],
                "group_code": selected_group,
                "simulated_fallback": output.get("is_simulated", False)
            })
            
        with col2:
            st.markdown("### Retrieved Context (FAISS)")
            if output["use_rag"] and output["retrieved_docs"]:
                docs_df = pd.DataFrame(output["retrieved_docs"])
                cols_to_show = ["company", "tickers", "period", "indicator", "amount", "score"]
                st.dataframe(docs_df[[c for c in cols_to_show if c in docs_df.columns]], width="stretch")
            elif not output["use_rag"]:
                st.warning("RAG is disabled for this experimental group. No records retrieved.")
            else:
                st.error("No relevant context found in vector store.")

with tab_results:
    st.subheader("Benchmark & Reproduction Results")
    
    if RESULTS_PATH.exists():
        results_data = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        comp = results_data.get("comparison_analysis", {})
        reproduction = results_data.get("reproduction_results", {})
        fog_metrics = reproduction.get("FOG", {}).get("metrics", {})
        achieved_recall = fog_metrics.get("overall_recall_at_5_pct", 0)

        col_m1.metric("Target FOG Accuracy", f"{comp.get('target_fog_accuracy', 78.6)}%")
        col_m2.metric("Target FOG Recall", f"{comp.get('target_fog_recall', 89.2)}%")
        col_m3.metric("Target Latency Reduction", f"{comp.get('paper_target_latency_reduction_pct', 34.8)}%")
        col_m4.metric("Achieved Retrieval Recall@5", f"{achieved_recall:.1f}%" if isinstance(achieved_recall, (int, float)) else str(achieved_recall))
        
        st.divider()
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### Paper Target Metrics (Table 15 - GPT Baseline)")
            targets = results_data.get("paper_targets", {})
            df_targets = pd.DataFrame([
                {"Configuration": k, "Accuracy (%)": v[0], "Recall (%)": v[1]}
                for k, v in targets.items()
            ])
            st.dataframe(df_targets, width="stretch")
            
        with col_b:
            st.markdown(f"#### Our Reproduction Metrics ({provider_label})")
            summary_table = []
            for group, details in reproduction.items():
                metrics = details.get("metrics", {})
                acc = metrics.get("overall_accuracy_pct")
                rec = metrics.get("recall_pct", metrics.get("entity_answer_recall_pct"))
                rec5 = metrics.get("overall_recall_at_5_pct")
                lat = metrics.get("avg_latency_seconds")
                ram = metrics.get("peak_memory_mb")
                active_model = EXPERIMENTAL_GROUPS.get(group, {}).get("llm") or details.get("config", {}).get("llm", "grok-beta")
                summary_table.append({
                    "Group": group,
                    "Model": active_model,
                    "RAG": "Enabled" if details["config"]["rag"] else "Disabled",
                    "Accuracy": f"{acc:.1f}%" if isinstance(acc, (int, float)) else str(acc),
                    "Recall (%)": f"{rec:.1f}%" if isinstance(rec, (int, float)) else str(rec),
                    "Retrieval Recall@5": f"{rec5:.1f}%" if isinstance(rec5, (int, float)) else str(rec5),
                    "Avg Latency": f"{lat:.2f}s" if isinstance(lat, (int, float)) else str(lat),
                    "Peak RAM": f"{ram:.1f} MB" if isinstance(ram, (int, float)) else str(ram)
                })
            st.dataframe(pd.DataFrame(summary_table), width="stretch")
            st.caption(
                "💡 **Metrics Explanation:** "
                "**Recall (%)** evaluates dynamic factual recall (did the model answer recall the target entity, period, metric, and numeric value). "
                "**Retrieval Recall@5** evaluates FAISS vector search (retrieving the correct 10-K filing in the top 5 candidates; 0.0% when RAG is disabled)."
            )
    else:
        st.info("No evaluation results found. Execute `python main.py` to run full evaluation.")

with tab_dataset:
    st.subheader("Raw Dataset Profile")
    if PROFILE_PATH.exists():
        profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
        st.json(profile)
    else:
        st.warning("Profile file missing.")

with tab_reports:
    st.subheader("Phase 1 Documentation & Audit Reports")
    if PREPROCESSING_REPORT.exists():
        st.markdown(PREPROCESSING_REPORT.read_text(encoding="utf-8"))
    else:
        st.warning("Preprocessing report file missing. Run `preprocessing.py` to generate.")