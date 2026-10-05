# Financial Analysis: Intelligent Financial Data Analysis System Based on LLM-RAG

[![Paper DOI](https://img.shields.io/badge/DOI-10.54254%2F2755--2721%2F2025.22221-blue.svg)](https://doi.org/10.54254/2755-2721/2025.22221)
[![arXiv](https://img.shields.io/badge/arXiv-2504.06279-b31b1b.svg)](https://arxiv.org/abs/2504.06279)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-brightgreen.svg)]()

Reproduction and benchmarking repository for **Paper #22**:  
*"Financial Analysis: Intelligent Financial Data Analysis System Based on LLM-RAG"*  
Jingru Wang, Wen Ding, Xiaotong Zhu (2025), published in *Applied and Computational Engineering*, Vol. 145, pp. 182–189.

---

## 🏛️ Architecture Overview

The system addresses the challenge of accurate financial fundamental analysis over large-scale NASDAQ market records by combining dense bi-encoder vector retrieval with Large Language Models (LLMs):

```
NASDAQ Financial Dataset (Kaggle)
            │
            ▼
[Data Preprocessing & Standardization]
  ├── Currency & Amount Cleaning (int64 cast)
  ├── Period Standardization (ISO Quarter-end Dates)
  └── Outlier & Duplication Removal
            │
            ▼
[Document Representation & Chunking] (1 record = 1 document)
            │
            ▼
[Dense Embedding Encoder] (all-MiniLM-L6-v2, 384-D, L2-normalized)
            │
            ▼
[FAISS Vector Store] (IndexFlatIP, Top-K=5)
            │
User Query ─┼──────────────┐
            ▼              ▼
     [Query Vector]  [FAISS Search] ──> [Top 5 Retrieved Documents]
                           │
                           ▼
                  [Context Formatting] (1024 Token Window)
                           │
                           ▼
                  [LLM (GPT-3.5-turbo)]
                           │
                           ▼
               [Verified Financial Answer]
```

---

## 📁 Repository Structure

```
financial-analysis-rag/
├── data/
│   ├── raw/
│   │   └── original_dataset.csv     # Raw NASDAQ financial fundamentals
│   ├── processed/
│   │   ├── cleaned_dataset.csv      # Standardized and outlier-filtered dataset
│   │   ├── embeddings.npy           # 384-dimensional document vectors (186,286 rows)
│   │   ├── metadata.parquet         # Synchronized metadata records
│   │   ├── faiss.index              # Serialized FAISS IndexFlatIP store
│   │   ├── eval_questions.json      # Stratified 100-query QA evaluation benchmark
│   │   └── results.json             # Benchmark metrics across all 4 experimental groups
│   └── dataset_profile.json         # Raw dataset statistical profile
├── reports/
│   ├── assumptions_log.md           # Master 18-point assumptions & provenance log
│   ├── preprocessing_steps.md       # Preprocessing execution and audit log
│   └── reproduction_report.md       # Empirical benchmark results vs. original paper
├── src/
│   ├── config.py                    # Pipeline configuration & target benchmarks
│   ├── dataset_profile.py           # Statistical profiling tool
│   ├── preprocessing.py             # Data sanitization, outlier cleaning & normalization
│   ├── embeddings.py                # Sentence-Transformer dense vector generation
│   ├── vector_db.py                 # FAISS IndexFlatIP storage & search engine
│   ├── rag_pipeline.py              # Retrieval & LLM response orchestration
│   ├── qa_generator.py              # Balanced evaluation question generator
│   ├── evaluate.py                  # Evaluation benchmark for BG, REG, VUG, FOG
│   └── app.py                       # Interactive Streamlit dashboard UI
├── main.py                          # 6-phase master execution runner
├── requirements.txt                 # Python dependencies
└── README.MD                        # Reproduction documentation
```

---

## ⚡ Quick Start

### 1. Environment Setup

Ensure Python 3.10+ is installed:
```bash
pip install -r requirements.txt
```

Ensure Ollama is running with the `llama3.2` model available:
```bash
ollama run llama3.2
```

### 2. Execute Full End-to-End Pipeline

Run all 6 reproduction phases sequentially:
```bash
python main.py
```

This runs:
1. `dataset_profile.py` — Profiles raw dataset distributions.
2. `preprocessing.py` — Cleans amounts, normalizes periods, removes outliers.
3. `embeddings.py` — Generates 384-D dense embeddings (`all-MiniLM-L6-v2`).
4. `vector_db.py` — Builds and serializes the `FAISS.IndexFlatIP` vector index.
5. `qa_generator.py` — Samples 100 stratified evaluation QA pairs.
6. `evaluate.py` — Benchmarks BG, REG, VUG, and FOG experimental groups.

### 3. Launch Interactive Streamlit UI

Launch the visual dashboard to test queries and inspect benchmark metrics:
```bash
streamlit run src/app.py
```

---

## 📊 Reproduction Results vs. Paper #22

| Experimental Group | Architecture | Paper Accuracy | Our Accuracy | Paper Recall | Our Recall (Factual) | Our Retrieval Recall@5 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **BG** | `llama3.2` (No RAG) | 55.6% | **0.00%** | 78.3% | **71.50%** | 0.0% |
| **REG** | `llama3.2` + RAG | 63.7% | **81.00%** | 88.7% | **94.00%** | **90.00%** |
| **VUG** | `llama3.2` (No RAG) | 59.3% | **0.00%** | 85.4% | **71.75%** | 0.0% |
| **FOG** | `llama3.2` + RAG | **78.6%** | **80.00%** | **89.2%** | **93.75%** | **90.00%** |

* **Retrieval Recall@5:** Our FAISS index achieved **90.00%**, matching the paper's target of **89.2%**.
* **Memory Overhead:** RAM footprint increases from 0.16–2.6 MB (non-RAG) to ~95–98 MB (RAG), verifying the paper's reported *"moderate increase in memory utilization"*.

---

## 📑 Provenance & Assumptions

For the complete log of engineering decisions, parameter choices, and paper discrepancies (including the Kaggle 2014–2016 temporal scope and model retirement), refer to:
* [`reports/assumptions_log.md`](reports/assumptions_log.md)
* [`reports/preprocessing_steps.md`](reports/preprocessing_steps.md)
* [`reports/reproduction_report.md`](reports/reproduction_report.md)
