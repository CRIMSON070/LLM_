"""Pipeline parameters and provenance tracking for Paper #22 reproduction."""

import os
import pathlib
from dataclasses import dataclass
from typing import Dict, Tuple

# Automatically load .env if present
ENV_PATH = pathlib.Path(__file__).resolve().parents[1] / ".env"
if ENV_PATH.exists():
    for line in ENV_PATH.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())



@dataclass
class VectorConfig:
    """Vectorisation, storage, and retrieval parameters."""
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2" # [PAPER] Table 4
    vector_dim: int = 384                                           # [PAPER] Table 4
    max_seq_length: int = 512                                       # [PAPER] Table 4
    index_type: str = "IndexFlatIP"                                 # [PAPER] Table 5
    top_k: int = 5                                                  # [PAPER] Table 4
    context_window: int = 1024                                      # [PAPER] Table 4 (A-13)
    normalize_embeddings: bool = True                               # [ASSUMED] A-03
    chunking_strategy: str = "one record = one document"            # [ASSUMED] A-02

@dataclass
class LLMConfig:
    """Language model parameters supporting Ollama and Grok."""
    provider: str = os.getenv("LLM_PROVIDER", "mixed")
    base_model: str = os.getenv("LLM_BASE_MODEL", "llama3.2")
    enhanced_model: str = os.getenv("LLM_ENHANCED_MODEL", "grok-beta")
    base_url: str = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
    grok_api_key: str = os.getenv("XAI_API_KEY", "")
    
    temperature: float = 0.0                    # [ASSUMED] A-12
    max_tokens: int = 256                       # [ASSUMED] A-12

# ---- Experimental Groups & Baselines ----
EXPERIMENTAL_GROUPS: Dict[str, Dict[str, bool | str]] = {
    "BG":  {"llm": "llama3.2",     "rag": False},
    "REG": {"llm": "llama3.2",     "rag": True},
    "VUG": {"llm": "qwen/qwen3.8-27b", "rag": False},
    "FOG": {"llm": "qwen/qwen3.8-27b", "rag": True},
}

# Target metrics from the original paper (accuracy %, recall %)
PAPER_RESULTS: Dict[str, Tuple[float, float]] = {
    "gpt-3.5-turbo": (55.6, 78.3),
    "gpt-3.5-turbo+RAG": (63.7, 88.7),
    "gpt-3.5-turbo-1106": (59.3, 85.4),
    "gpt-3.5-turbo-1106+RAG": (78.6, 89.2),
    "gpt-4.0-mini": (68.2, 86.9),
}

# Target hardware & performance benchmarks
PAPER_BENCHMARKS = {
    "response_time_reduction_pct": 34.8,
    "target_accuracy_pct": 78.6,
    "target_recall_pct": 89.2,
    "hardware_reported": {
        "os": "Ubuntu 22.04 LTS",
        "gpu": "NVIDIA A800 80GB HBM2e",
        "cpu": "AMD EPYC 7763 64-Core",
        "ram": "256 GB DDR4",
    }
}