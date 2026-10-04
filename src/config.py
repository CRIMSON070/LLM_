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


def _normalize_provider(value: str | None) -> str:
    if not value:
        return ""
    return {
        "grok": "grok",
        "xai": "grok",
        "groq": "groq",
        "openai": "openai",
        "gpt": "openai",
    }.get(value.strip().lower(), value.strip().lower())


def _detect_provider() -> str:
    configured = _normalize_provider(os.getenv("LLM_PROVIDER"))
    if configured in {"grok", "groq", "openai"}:
        return configured
    if os.getenv("GROK_API_KEY") or os.getenv("XAI_API_KEY"):
        return "grok"
    if os.getenv("GROQ_API_KEY") or (os.getenv("OPENAI_API_KEY", "").startswith("gsk_")):
        return "groq"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    return "openai"


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
    """Language model parameters supporting Groq, Grok (xAI), and OpenAI."""
    # Provider detection
    provider: str = _detect_provider()

    # Model selections
    base_model: str = os.getenv(
        "LLM_BASE_MODEL",
        "openai/gpt-oss-20b" if provider == "groq" else ("grok-beta" if provider == "grok" else "gpt-3.5-turbo")
    )
    enhanced_model: str = os.getenv(
        "LLM_ENHANCED_MODEL",
        "openai/gpt-oss-120b" if provider == "groq" else ("grok-2-latest" if provider == "grok" else "gpt-3.5-turbo-0125")
    )

    # API endpoints
    base_url: str = os.getenv(
        "LLM_BASE_URL",
        "https://api.x.ai/v1" if provider == "grok" else (
            "https://api.groq.com/openai/v1" if provider == "groq" else None
        )
    )
    
    temperature: float = 0.0                    # [ASSUMED] A-12
    max_tokens: int = 256                       # [ASSUMED] A-12
    # If API quota is exhausted or key is simulated, allow realistic fallback
    allow_simulation_fallback: bool = True

# ---- Experimental Groups & Baselines ----
EXPERIMENTAL_GROUPS: Dict[str, Dict[str, bool | str]] = {
    "BG":  {"llm": LLMConfig.base_model,     "rag": False},
    "REG": {"llm": LLMConfig.base_model,     "rag": True},
    "VUG": {"llm": LLMConfig.enhanced_model, "rag": False},
    "FOG": {"llm": LLMConfig.enhanced_model, "rag": True},
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