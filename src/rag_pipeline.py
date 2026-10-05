import os
import re
import hashlib
import pathlib
from typing import Dict, List, Any
from openai import OpenAI, RateLimitError, AuthenticationError

try:
    from config import LLMConfig, VectorConfig
    from vector_db import VectorStore
except ImportError:
    from src.config import LLMConfig, VectorConfig
    from src.vector_db import VectorStore

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]

# Prompt from Paper #22 Specification (§11 / A-12)
SYSTEM_PROMPT = (
    "You are an intelligent financial data analysis assistant. "
    "Answer the user's financial query accurately based strictly on the provided context. "
    "State the numeric value clearly with currency units."
)

class RAGPipeline:
    """Orchestrates document retrieval and LLM context formatting with graceful quota fallbacks."""
    
    def __init__(self, use_rag: bool = True, model_name: str = LLMConfig.base_model):
        self.use_rag = use_rag
        self.model_name = model_name
        self.vector_store = VectorStore() if use_rag else None

        if "grok" in self.model_name.lower():
            api_key = LLMConfig.grok_api_key
            base_url = "https://api.x.ai/v1"
        elif self.model_name.startswith("qwen/") or self.model_name.startswith("openai/") or self.model_name.startswith("allam"):
            api_key = LLMConfig.grok_api_key
            base_url = "https://api.groq.com/openai/v1"
        else:
            api_key = "ollama"
            base_url = LLMConfig.base_url

        self.client = OpenAI(api_key=api_key, base_url=base_url)

        self.api_key_valid = bool(api_key and not api_key.startswith("mock_"))

    def format_retrieved_context(self, search_results: List[Dict[str, Any]], max_chars: int = 4000) -> str:
        """Formats retrieved vector results into a constrained context block (A-13 / 1024 token window)."""
        context_blocks = []
        char_count = 0
        
        for idx, res in enumerate(search_results, 1):
            unit = "USD" if res.get("unit") == "US Dollar" else res.get("unit", "USD")
            block = (
                f"Document [{idx}]:\n"
                f"Company: {res['company']}\n"
                f"Ticker: {res['tickers']}\n"
                f"Period: {res['period']}\n"
                f"Indicator: {res['indicator']}\n"
                f"Amount: {res['amount']} {unit}\n"
            )
            if char_count + len(block) > max_chars:
                break
            context_blocks.append(block)
            char_count += len(block)
            
        return "\n".join(context_blocks)



    def query(self, user_question: str) -> Dict[str, Any]:
        """Processes a query through optional retrieval and LLM response generation."""
        retrieved_docs = []
        context_text = ""
        
        if self.use_rag and self.vector_store:
            retrieved_docs = self.vector_store.search(user_question, top_k=VectorConfig.top_k)
            context_text = self.format_retrieved_context(retrieved_docs)
            
        if self.use_rag:
            sys_prompt = (
                "You are an intelligent financial data analysis assistant. "
                "Answer the user's financial query accurately based strictly on the provided context. "
                "State the numeric value clearly with currency units."
            )
        else:
            sys_prompt = (
                "You are an intelligent financial data analysis assistant. "
                "Answer the user's financial query. Even if you are unsure of the exact figure, "
                "you MUST provide your best estimated numerical guess for the financial amount based on your training data."
            )
            
        messages = [{"role": "system", "content": sys_prompt}]
        
        if self.use_rag:
            user_content = f"Context:\n{context_text}\n\nQuestion: {user_question}"
        else:
            user_content = f"Question: {user_question}"
            
        messages.append({"role": "user", "content": user_content})

        answer = ""
        used_simulation = False
        
        if self.client:
            import time
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=LLMConfig.temperature,
                        max_tokens=LLMConfig.max_tokens,
                    )
                    answer = response.choices[0].message.content.strip()
                    break
                except RateLimitError as e:
                    if attempt < max_retries - 1:
                        time.sleep(3.0 * (attempt + 1))
                        continue
                    else:
                        answer = f"OpenAI Quota Exhausted (429): {str(e)}"
                        break
                except AuthenticationError as e:
                    answer = f"OpenAI Authentication Error (401): {str(e)}"
                    break
                except Exception as e:
                    answer = f"API Error: {str(e)}"
                    break
        else:
            answer = "OpenAI API Key not configured."

        return {
            "question": user_question,
            "answer": answer,
            "use_rag": self.use_rag,
            "model_used": self.model_name,
            "retrieved_docs": retrieved_docs,
            "is_simulated": used_simulation
        }

if __name__ == "__main__":
    pipeline = RAGPipeline(use_rag=True)
    sample_query = "What was the Operating Income (Loss) for INTEL CORP (INTC) in 2014 Q4?"
    output = pipeline.query(sample_query)
    
    print("\n--- Pipeline Execution Output ---")
    print(f"Query: {output['question']}")
    print(f"RAG Enabled: {output['use_rag']}")
    print(f"Retrieved Records: {len(output['retrieved_docs'])}")
    print(f"Simulated: {output['is_simulated']}")
    safe_answer = output["answer"].encode("ascii", "replace").decode("ascii")
    print(f"Response: {safe_answer}")