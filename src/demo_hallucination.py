from rag_pipeline import RAGPipeline
from config import LLMConfig

def main():
    question = "What was the Operating Income (Loss) for INTEL CORP (INTC) in 2014 Q4?"
    
    # 1. No RAG (Hallucination Baseline)
    bg_pipeline = RAGPipeline(use_rag=False, model_name=LLMConfig.base_model)
    bg_result = bg_pipeline.query(question)
    
    # 2. With RAG (Accurate Grounded)
    reg_pipeline = RAGPipeline(use_rag=True, model_name=LLMConfig.base_model)
    reg_result = reg_pipeline.query(question)
    
    print("--- QUESTION ---")
    print(question)
    print("\n--- NO RAG (Baseline Hallucination) ---")
    print(bg_result["answer"])
    print("\n--- WITH RAG (Accurate) ---")
    print(reg_result["answer"])

if __name__ == "__main__":
    main()
