import json
import re
import time
import tracemalloc
import pathlib
from typing import Dict, List, Any, Tuple
import pandas as pd

try:
    from config import EXPERIMENTAL_GROUPS, PAPER_RESULTS, PAPER_BENCHMARKS, LLMConfig, VectorConfig
    from rag_pipeline import RAGPipeline
except ImportError:
    from src.config import EXPERIMENTAL_GROUPS, PAPER_RESULTS, PAPER_BENCHMARKS, LLMConfig, VectorConfig
    from src.rag_pipeline import RAGPipeline

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
QA_TEST_SET_PATH = ROOT_DIR / "data" / "processed" / "eval_questions.json"
RESULTS_OUTPUT_PATH = ROOT_DIR / "data" / "processed" / "results.json"

def extract_numeric_value(text: str) -> float | None:
    """Extracts the primary monetary figure from an LLM response string (A-14)."""
    # 1. First look for amounts explicitly marked with USD or currency symbols
    usd_matches = re.findall(r"(-?\d[\d,.]*)\s*(?:USD|US Dollar|dollars)", text, flags=re.IGNORECASE)
    if usd_matches:
        val_str = re.sub(r"[,\s]", "", usd_matches[0])
        try:
            return float(val_str)
        except ValueError:
            pass

    # 2. Look for amounts with currency symbols: $1,234,567
    symbol_matches = re.findall(r"[$€£¥]\s*(-?\d[\d,.]*)", text)
    if symbol_matches:
        val_str = re.sub(r"[,\s]", "", symbol_matches[0])
        try:
            return float(val_str)
        except ValueError:
            pass

    # 3. Look for numbers while filtering out calendar years (1900-2099), HTTP codes, and rank digits
    cleaned = re.sub(r"[$€£¥,]", "", text)
    cleaned = re.sub(r"\b(?:401|429|500)\b", "", cleaned)
    tokens = re.findall(r"-?\d+(?:\.\d+)?", cleaned)
    
    candidates = []
    for t in tokens:
        try:
            val = float(t)
            # Filter out calendar years and rank digits
            if 1900 <= val <= 2099:
                continue
            if val in [1, 2, 3, 4, 5]:
                continue
            candidates.append(val)
        except ValueError:
            continue

    if candidates:
        return candidates[0]
    return None

def check_numeric_accuracy(predicted_text: str, ground_truth_amount: int, tolerance_pct: float = 0.01) -> bool:
    """Evaluates if the extracted number matches ground truth within tolerance (A-14)."""
    pred_val = extract_numeric_value(predicted_text)
    if pred_val is None:
        return False
    
    # Check exact match or match within 1% relative tolerance
    if ground_truth_amount == 0:
        return abs(pred_val) < 1e-5
    
    diff = abs(pred_val - ground_truth_amount)
    relative_error = diff / abs(ground_truth_amount)
    return relative_error <= tolerance_pct

def check_retrieval_recall(retrieved_docs: List[Dict[str, Any]], target_record_id: str) -> bool:
    """Checks if the target ground-truth document was retrieved in Top-K (Recall@5) (A-14)."""
    retrieved_ids = {doc.get("record_id") for doc in retrieved_docs}
    return target_record_id in retrieved_ids

def compute_factual_recall(response_text: str, target_company: str, target_period: str, target_indicator: str, is_correct: bool) -> float:
    """Calculates granular factual recall of ground-truth query elements (company, period, metric, value)."""
    lower_text = response_text.lower()
    comp_parts = [p.lower() for p in re.findall(r"\w+", target_company) if len(p) > 2]
    has_company = any(p in lower_text for p in comp_parts) if comp_parts else False
    has_period = target_period.lower() in lower_text
    ind_parts = [p.lower() for p in re.findall(r"\w+", target_indicator) if len(p) > 3]
    has_indicator = any(p in lower_text for p in ind_parts) if ind_parts else True
    has_value = is_correct
    elements = [has_company, has_period, has_indicator, has_value]
    return sum(elements) / len(elements)

def check_entity_recall(response_text: str, target_company: str, target_period: str) -> bool:
    """Evaluates answer entity presence recall (company + period) for baseline LLM answers."""
    lower_text = response_text.lower()
    comp_parts = target_company.lower().split()
    first_name = comp_parts[0] if comp_parts else ""
    return (first_name in lower_text) and (target_period.lower() in lower_text)

def run_group_evaluation(group_code: str, group_spec: Dict[str, Any], qa_dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Executes evaluation for a single experimental group (BG, REG, VUG, FOG) with latency and memory tracking."""
    print(f"\n--- Evaluating Group: {group_code} (Model: {group_spec['llm']}, RAG: {group_spec['rag']}) ---")
    
    tracemalloc.start()
    
    pipeline = RAGPipeline(
        use_rag=group_spec["rag"],
        model_name=group_spec["llm"]
    )
    
    correct_count = 0
    recalled_count = 0
    entity_recalled_count = 0
    total_fact_recall = 0.0
    total_latency = 0.0
    
    category_breakdown = {
        "well_known": {"total": 0, "correct": 0, "recalled": 0},
        "obscure": {"total": 0, "correct": 0, "recalled": 0}
    }
    
    eval_records = []
    
    for item in qa_dataset:
        q_id = item["question_id"]
        question = item["question"]
        gt_amount = item["ground_truth_amount"]
        target_id = item["target_record_id"]
        category = item["category"]
        company = item["company"]
        period = item["period"]
        indicator = item.get("indicator", "")
        
        # Measure per-query response latency
        t_query_start = time.perf_counter()
        res = pipeline.query(question)
        latency_sec = time.perf_counter() - t_query_start
        total_latency += latency_sec
        
        ans_text = res["answer"]
        retrieved_docs = res["retrieved_docs"]
        
        # Calculate metrics dynamically
        is_correct = check_numeric_accuracy(ans_text, gt_amount)
        is_recalled = check_retrieval_recall(retrieved_docs, target_id) if group_spec["rag"] else False
        is_entity_recalled = check_entity_recall(ans_text, company, period)
        fact_recall = compute_factual_recall(ans_text, company, period, indicator, is_correct)
        total_fact_recall += fact_recall
        
        if is_correct:
            correct_count += 1
            category_breakdown[category]["correct"] += 1
            
        if is_recalled:
            recalled_count += 1
            category_breakdown[category]["recalled"] += 1
            
        if is_entity_recalled:
            entity_recalled_count += 1
            
        category_breakdown[category]["total"] += 1
        
        eval_records.append({
            "question_id": q_id,
            "category": category,
            "question": question,
            "ground_truth": gt_amount,
            "llm_response": ans_text,
            "extracted_val": extract_numeric_value(ans_text),
            "latency_seconds": round(latency_sec, 4),
            "is_correct": is_correct,
            "is_recalled": is_recalled,
            "is_entity_recalled": is_entity_recalled,
            "factual_recall": round(fact_recall, 4)
        })

    total_q = len(qa_dataset)
    accuracy = (correct_count / total_q) * 100 if total_q > 0 else 0.0
    overall_recall = (total_fact_recall / total_q) * 100 if total_q > 0 else 0.0
    retrieval_recall = (recalled_count / total_q) * 100 if (total_q > 0 and group_spec["rag"]) else 0.0
    entity_recall = (entity_recalled_count / total_q) * 100 if total_q > 0 else 0.0
    avg_latency = total_latency / total_q if total_q > 0 else 0.0
    
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mem_mb = round(peak_mem / (1024 * 1024), 2)

    print(f"[{group_code}] Accuracy: {accuracy:.2f}%")
    print(f"[{group_code}] Answer/Factual Recall: {overall_recall:.2f}%")
    if group_spec["rag"]:
        print(f"[{group_code}] Retrieval Recall@5: {retrieval_recall:.2f}%")
    print(f"[{group_code}] Entity Presence: {entity_recall:.2f}%")
    print(f"[{group_code}] Avg Response Latency: {avg_latency:.4f}s")
    print(f"[{group_code}] Peak Memory Usage: {peak_mem_mb} MB")
        
    return {
        "group_code": group_code,
        "config": group_spec,
        "metrics": {
            "overall_accuracy_pct": round(accuracy, 2),
            "recall_pct": round(overall_recall, 2),
            "overall_recall_at_5_pct": round(retrieval_recall, 2),
            "entity_answer_recall_pct": round(entity_recall, 2),
            "avg_latency_seconds": round(avg_latency, 4),
            "peak_memory_mb": peak_mem_mb,
            "category_breakdown": category_breakdown
        },
        "detailed_results": eval_records
    }

def main():
    if not QA_TEST_SET_PATH.exists():
        raise FileNotFoundError(f"Missing question set. Run src/qa_generator.py first.")
        
    qa_data = json.loads(QA_TEST_SET_PATH.read_text(encoding="utf-8"))
    print(f"Loaded {len(qa_data)} test questions.")
    
    summary_results = {
        "paper_targets": PAPER_RESULTS,
        "paper_benchmarks": PAPER_BENCHMARKS,
        "reproduction_results": {},
        "comparison_analysis": {}
    }

    # Evaluate each experimental group (BG, REG, VUG, FOG)
    for group_code, group_spec in EXPERIMENTAL_GROUPS.items():
        res = run_group_evaluation(group_code, group_spec, qa_data)
        summary_results["reproduction_results"][group_code] = res

    # Compute response time reduction comparison (Target: 34.8% reduction)
    bg_latency = summary_results["reproduction_results"]["BG"]["metrics"]["avg_latency_seconds"]
    fog_latency = summary_results["reproduction_results"]["FOG"]["metrics"]["avg_latency_seconds"]
    if bg_latency > 0:
        latency_reduction = round(((bg_latency - fog_latency) / bg_latency) * 100, 2)
    else:
        latency_reduction = 0.0

    summary_results["comparison_analysis"] = {
        "latency_reduction_pct": latency_reduction,
        "paper_target_latency_reduction_pct": PAPER_BENCHMARKS["response_time_reduction_pct"],
        "target_fog_accuracy": PAPER_RESULTS["gpt-3.5-turbo-1106+RAG"][0],
        "target_fog_recall": PAPER_RESULTS["gpt-3.5-turbo-1106+RAG"][1],
    }

    RESULTS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_OUTPUT_PATH.write_text(json.dumps(summary_results, indent=2), encoding="utf-8")
    print(f"\nEvaluation complete. Full report written to: {RESULTS_OUTPUT_PATH}")

if __name__ == "__main__":
    main()