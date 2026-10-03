import json
import pathlib
import pandas as pd

try:
    from config import VectorConfig
except ImportError:
    from src.config import VectorConfig

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
CLEANED_DATA_PATH = ROOT_DIR / "data" / "processed" / "cleaned_dataset.csv"
OUTPUT_QA_PATH = ROOT_DIR / "data" / "processed" / "eval_questions.json"

# Tickers recognized as globally prominent for testing memorization vs. retrieval (A-15)
WELL_KNOWN_TICKERS = {"AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA", "META", "TSLA", "NFLX", "INTC"}

def classify_company_type(tickers_str: str) -> str:
    """Classifies a company as 'well_known' if its tickers contain major tech symbols, else 'obscure'."""
    tickers = {t.strip() for t in str(tickers_str).split(",")}
    return "well_known" if bool(tickers & WELL_KNOWN_TICKERS) else "obscure"

def generate_question_templates(row: pd.Series) -> str:
    """Generates a structured financial query based on a cleaned dataset row."""
    return f"What was the {row['indicator']} for {row['company']} ({row['tickers']}) in {row['period']}?"

def build_evaluation_dataset(samples_per_category: int = 50, seed: int = 42) -> None:
    """Generates balanced QA evaluation samples across well-known and obscure companies."""
    print(f"Loading cleaned data from {CLEANED_DATA_PATH}...")
    df = pd.read_csv(CLEANED_DATA_PATH, dtype=str)
    
    # Filter strictly for evaluation-eligible rows (A-06, A-07)
    eligible_df = df[df["eval_eligible"].astype(str) == "True"].copy()
    
    # Tag company classification
    eligible_df["category"] = eligible_df["tickers"].apply(classify_company_type)
    
    well_known_df = eligible_df[eligible_df["category"] == "well_known"]
    obscure_df = eligible_df[eligible_df["category"] == "obscure"]
    
    # Sample proportionally
    well_known_sample = well_known_df.sample(
        n=min(samples_per_category, len(well_known_df)), random_state=seed
    )
    obscure_sample = obscure_df.sample(
        n=min(samples_per_category, len(obscure_df)), random_state=seed
    )
    
    sample_df = pd.concat([well_known_sample, obscure_sample]).reset_index(drop=True)
    
    qa_pairs = []
    for idx, row in sample_df.iterrows():
        unit = "USD" if row["unit"] == "US Dollar" else row["unit"]
        qa_pairs.append({
            "question_id": f"q_{str(idx + 1).zfill(4)}",
            "company": row["company"],
            "ticker": row["tickers"],
            "period": row["period"],
            "indicator": row["indicator"],
            "category": row["category"],
            "question": generate_question_templates(row),
            "ground_truth_amount": int(row["amount"]),
            "ground_truth_text": f"{row['amount']} {unit}",
            "target_record_id": row["record_id"]
        })
        
    print(f"Generated {len(qa_pairs)} evaluation questions ({len(well_known_sample)} well-known, {len(obscure_sample)} obscure).")
    
    OUTPUT_QA_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_QA_PATH.write_text(json.dumps(qa_pairs, indent=2), encoding="utf-8")
    print(f"Saved evaluation questions to {OUTPUT_QA_PATH}")

if __name__ == "__main__":
    build_evaluation_dataset()