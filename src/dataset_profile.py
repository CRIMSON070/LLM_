import hashlib
import json
import pathlib
import pandas as pd

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "original_dataset.csv"
DUPLICATE_KEYS = ["period", "company", "tickers", "indicator"]

def generate_profile():
    """Reads the raw dataset and generates a statistical profile JSON."""
    file_hash = hashlib.sha256(RAW_DATA_PATH.read_bytes()).hexdigest()
    df = pd.read_csv(RAW_DATA_PATH, dtype=str, keep_default_na=False)
    
    # Clean amounts for statistical analysis
    df["numeric_amount"] = pd.to_numeric(df.amount.str.replace(",", "", regex=False), errors="coerce")
    
    # Calculate key metrics
    duplicates_mask = df[df.duplicated(DUPLICATE_KEYS, keep=False)]
    conflicting_amounts = duplicates_mask.groupby(DUPLICATE_KEYS)["numeric_amount"].nunique()
    periods_per_company = df.groupby("company").period.nunique()
    
    profile_report = {
        "file_metadata": {
            "path": str(RAW_DATA_PATH.relative_to(ROOT_DIR)),
            "bytes": RAW_DATA_PATH.stat().st_size,
            "sha256_hash": file_hash
        },
        "general_stats": {
            "total_rows": len(df),
            "unique_companies": int(df.company.nunique()),
            "unique_tickers": int(df.tickers.nunique()),
            "period_range": f"{df.period.min()} to {df.period.max()}"
        },
        "data_quality": {
            "exact_duplicate_rows": int(df.duplicated().sum()),
            "key_groups_with_conflicts": int((conflicting_amounts > 1).sum()),
            "non_usd_rows": int((df.unit != "US Dollar").sum()),
        },
        "financial_amounts": {
            "unparseable_values": int(df["numeric_amount"].isna().sum()),
            "negative_values": int((df["numeric_amount"] < 0).sum()),
            "zero_values": int((df["numeric_amount"] == 0).sum()),
            "median_amount": float(df["numeric_amount"].median()),
        }
    }

    # Save to disk
    output_path = ROOT_DIR / "data" / "dataset_profile.json"
    output_path.write_text(json.dumps(profile_report, indent=2, default=str))
    print("Profile generated successfully.")

if __name__ == "__main__":
    generate_profile()