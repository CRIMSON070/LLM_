import json
import pathlib
import pandas as pd
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW_FILE_PATH = ROOT / "data" / "raw" / "original_dataset.csv"
OUT_DIR = ROOT / "data" / "processed"
REPORT_PATH = ROOT / "reports" / "preprocessing_steps.md"
GROUPING_KEYS = ["period", "company", "tickers", "indicator", "unit"]
QUARTER_END_DATES = {"1": "03-31", "2": "06-30", "3": "09-30", "4": "12-31"}

preprocessing_log = []

def log_step(name, decision_tag, description, rows_before, rows_after, note=""):
    """Records each preprocessing step for the final report."""
    preprocessing_log.append({
        "step": len(preprocessing_log) + 1,
        "name": name,
        "decision": decision_tag,
        "description": description,
        "rows_before": rows_before,
        "rows_after": rows_after,
        "rows_changed_or_removed": rows_before - rows_after,
        "note": note,
    })

def clean_ticker_string(ticker_string):
    """Removes padding, drops empty values, and deduplicates a comma-separated ticker string."""
    parts = [part.strip() for part in str(ticker_string).split(",")]
    seen, unique_tickers = set(), []
    for part in parts:
        if part and part not in seen:
            seen.add(part)
            unique_tickers.append(part)
    return ", ".join(unique_tickers)

def format_record_as_text(row):
    """Formats a single row into the required text template for embeddings (A-01)."""
    unit = "USD" if row["unit"] == "US Dollar" else row["unit"]
    return (f"[Company] {row['company']}\n[Ticker] {row['tickers']}\n"
            f"[Period] {row['period']}\n[Indicator] {row['indicator']}\n"
            f"[Amount] {row['amount']} {unit}")

def trim_whitespace(df):
    """Strips leading/trailing whitespace from all columns."""
    rows_before = len(df)
    original_cols = list(df.columns)
    
    cells_changed = sum((df[col] != df[col].astype(str).str.strip()).sum() for col in original_cols)
    for col in original_cols:
        df[col] = df[col].astype(str).str.strip()
        
    log_step("trim_whitespace", "Paper: 'cleaning'", "Strip leading/trailing whitespace across all text columns", 
             rows_before, len(df), f"{cells_changed} cell values sanitized")
    return df

def drop_missing_values(df):
    """Removes any row containing an empty field."""
    rows_before = len(df)
    empty_mask = (df == "").any(axis=1) | df.isna().any(axis=1)
    missing_count = int(empty_mask.sum())
    
    df = df[~empty_mask]
    log_step("drop_missing_values", "A-10", "Drop rows with missing or empty fields", 
             rows_before, len(df), f"{missing_count} rows dropped")
    return df

def clean_financial_amounts(df):
    """Strips currency symbols and commas, then converts amounts to integers."""
    rows_before = len(df)
    
    commas_found = int(df.amount.str.contains(",", regex=False).sum())
    symbols_found = int(df.amount.str.contains(r"[$€£¥]").sum())
    
    numeric_amounts = pd.to_numeric(df.amount.str.replace(r"[$€£¥,\s]", "", regex=True), errors="coerce")
    df = df[numeric_amounts.notna()].copy()
    df["amount"] = numeric_amounts[numeric_amounts.notna()].astype("int64")
    
    log_step("amount_to_integer", "A-05", "Strip currency symbols, commas, and cast to int64", 
             rows_before, len(df), f"{commas_found} commas and {symbols_found} currency symbols stripped")
    return df

def clean_outliers(df):
    """Handles noisy/corrupted financial outliers per paper Section 4 & A-08."""
    rows_before = len(df)
    
    # 1. Non-negativity check for indicators that cannot physically be negative
    strictly_positive_indicators = {
        "Revenues", "Revenue", "Total Revenue", "Assets", "Total Assets",
        "Cash and Cash Equivalents", "Common Stock, Shares Outstanding"
    }
    invalid_negatives = df["indicator"].isin(strictly_positive_indicators) & (df["amount"] < 0)
    cleaned_df = df[~invalid_negatives].copy()
    negatives_dropped = int(invalid_negatives.sum())
    
    # 2. Extreme corrupted values check (|amount| > 1e14, beyond the GDP of the planet)
    corrupted_bounds = cleaned_df["amount"].abs() > 1e14
    corrupted_count = int(corrupted_bounds.sum())
    cleaned_df = cleaned_df[~corrupted_bounds].copy()
    
    log_step("outlier_handling", "Paper §4 / A-08", "Clean corrupted amounts and invalid negative indicators", 
             rows_before, len(cleaned_df), 
             f"Removed {negatives_dropped} invalid negative indicators, {corrupted_count} extreme numerical errors")
    return cleaned_df

def normalize_periods(df):
    """Extracts year and quarter, and appends the standard calendar quarter-end date."""
    rows_before = len(df)
    parts = df.period.str.extract(r"^(\d{4}) Q([1-4])$")
    
    df["year"] = parts[0].astype(int)
    df["quarter"] = parts[1].astype(int)
    df["period_end"] = df.year.astype(str) + "-" + parts[1].map(QUARTER_END_DATES)
    
    log_step("normalize_period", "A-04", "Standardize financial period to calendar quarter-end date", 
             rows_before, len(df), "Mapped YYYY Q[1-4] to ISO-8601 YYYY-MM-DD")
    return df

def save_preprocessing_report():
    """Generates and writes a detailed markdown report of all preprocessing transformations."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    md_content = [
        "# Financial Dataset Preprocessing & Standardization Report",
        "",
        "**Paper Reference:** Wang, Ding & Zhu (2025), Section 4 (*Data Preprocessing*)\n",
        "This report documents all data standardization, deduplication, period normalization, and outlier cleaning steps.\n",
        "## Preprocessing Execution Log\n",
        "| Step | Action Name | Specification ID | Description | Rows Before | Rows After | Delta | Notes |",
        "| :---: | :--- | :---: | :--- | :---: | :---: | :---: | :--- |"
    ]
    
    for item in preprocessing_log:
        md_content.append(
            f"| {item['step']} | `{item['name']}` | `{item['decision']}` | {item['description']} | "
            f"{item['rows_before']:,} | {item['rows_after']:,} | {item['rows_changed_or_removed']:,} | {item['note']} |"
        )
        
    md_content.extend([
        "",
        "## Summary of Data Transformations",
        "1. **Whitespace & String Sanitization:** Stripped leading/trailing spaces across all text columns.",
        "2. **Missing Values Handling (A-10):** Purged rows with null or empty fields.",
        "3. **Financial Amount Cast (A-05):** Removed currency symbols (`$`, `€`, `£`, `¥`) and thousand-separator commas; cast to 64-bit integer.",
        "4. **Ticker Normalization:** Deduplicated repeated ticker tokens (e.g. `AAPL, AAPL` -> `AAPL`).",
        "5. **Outlier & Corruption Filtering (§4 / A-08):** Removed invalid negative values on strictly non-negative accounts (Revenues, Assets) and overflow corruptions.",
        "6. **Date Normalization (§4 / A-04):** Mapped `YYYY Q[1-4]` strings to ISO quarter-end dates (`YYYY-03-31`, `YYYY-06-30`, `YYYY-09-30`, `YYYY-12-31`).",
        "7. **USD Standardization (A-07):** Retained original currency unit; flagged `is_non_usd` and set `eval_eligible` for USD records.",
        ""
    ])
    
    REPORT_PATH.write_text("\n".join(md_content), encoding="utf-8")
    print(f"Preprocessing report written to {REPORT_PATH}")

def run_pipeline():
    """Executes the full data cleaning pipeline sequentially."""
    print("Reading raw dataset from:", RAW_FILE_PATH)
    df = pd.read_csv(RAW_FILE_PATH, dtype=str, keep_default_na=False)
    
    rows_initial = len(df)
    
    df = trim_whitespace(df)
    df = drop_missing_values(df)
    
    # Drop exact raw duplicates
    rows_before_dup = len(df)
    df = df.drop_duplicates(keep="first")
    log_step("drop_raw_duplicates", "A-06", "Drop exact duplicate rows", 
             rows_before_dup, len(df), f"{rows_before_dup - len(df)} exact duplicates dropped")
    
    df = clean_financial_amounts(df)
    
    # Normalize tickers
    rows_before_t = len(df)
    df["tickers"] = df.tickers.map(clean_ticker_string)
    df = df.drop_duplicates(keep="first")
    log_step("normalize_tickers", "Paper 'cleaning'", "Clean comma-separated ticker lists and re-deduplicate", 
             rows_before_t, len(df))
    
    # Clean outliers
    df = clean_outliers(df)
    
    # Normalize periods
    df = normalize_periods(df)
    df = df.reset_index(drop=True)
    df["record_id"] = [f"record_{index:06d}" for index in df.index]
    
    # Flag Non-USD and mark for eligibility
    df["is_non_usd"] = df.unit != "US Dollar"
    df["eval_eligible"] = ~df.is_non_usd
    
    # Save outputs
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / "cleaned_dataset.csv"
    df.to_csv(out_csv, index=False)
    
    save_preprocessing_report()
    
    print(f"Pipeline complete. Rows processed: {rows_initial:,} -> {len(df):,}. Saved to: {out_csv}")

if __name__ == "__main__":
    run_pipeline()