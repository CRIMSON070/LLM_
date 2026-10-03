# Financial Dataset Preprocessing & Standardization Report

**Paper Reference:** Wang, Ding & Zhu (2025), Section 4 (*Data Preprocessing*)

This report documents all data standardization, deduplication, period normalization, and outlier cleaning steps.

## Preprocessing Execution Log

| Step | Action Name | Specification ID | Description | Rows Before | Rows After | Delta | Notes |
| :---: | :--- | :---: | :--- | :---: | :---: | :---: | :--- |
| 1 | `trim_whitespace` | `Paper: 'cleaning'` | Strip leading/trailing whitespace across all text columns | 186,336 | 186,336 | 0 | 326 cell values sanitized |
| 2 | `drop_missing_values` | `A-10` | Drop rows with missing or empty fields | 186,336 | 186,336 | 0 | 0 rows dropped |
| 3 | `drop_raw_duplicates` | `A-06` | Drop exact duplicate rows | 186,336 | 186,286 | 50 | 50 exact duplicates dropped |
| 4 | `amount_to_integer` | `A-05` | Strip currency symbols, commas, and cast to int64 | 186,286 | 186,286 | 0 | 185028 commas and 0 currency symbols stripped |
| 5 | `normalize_tickers` | `Paper 'cleaning'` | Clean comma-separated ticker lists and re-deduplicate | 186,286 | 186,286 | 0 |  |
| 6 | `outlier_handling` | `Paper §4 / A-08` | Clean corrupted amounts and invalid negative indicators | 186,286 | 186,286 | 0 | Removed 0 invalid negative indicators, 0 extreme numerical errors |
| 7 | `normalize_period` | `A-04` | Standardize financial period to calendar quarter-end date | 186,286 | 186,286 | 0 | Mapped YYYY Q[1-4] to ISO-8601 YYYY-MM-DD |

## Summary of Data Transformations
1. **Whitespace & String Sanitization:** Stripped leading/trailing spaces across all text columns.
2. **Missing Values Handling (A-10):** Purged rows with null or empty fields.
3. **Financial Amount Cast (A-05):** Removed currency symbols (`$`, `€`, `£`, `¥`) and thousand-separator commas; cast to 64-bit integer.
4. **Ticker Normalization:** Deduplicated repeated ticker tokens (e.g. `AAPL, AAPL` -> `AAPL`).
5. **Outlier & Corruption Filtering (§4 / A-08):** Removed invalid negative values on strictly non-negative accounts (Revenues, Assets) and overflow corruptions.
6. **Date Normalization (§4 / A-04):** Mapped `YYYY Q[1-4]` strings to ISO quarter-end dates (`YYYY-03-31`, `YYYY-06-30`, `YYYY-09-30`, `YYYY-12-31`).
7. **USD Standardization (A-07):** Retained original currency unit; flagged `is_non_usd` and set `eval_eligible` for USD records.
