import pathlib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from config import VectorConfig

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
CLEANED_DATA_PATH = ROOT_DIR / "data" / "processed" / "cleaned_dataset.csv"
EMBEDDINGS_OUTPUT_PATH = ROOT_DIR / "data" / "processed" / "embeddings.npy"
METADATA_OUTPUT_PATH = ROOT_DIR / "data" / "processed" / "metadata.parquet"

def format_record_to_text(row: pd.Series) -> str:
    """Formats a row into the structured text template (A-01)."""
    unit = "USD" if row["unit"] == "US Dollar" else row["unit"]
    return (
        f"[Company] {row['company']}\n"
        f"[Ticker] {row['tickers']}\n"
        f"[Period] {row['period']}\n"
        f"[Indicator] {row['indicator']}\n"
        f"[Amount] {row['amount']} {unit}"
    )

def generate_embeddings() -> None:
    """Loads cleaned data, generates normalized vector embeddings, and saves outputs."""
    config = VectorConfig()
    
    print(f"Loading cleaned dataset from {CLEANED_DATA_PATH}...")
    df = pd.read_csv(CLEANED_DATA_PATH, dtype=str)
    
    # Format each record as a text document
    print("Formatting records to text representation...")
    texts = df.apply(format_record_to_text, axis=1).tolist()
    
    # Load embedding model
    print(f"Loading transformer model '{config.embedding_model}'...")
    model = SentenceTransformer(config.embedding_model)
    model.max_seq_length = config.max_seq_length
    
    # Generate embeddings
    print("Encoding document texts...")
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=config.normalize_embeddings,  # Ensures IP == Cosine (A-03)
        convert_to_numpy=True
    ).astype(np.float32)
    
    # Save outputs
    print("Saving vector matrix and metadata...")
    np.save(EMBEDDINGS_OUTPUT_PATH, embeddings)
    
    # Store metadata matching exact row order of embeddings
    df[["record_id", "period", "company", "tickers", "indicator", "amount", "eval_eligible"]].to_parquet(
        METADATA_OUTPUT_PATH, index=False
    )
    
    print(f"Saved {embeddings.shape[0]} vectors of dimension {embeddings.shape[1]}.")

if __name__ == "__main__":
    generate_embeddings()