import pathlib
from typing import Dict, List, Any
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from config import VectorConfig

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
EMBEDDINGS_PATH = ROOT_DIR / "data" / "processed" / "embeddings.npy"
METADATA_PATH = ROOT_DIR / "data" / "processed" / "metadata.parquet"
INDEX_PATH = ROOT_DIR / "data" / "processed" / "faiss.index"

class VectorStore:
    """Manages FAISS IndexFlatIP vector index and document retrieval."""
    
    def __init__(self, config: VectorConfig = VectorConfig()):
        self.config = config
        self.index: faiss.IndexFlatIP | None = None
        self.metadata: pd.DataFrame | None = None
        self.encoder: SentenceTransformer | None = None

    def build_and_save_index(self) -> None:
        """Loads embeddings from disk and builds a FAISS IndexFlatIP index."""
        print("Loading embeddings and metadata...")
        embeddings = np.load(EMBEDDINGS_PATH).astype(np.float32)
        self.metadata = pd.read_parquet(METADATA_PATH)
        
        dim = embeddings.shape[1]
        assert dim == self.config.vector_dim, f"Dimension mismatch: expected {self.config.vector_dim}, got {dim}"
        
        print(f"Building FAISS {self.config.index_type} index...")
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        
        print(f"Saving index to {INDEX_PATH}...")
        faiss.write_index(self.index, str(INDEX_PATH))
        print("Index successfully built and saved.")

    def load_index(self) -> None:
        """Loads an existing index and metadata from disk."""
        if not INDEX_PATH.exists() or not METADATA_PATH.exists():
            raise FileNotFoundError("Index or metadata files missing. Run build_and_save_index() first.")
            
        self.index = faiss.read_index(str(INDEX_PATH))
        self.metadata = pd.read_parquet(METADATA_PATH)
        self.encoder = SentenceTransformer(self.config.embedding_model)

    def search(self, query: str, top_k: int | None = None) -> List[Dict[str, Any]]:
        """Encodes a search query and returns top-K matching document records."""
        if self.index is None or self.encoder is None:
            self.load_index()

        k = top_k if top_k is not None else self.config.top_k
        
        # Vectorize and normalize the query string
        query_vector = self.encoder.encode(
            [query],
            normalize_embeddings=self.config.normalize_embeddings,
            convert_to_numpy=True
        ).astype(np.float32)

        # FAISS search
        scores, indices = self.index.search(query_vector, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            meta = self.metadata.iloc[idx].to_dict()
            meta["score"] = float(score)
            results.append(meta)
            
        return results

if __name__ == "__main__":
    store = VectorStore()
    store.build_and_save_index()
    
    # Sanity search check
    test_query = "What was Apple's revenue in 2014 Q1?"
    print(f"\nTest Query: '{test_query}'")
    hits = store.search(test_query, top_k=3)
    for i, hit in enumerate(hits, 1):
        print(f"{i}. {hit['company']} ({hit['tickers']}) | {hit['period']} | {hit['indicator']} | Amount: {hit['amount']} | Score: {hit['score']:.4f}")