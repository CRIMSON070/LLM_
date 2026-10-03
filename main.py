import sys
import time
import pathlib

ROOT_DIR = pathlib.Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR / "src"))

from dataset_profile import generate_profile
from preprocessing import run_pipeline as run_preprocessing
from embeddings import generate_embeddings
from vector_db import VectorStore
from qa_generator import build_evaluation_dataset
from evaluate import main as run_evaluation

def run_full_reproduction_pipeline():
    """Executes all 6 pipeline phases sequentially."""
    start_time = time.time()
    print("=" * 60)
    print("STARTING FINANCIAL ANALYSIS RAG REPRODUCTION PIPELINE")
    print("=" * 60)
    
    print("\n[Phase 1/6] Profiling Raw Dataset...")
    generate_profile()
    
    print("\n[Phase 2/6] Running Preprocessing & Data Cleaning...")
    run_preprocessing()
    
    print("\n[Phase 3/6] Generating Document Vector Embeddings...")
    generate_embeddings()
    
    print("\n[Phase 4/6] Building FAISS Index FlatIP Store...")
    vector_store = VectorStore()
    vector_store.build_and_save_index()
    
    print("\n[Phase 5/6] Generating QA Evaluation Test Set...")
    build_evaluation_dataset(samples_per_category=50, seed=42)
    
    print("\n[Phase 6/6] Running Experimental Group Evaluations...")
    run_evaluation()
    
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"PIPELINE EXECUTED SUCCESSFULLY IN {elapsed:.2f} SECONDS")
    print("=" * 60)
    print("Run `streamlit run src/app.py` to launch the interactive user interface.")

if __name__ == "__main__":
    run_full_reproduction_pipeline()