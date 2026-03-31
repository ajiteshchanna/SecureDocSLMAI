"""
SecureDocAI - One-Time Model Downloader
Run this ONCE while connected to the internet.
After this, the system works fully offline forever.

Usage:
    python download_models.py
"""

import sys
from pathlib import Path

print("""
╔══════════════════════════════════════════════════════════╗
║         SecureDocAI — One-Time Model Download            ║
║   Run this once with internet. Then go fully offline.    ║
╚══════════════════════════════════════════════════════════╝
""")

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

EMBEDDING_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL   = "cross-encoder/ms-marco-MiniLM-L-6-v2"

errors = []

# ─── 1. Embedding Model ───────────────────────────────────────────────────────
print("  [1/2] Downloading embedding model...")
print(f"        {EMBEDDING_MODEL}")
try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDING_MODEL)
    # Force a test encode so all files are cached
    model.encode(["test sentence"])
    cache_path = Path(model._model_card_data.base_model_path) \
        if hasattr(model, '_model_card_data') else "HuggingFace cache"
    print(f"  ✅ Embedding model downloaded and cached.\n")
except Exception as e:
    print(f"  ❌ Failed: {e}\n")
    errors.append(f"Embedding model: {e}")

# ─── 2. Re-ranker Model ───────────────────────────────────────────────────────
print("  [2/2] Downloading re-ranker model...")
print(f"        {RERANKER_MODEL}")
try:
    from sentence_transformers import CrossEncoder
    reranker = CrossEncoder(RERANKER_MODEL)
    print(f"  ✅ Re-ranker model downloaded and cached.\n")
except Exception as e:
    print(f"  ❌ Failed: {e}\n")
    errors.append(f"Re-ranker: {e}")

# ─── Summary ──────────────────────────────────────────────────────────────────
print("─" * 58)
if not errors:
    print("""
  ✅ All models downloaded successfully!

  Next steps:
    1. Open config.py
    2. Set:  TRANSFORMERS_OFFLINE = "1"
             HF_DATASETS_OFFLINE  = "1"
    3. Disconnect internet
    4. Run:  python main.py

  Everything will now work 100% offline.
""")
else:
    print("\n  ⚠  Some downloads failed:")
    for e in errors:
        print(f"     • {e}")
    print("\n  Check your internet connection and try again.")