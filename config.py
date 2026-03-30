"""
SecureDocAI - Centralized Configuration
Unified single-DB, single-folder system. No domain separation.
"""

import os
from pathlib import Path

# ─── FORCE OFFLINE MODE ───────────────────────────────────────────────────────
os.environ["TRANSFORMERS_OFFLINE"] = "1"   # Set to "1" after first model download
os.environ["HF_DATASETS_OFFLINE"]  = "1"

# ─── BASE PATHS ───────────────────────────────────────────────────────────────

BASE_DIR          = Path(__file__).resolve().parent
DATA_DIR          = BASE_DIR / "data"
RAW_DOCS_DIR      = DATA_DIR / "raw_docs"      # All documents go here
VECTOR_DB_PATH    = DATA_DIR / "vector_db"     # Single unified FAISS index
MODELS_DIR        = BASE_DIR / "models"
LORA_ADAPTERS_DIR = MODELS_DIR / "lora_adapters"

for _d in [RAW_DOCS_DIR, VECTOR_DB_PATH, MODELS_DIR, LORA_ADAPTERS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ─── DOCUMENT INGESTION ───────────────────────────────────────────────────────

CHUNK_SIZE           = 800
CHUNK_OVERLAP        = 150
SUPPORTED_EXTENSIONS = [".pdf", ".docx", ".txt"]

# ─── EMBEDDING MODEL ──────────────────────────────────────────────────────────

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DEVICE     = "cpu"

# ─── RETRIEVAL CONFIGURATION (optimized) ─────────────────────────────────────

FAISS_TOP_K        = 6
BM25_TOP_K         = 4
RERANKER_TOP_K     = 4
MAX_CONTEXT_CHARS  = 2500
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ─── SLM CONFIGURATION ────────────────────────────────────────────────────────

OLLAMA_BASE_URL             = "http://localhost:11434"
OLLAMA_MODEL                = "phi3"
TRANSFORMERS_MODEL_NAME     = "microsoft/phi-2"
TRANSFORMERS_MAX_NEW_TOKENS = 512
TRANSFORMERS_TEMPERATURE    = 0.1
TRANSFORMERS_DO_SAMPLE      = False
SLM_BACKEND                 = "ollama"

# ─── PROMPTS ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a precise document intelligence assistant.
Answer questions ONLY from the provided context.
Rules:
- Use ONLY the given context. Never use outside knowledge.
- If the answer is not in the context, say: "Not found in provided documents."
- Be concise, accurate, and structured.
- Cite source document and page number when possible.
"""

RAG_PROMPT_TEMPLATE = """Context:
{context}

Question: {question}

Answer strictly from the context above.
If the answer is unavailable, respond: "Not found in provided documents."

Answer:"""

# ─── FINE-TUNING ──────────────────────────────────────────────────────────────

LORA_CONFIG = {
    "r": 16, "lora_alpha": 32, "lora_dropout": 0.05,
    "target_modules": ["q_proj", "v_proj"],
    "bias": "none", "task_type": "CAUSAL_LM",
}

TRAINING_CONFIG = {
    "output_dir": str(LORA_ADAPTERS_DIR),
    "num_train_epochs": 3,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "warmup_steps": 50, "learning_rate": 2e-4,
    "fp16": False, "logging_steps": 10, "save_steps": 100,
}

# ─── CLI DISPLAY ──────────────────────────────────────────────────────────────

APP_NAME        = "SecureDocAI"
APP_VERSION     = "2.0.0"
APP_DESCRIPTION = "Fully Offline Document Intelligence | SLM + RAG"

BANNER = """
╔══════════════════════════════════════════════════════════╗
║                 SecureDocAI v2.0.0                       ║
║        Fully Offline Document Intelligence | SLM + RAG   ║
║                 100% Offline & Private                   ║
╚══════════════════════════════════════════════════════════╝
"""