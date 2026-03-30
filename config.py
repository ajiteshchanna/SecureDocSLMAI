"""
SecureDocAI - Centralized Configuration
"""

import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

from pathlib import Path

# ─── BASE PATHS ───────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DOCS_DIR = DATA_DIR / "raw_docs"
VECTOR_DB_DIR = DATA_DIR / "vector_db"
MODELS_DIR = BASE_DIR / "models"
LORA_ADAPTERS_DIR = MODELS_DIR / "lora_adapters"

# Ensure directories exist
for d in [RAW_DOCS_DIR, VECTOR_DB_DIR, MODELS_DIR, LORA_ADAPTERS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── DOMAIN CONFIGURATION ─────────────────────────────────────────────────────

SUPPORTED_DOMAINS = ["legal", "sports", "finance", "default", "custom"]

DOMAIN_VECTOR_DB_PATHS = {
    domain: str(VECTOR_DB_DIR / domain)
    for domain in SUPPORTED_DOMAINS
}

# ─── DOCUMENT INGESTION ───────────────────────────────────────────────────────

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
SUPPORTED_EXTENSIONS = [".pdf", ".docx", ".txt"]

# ─── EMBEDDING MODEL ──────────────────────────────────────────────────────────

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DEVICE = "cpu"  # Use 'cuda' if GPU available

# ─── RETRIEVAL CONFIGURATION ──────────────────────────────────────────────────

FAISS_TOP_K = 8          # Semantic search candidates
BM25_TOP_K = 8           # Keyword search candidates
RERANKER_TOP_K = 5       # Final chunks after re-ranking

RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ─── SLM CONFIGURATION ────────────────────────────────────────────────────────

# Ollama settings (preferred - runs fully offline)
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "phi3"               # phi3, tinyllama, mistral

# Transformers fallback (HuggingFace local)
TRANSFORMERS_MODEL_NAME = "microsoft/phi-2"
TRANSFORMERS_MAX_NEW_TOKENS = 512
TRANSFORMERS_TEMPERATURE = 0.1
TRANSFORMERS_DO_SAMPLE = False

# Which backend to use: "ollama" or "transformers"
SLM_BACKEND = "ollama"

# ─── PROMPT TEMPLATE ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a precise document intelligence assistant.
Your job is to answer questions ONLY based on the provided context.
Rules:
- Answer only from the given context. Do NOT use external knowledge.
- If the answer is not found in the context, say: "Not found in provided documents."
- Be concise, accurate, and structured.
- Always cite your sources (document name and page number).
"""

RAG_PROMPT_TEMPLATE = """Context:
{context}

Question: {question}

Instructions: Answer the question strictly based on the context above. 
If the answer is not available in the context, respond with: "Not found in provided documents."
Provide a clear, structured answer and mention the source document(s) where applicable.

Answer:"""

# ─── FINE-TUNING CONFIGURATION ────────────────────────────────────────────────

LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "v_proj"],
    "bias": "none",
    "task_type": "CAUSAL_LM",
}

TRAINING_CONFIG = {
    "output_dir": str(LORA_ADAPTERS_DIR),
    "num_train_epochs": 3,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "warmup_steps": 50,
    "learning_rate": 2e-4,
    "fp16": False,
    "logging_steps": 10,
    "save_steps": 100,
    "eval_steps": 100,
}

# ─── CLI DISPLAY ──────────────────────────────────────────────────────────────

APP_NAME = "SecureDocAI"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = "Fully Offline Document Intelligence System | SLM + RAG"

BANNER = f"""
╔══════════════════════════════════════════════════════════╗
║              {APP_NAME} v{APP_VERSION}                         ║
║      {APP_DESCRIPTION}      ║
║                   🔒 100% Offline & Private               ║
╚══════════════════════════════════════════════════════════╝
"""
