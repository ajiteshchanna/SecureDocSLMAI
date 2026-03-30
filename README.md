# 🔒 SecureDocAI

**Fully Offline, CLI-Based Document Intelligence System**  
Powered by Small Language Models (SLM) + Retrieval-Augmented Generation (RAG)

---

## 🎯 What It Does

SecureDocAI lets organizations ingest private documents (PDF, DOCX, TXT) and ask natural language questions — with accurate, context-grounded answers and full source citations — **without any internet connection or cloud API**.

---

## 🧠 Architecture

```
User Query
  ↓
Embedding (all-MiniLM-L6-v2)
  ↓
Hybrid Retrieval:
  ├── Semantic Search (FAISS, Top-8)
  └── Keyword Search  (BM25,  Top-8)
  ↓
Merge + Deduplicate
  ↓
Re-rank (CrossEncoder: ms-marco-MiniLM-L-6-v2, Top-5)
  ↓
Context Construction
  ↓
SLM Generation (Phi-3 via Ollama / TinyLlama via Transformers)
  ↓
Answer + Source Citations
```

---

## 📁 Project Structure

```
SecureDocAI/
├── main.py                  # CLI entry point
├── test.py                  # Full test suite
├── config.py                # Centralized configuration
├── requirements.txt
│
├── backend/
│   ├── __init__.py
│   ├── ingest.py            # Document loading & chunking
│   ├── embeddings.py        # HuggingFace embedding model
│   ├── vectorstore.py       # FAISS CRUD operations
│   ├── rag_pipeline.py      # Hybrid retrieval + reranking + generation
│   └── slm_handler.py       # SLM: Ollama or Transformers backends
│
├── finetuning/
│   ├── __init__.py
│   ├── config.py            # LoRA/training configuration
│   ├── model_loader.py      # Base model + adapter loading
│   ├── train_lora.py        # LoRA fine-tuning script
│   └── dataset/
│       └── train.json       # Sample Q&A training data
│
├── data/
│   ├── raw_docs/            # Input documents (organized by domain)
│   └── vector_db/           # FAISS indices (organized by domain)
│
└── models/
    └── lora_adapters/       # Saved LoRA adapter weights
```

---

## ⚙️ Setup

### 1. Clone / create project
```bash
cd SecureDocAI
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Setup SLM backend (choose one)

**Option A — Ollama (recommended)**
```bash
# Install Ollama: https://ollama.ai/download
ollama serve          # Start Ollama server
ollama pull phi3      # Download Phi-3 (3.8B, ~2.2GB)
# Alternatives:
# ollama pull tinyllama
# ollama pull mistral
```

**Option B — Transformers (no extra install)**
```python
# In config.py, set:
SLM_BACKEND = "transformers"
TRANSFORMERS_MODEL_NAME = "microsoft/phi-2"
```
The model downloads automatically on first run (~2GB).

---

## 🚀 Running the CLI

```bash
python main.py
```

With pre-selected domain:
```bash
python main.py --domain legal
python main.py --domain finance --backend ollama
```

### CLI Menu

```
╔═══════════════════════════════════════════════════════════╗
║              SecureDocAI v1.0.0                            ║
║      Fully Offline Document Intelligence System | SLM+RAG  ║
║                   🔒 100% Offline & Private                ║
╚═══════════════════════════════════════════════════════════╝

  ──────────────────────────────────────────────────────────
    1. Select Domain
    2. Upload Documents
    3. Process Documents
    4. Ask Question
    5. System Status
    6. Exit
  ──────────────────────────────────────────────────────────
```

### Example Flow

```
▶ Select option: 1
▶ Enter domain name or number: legal
✅ Domain set to: 'legal'

▶ Select option: 2
▶ File path: /path/to/constitution.pdf
✅ Copied: constitution.pdf

▶ Select option: 3
  📂 Scanning: data/raw_docs/legal
  ✓ Loaded: constitution.pdf (42 pages)
  📄 Total raw pages: 42
  🔪 Total chunks: 158
  ✅ Knowledge base ready for domain: 'legal'

▶ Select option: 4
▶ Your question: What does Article 21 guarantee?

  🔍 Step 1/4: Semantic search (FAISS)...
  🔍 Step 2/4: Keyword search (BM25)...
  🔀 Step 3/4: Merging & deduplicating results...
  📊 Step 4/4: Re-ranking with CrossEncoder...
  🤖 Generating answer with SLM...

  ────────────────────────────────────────────────────────────
  📋 ANSWER

  Article 21 of the Indian Constitution guarantees the right
  to life and personal liberty. No person shall be deprived
  of these rights except according to procedure established
  by law. The Supreme Court has interpreted this broadly to
  include the right to livelihood, health, and dignity.

  📎 SOURCES
    • constitution.pdf  (Page 12)
    • constitution.pdf  (Page 15)
  ────────────────────────────────────────────────────────────
```

---

## 🌐 Supported Domains

| Domain  | Vector DB Path         | Use Case                       |
|---------|------------------------|--------------------------------|
| legal   | data/vector_db/legal/  | Contracts, laws, regulations   |
| finance | data/vector_db/finance/| Reports, filings, audits       |
| sports  | data/vector_db/sports/ | Rulebooks, stats, history      |
| default | data/vector_db/default/| General purpose                |
| custom  | data/vector_db/custom/ | Ad-hoc / session uploads       |

---

## 🧪 Testing

```bash
# Full test suite
python test.py

# Skip slow model loading
python test.py --quick

# Test specific component
python test.py --component ingest
python test.py --component retrieval
python test.py --component pipeline
```

---

## 🧠 Optional: LoRA Fine-tuning

Fine-tune the SLM's *behavior* (not its knowledge — RAG handles that).

```bash
# Add training examples to:
finetuning/dataset/train.json

# Train
python -m finetuning.train_lora

# Or with custom dataset:
python -m finetuning.train_lora --dataset my_data.json
```

**Training data format:**
```json
[
  {
    "instruction": "What is habeas corpus?",
    "context": "Habeas corpus is a legal writ...",
    "response": "Habeas corpus is a fundamental protection..."
  }
]
```

---

## ⚙️ Configuration

All settings in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CHUNK_SIZE` | 800 | Characters per chunk |
| `CHUNK_OVERLAP` | 150 | Overlap between chunks |
| `FAISS_TOP_K` | 8 | Semantic search candidates |
| `BM25_TOP_K` | 8 | Keyword search candidates |
| `RERANKER_TOP_K` | 5 | Final chunks after re-ranking |
| `SLM_BACKEND` | `"ollama"` | `"ollama"` or `"transformers"` |
| `OLLAMA_MODEL` | `"phi3"` | Ollama model name |
| `EMBEDDING_MODEL_NAME` | `all-MiniLM-L6-v2` | Embedding model |

---

## 🔒 Privacy Guarantees

- ✅ No API calls — all inference is local
- ✅ No data leaves your machine
- ✅ No telemetry or logging to external services
- ✅ FAISS indices stored on local filesystem
- ✅ Models downloaded once and cached locally
- ✅ Windows, macOS, Linux compatible

---

## 🛠️ Troubleshooting

**Ollama not found:**
```
Cannot reach Ollama at http://localhost:11434
```
→ Run `ollama serve` in a separate terminal, then `ollama pull phi3`.

**Out of memory:**
→ Use `tinyllama` instead of `phi3`: set `OLLAMA_MODEL = "tinyllama"` in `config.py`.

**FAISS not found:**
→ Run `pip install faiss-cpu`

**BM25 warnings:**
→ Run `pip install rank-bm25` (optional but recommended)
