"""
SecureDocAI - SLM Handler
Singleton SLM — loaded once, reused for every query.
Supports Ollama (preferred) with auto-fallback to Transformers.
"""

import logging
from typing import Optional
from config import (
    OLLAMA_BASE_URL, OLLAMA_MODEL,
    TRANSFORMERS_MODEL_NAME, TRANSFORMERS_MAX_NEW_TOKENS,
    TRANSFORMERS_TEMPERATURE, TRANSFORMERS_DO_SAMPLE,
    SLM_BACKEND, SYSTEM_PROMPT, RAG_PROMPT_TEMPLATE,
)

logger = logging.getLogger(__name__)


# ─── OLLAMA BACKEND ───────────────────────────────────────────────────────────

class OllamaBackend:
    def __init__(self):
        from langchain_community.llms import Ollama
        self._check_server()
        self._llm = Ollama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.1,
            num_predict=512,
            system=SYSTEM_PROMPT,
        )
        print(f"  ✓ Ollama ready  (model: {OLLAMA_MODEL})")

    def _check_server(self):
        import urllib.request
        try:
            urllib.request.urlopen(OLLAMA_BASE_URL, timeout=3)
        except Exception:
            raise ConnectionError(
                f"Ollama not reachable at {OLLAMA_BASE_URL}.\n"
                f"Run: ollama serve   then   ollama pull {OLLAMA_MODEL}"
            )

    def generate(self, prompt: str) -> str:
        return self._llm.invoke(prompt).strip()

    def info(self) -> dict:
        return {"backend": "ollama", "model": OLLAMA_MODEL}


# ─── TRANSFORMERS BACKEND ─────────────────────────────────────────────────────

class TransformersBackend:
    def __init__(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

        print(f"  🔧 Loading SLM: {TRANSFORMERS_MODEL_NAME}...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype  = torch.float16 if device == "cuda" else torch.float32

        tok = AutoTokenizer.from_pretrained(
            TRANSFORMERS_MODEL_NAME, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            TRANSFORMERS_MODEL_NAME, torch_dtype=dtype,
            trust_remote_code=True, low_cpu_mem_usage=True)

        self._pipe = pipeline(
            "text-generation", model=model, tokenizer=tok,
            device=0 if device == "cuda" else -1,
            max_new_tokens=TRANSFORMERS_MAX_NEW_TOKENS,
            temperature=TRANSFORMERS_TEMPERATURE,
            do_sample=TRANSFORMERS_DO_SAMPLE,
            pad_token_id=tok.eos_token_id,
            return_full_text=False,
        )
        print(f"  ✓ Transformers SLM ready  ({TRANSFORMERS_MODEL_NAME} on {device})")

    def generate(self, prompt: str) -> str:
        out = self._pipe(f"{SYSTEM_PROMPT}\n\n{prompt}")
        return out[0]["generated_text"].strip()

    def info(self) -> dict:
        return {"backend": "transformers", "model": TRANSFORMERS_MODEL_NAME}


# ─── UNIFIED SLM HANDLER ──────────────────────────────────────────────────────

class SLMHandler:
    def __init__(self, backend: str = SLM_BACKEND):
        self._backend = self._load(backend)

    def _load(self, backend: str):
        if backend == "ollama":
            try:
                return OllamaBackend()
            except Exception as e:
                print(f"\n  ⚠  Ollama failed: {e}")
                print("  🔄 Falling back to Transformers...")
                return TransformersBackend()
        elif backend == "transformers":
            return TransformersBackend()
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def generate_answer(self, question: str, context: str) -> str:
        prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question)
        try:
            return self._backend.generate(prompt)
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return f"Error generating answer: {e}"

    def info(self) -> dict:
        return self._backend.info()


# ─── SINGLETON ────────────────────────────────────────────────────────────────

_slm: Optional[SLMHandler] = None

def get_slm() -> SLMHandler:
    """Return the singleton SLM — initialized only on first call."""
    global _slm
    if _slm is None:
        print(f"\n  🤖 Initializing SLM (backend: {SLM_BACKEND})...")
        _slm = SLMHandler()
    return _slm