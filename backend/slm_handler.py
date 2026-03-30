"""
SecureDocAI - SLM Handler
Manages the Small Language Model (SLM) for text generation.
Supports two backends:
  1. Ollama (recommended) — fastest, most capable offline inference
  2. HuggingFace Transformers — fallback if Ollama is unavailable

Both backends run 100% offline.
"""

import logging
from typing import Optional, Tuple

from config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    TRANSFORMERS_MODEL_NAME,
    TRANSFORMERS_MAX_NEW_TOKENS,
    TRANSFORMERS_TEMPERATURE,
    TRANSFORMERS_DO_SAMPLE,
    SLM_BACKEND,
    SYSTEM_PROMPT,
    RAG_PROMPT_TEMPLATE,
)

logger = logging.getLogger(__name__)

# ─── OLLAMA BACKEND ───────────────────────────────────────────────────────────

class OllamaBackend:
    """
    Uses Ollama's local inference server for generation.
    Ollama must be installed and running: https://ollama.ai
    Pull the model first: `ollama pull phi3`
    """

    def __init__(self, model: str = OLLAMA_MODEL, base_url: str = OLLAMA_BASE_URL):
        self.model = model
        self.base_url = base_url
        self._client = None
        self._initialize()

    def _initialize(self):
        try:
            from langchain_community.llms import Ollama
            self._client = Ollama(
                model=self.model,
                base_url=self.base_url,
                temperature=0.1,
                num_predict=512,
                system=SYSTEM_PROMPT,
            )
            # Quick connectivity test
            self._check_connection()
            print(f"  ✓ Ollama backend ready (model: {self.model})")
            logger.info(f"Ollama backend initialized with model: {self.model}")
        except Exception as e:
            logger.error(f"Ollama initialization failed: {e}")
            raise

    def _check_connection(self):
        """Verify Ollama server is reachable."""
        import urllib.request
        try:
            with urllib.request.urlopen(self.base_url, timeout=3) as resp:
                pass
        except Exception:
            raise ConnectionError(
                f"Cannot reach Ollama at {self.base_url}.\n"
                f"Please ensure Ollama is installed and running:\n"
                f"  1. Install: https://ollama.ai\n"
                f"  2. Start:   ollama serve\n"
                f"  3. Pull:    ollama pull {self.model}"
            )

    def generate(self, prompt: str) -> str:
        """Generate text from prompt using Ollama."""
        try:
            response = self._client.invoke(prompt)
            return response.strip()
        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            raise

    def get_model_info(self) -> dict:
        return {"backend": "ollama", "model": self.model, "url": self.base_url}


# ─── TRANSFORMERS BACKEND ─────────────────────────────────────────────────────

class TransformersBackend:
    """
    Uses HuggingFace Transformers for local inference.
    Models are downloaded once and cached locally (~/.cache/huggingface).
    No internet required after first download.
    """

    def __init__(
        self,
        model_name: str = TRANSFORMERS_MODEL_NAME,
        max_new_tokens: int = TRANSFORMERS_MAX_NEW_TOKENS,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self._pipeline = None
        self._initialize()

    def _initialize(self):
        """Load the model and tokenizer."""
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

            print(f"  🔧 Loading SLM: {self.model_name} (this may take a moment)...")

            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )

            # Load with appropriate precision
            device = "cuda" if self._has_gpu() else "cpu"
            dtype = torch.float16 if device == "cuda" else torch.float32

            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )

            self._pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=0 if device == "cuda" else -1,
                max_new_tokens=self.max_new_tokens,
                temperature=TRANSFORMERS_TEMPERATURE,
                do_sample=TRANSFORMERS_DO_SAMPLE,
                pad_token_id=tokenizer.eos_token_id,
                return_full_text=False,
            )

            print(f"  ✓ Transformers backend ready (model: {self.model_name}, device: {device})")
            logger.info(f"Transformers backend loaded: {self.model_name} on {device}")

        except ImportError:
            raise ImportError(
                "transformers or torch not installed.\n"
                "Run: pip install transformers torch"
            )
        except Exception as e:
            logger.error(f"Transformers initialization failed: {e}")
            raise

    def _has_gpu(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    def generate(self, prompt: str) -> str:
        """Generate text from prompt using local Transformers pipeline."""
        try:
            full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"
            outputs = self._pipeline(full_prompt)
            text = outputs[0]["generated_text"].strip()
            return text
        except Exception as e:
            logger.error(f"Transformers generation error: {e}")
            raise

    def get_model_info(self) -> dict:
        return {"backend": "transformers", "model": self.model_name}


# ─── SLM HANDLER (UNIFIED INTERFACE) ─────────────────────────────────────────

class SLMHandler:
    """
    Unified interface for the Small Language Model.
    Selects the appropriate backend (Ollama or Transformers) based on config.
    Falls back to Transformers if Ollama is unavailable.
    """

    def __init__(self, backend: str = SLM_BACKEND):
        self.backend_name = backend
        self._backend = None
        self._load_backend(backend)

    def _load_backend(self, backend: str):
        if backend == "ollama":
            try:
                self._backend = OllamaBackend()
                self.backend_name = "ollama"
            except Exception as e:
                print(f"\n  ⚠  Ollama unavailable: {e}")
                print("  🔄 Falling back to Transformers backend...")
                self._load_backend("transformers")
        elif backend == "transformers":
            try:
                self._backend = TransformersBackend()
                self.backend_name = "transformers"
            except Exception as e:
                raise RuntimeError(
                    f"Both Ollama and Transformers backends failed to initialize.\n"
                    f"Last error: {e}\n\n"
                    f"To fix:\n"
                    f"  Option A: Install Ollama → https://ollama.ai then run: ollama pull phi3\n"
                    f"  Option B: pip install transformers torch"
                )
        else:
            raise ValueError(f"Unknown backend: '{backend}'. Choose 'ollama' or 'transformers'.")

    def generate_answer(self, question: str, context: str) -> str:
        """
        Generate a grounded answer from context using the SLM.

        Args:
            question: User's natural language question.
            context: Retrieved document context.

        Returns:
            Generated answer string.
        """
        prompt = RAG_PROMPT_TEMPLATE.format(
            context=context,
            question=question,
        )

        try:
            answer = self._backend.generate(prompt)
            return answer
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return f"Error generating answer: {e}"

    def get_model_info(self) -> dict:
        """Return current model metadata."""
        info = self._backend.get_model_info() if self._backend else {}
        info["active_backend"] = self.backend_name
        return info


# ─── SINGLETON ────────────────────────────────────────────────────────────────

_slm_instance: Optional[SLMHandler] = None


def get_slm(backend: str = SLM_BACKEND) -> SLMHandler:
    """Return a cached SLMHandler instance."""
    global _slm_instance
    if _slm_instance is None:
        print(f"\n  🤖 Initializing SLM (backend: {backend})...")
        _slm_instance = SLMHandler(backend=backend)
    return _slm_instance
