"""
SecureDocAI - Fine-tuned Model Loader
Loads the base SLM with optional LoRA adapter for enhanced behavior.
"""

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

from finetuning.config import (
    FINETUNE_BASE_MODEL,
    OUTPUT_DIR,
    FINETUNE_PROMPT_TEMPLATE,
)

logger = logging.getLogger(__name__)


def load_base_model(model_name: str = FINETUNE_BASE_MODEL):
    """
    Load the base model and tokenizer for fine-tuning or inference.

    Returns:
        Tuple of (model, tokenizer)
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"  🔧 Loading base model: {model_name}...")

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="right",  # Required for causal LM training
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        device = "cuda" if _has_gpu() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map="auto" if device == "cuda" else None,
        )

        if device == "cpu":
            model = model.to(device)

        print(f"  ✓ Base model loaded on {device}.")
        logger.info(f"Base model loaded: {model_name} on {device}")
        return model, tokenizer

    except ImportError as e:
        raise ImportError(f"Missing dependency: {e}. Run: pip install transformers torch")


def load_finetuned_model(adapter_path: Optional[str] = None):
    """
    Load the base model with LoRA adapter merged for inference.

    Args:
        adapter_path: Path to the LoRA adapter directory.
                      Defaults to OUTPUT_DIR from config.

    Returns:
        Tuple of (model, tokenizer) with adapter merged.
    """
    if adapter_path is None:
        adapter_path = OUTPUT_DIR

    if not Path(adapter_path).exists():
        raise FileNotFoundError(
            f"No fine-tuned adapter found at: {adapter_path}\n"
            f"Train first using: python -m finetuning.train_lora"
        )

    try:
        from peft import PeftModel

        model, tokenizer = load_base_model()

        print(f"  🔧 Loading LoRA adapter from: {adapter_path}...")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()  # Merge weights for faster inference
        print(f"  ✓ Fine-tuned model ready (adapter merged).")
        logger.info(f"LoRA adapter merged from {adapter_path}.")

        return model, tokenizer

    except ImportError:
        raise ImportError("PEFT not installed. Run: pip install peft")


def format_inference_prompt(
    instruction: str,
    context: str = "",
) -> str:
    """Format a prompt for fine-tuned model inference."""
    return FINETUNE_PROMPT_TEMPLATE.format(
        instruction=instruction,
        context=context,
        response="",  # Left blank for generation
    )


def _has_gpu() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False
