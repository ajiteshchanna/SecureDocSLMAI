"""
SecureDocAI - LoRA Fine-tuning Script
Trains a LoRA adapter on top of the base SLM to improve:
  - Answer structure and formatting
  - Tone (precise, professional, document-grounded)
  - Reasoning quality

CRITICAL REMINDER:
  → Fine-tuning does NOT store knowledge. RAG handles knowledge retrieval.
  → LoRA only adjusts behavioral tendencies of the model.

Usage:
  python -m finetuning.train_lora
  python -m finetuning.train_lora --dataset finetuning/dataset/train.json
"""

import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict

logger = logging.getLogger(__name__)


# ─── DATASET LOADER ───────────────────────────────────────────────────────────

def load_dataset(dataset_path: str) -> List[Dict]:
    """
    Load Q&A training examples from a JSON file.

    Expected format:
    [
        {
            "instruction": "What is habeas corpus?",
            "context": "Habeas corpus is a legal principle...",
            "response": "Habeas corpus protects individuals from..."
        },
        ...
    ]
    """
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}\n"
            f"Create a JSON file at: finetuning/dataset/train.json"
        )

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or not data:
        raise ValueError("Dataset must be a non-empty JSON array.")

    required_keys = {"instruction", "response"}
    for i, item in enumerate(data):
        missing = required_keys - set(item.keys())
        if missing:
            raise ValueError(f"Item {i} missing keys: {missing}")

    logger.info(f"Loaded {len(data)} training examples from {dataset_path}.")
    return data


def format_training_examples(data: List[Dict], tokenizer) -> "datasets.Dataset":
    """Format and tokenize training examples."""
    from finetuning.config import FINETUNE_PROMPT_TEMPLATE
    import datasets

    def format_example(item):
        prompt = FINETUNE_PROMPT_TEMPLATE.format(
            instruction=item["instruction"],
            context=item.get("context", ""),
            response=item["response"],
        )
        return {"text": prompt}

    formatted = [format_example(item) for item in data]
    dataset = datasets.Dataset.from_list(formatted)
    return dataset


# ─── TRAINING ─────────────────────────────────────────────────────────────────

def train(dataset_path: str = None):
    """
    Full LoRA fine-tuning pipeline.

    Args:
        dataset_path: Path to training JSON. Defaults to finetuning/config.DATASET_PATH.
    """
    try:
        import torch
        from peft import LoraConfig, get_peft_model, TaskType
        from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
    except ImportError as e:
        raise ImportError(
            f"Missing dependency: {e}\n"
            f"Install with: pip install peft transformers datasets torch"
        )

    from finetuning.config import (
        DATASET_PATH, OUTPUT_DIR,
        LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES,
        NUM_EPOCHS, BATCH_SIZE, GRAD_ACCUM, LEARNING_RATE,
        WARMUP_STEPS, SAVE_STEPS, LOGGING_STEPS,
    )
    from finetuning.model_loader import load_base_model

    if dataset_path is None:
        dataset_path = DATASET_PATH

    print("\n" + "─" * 55)
    print("  SecureDocAI — LoRA Fine-tuning")
    print("─" * 55)
    print(f"  Dataset:    {dataset_path}")
    print(f"  Output dir: {OUTPUT_DIR}")
    print("─" * 55 + "\n")

    # Step 1: Load model
    model, tokenizer = load_base_model()

    # Step 2: Configure LoRA
    print("  🔧 Configuring LoRA adapter...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Step 3: Load and format dataset
    print(f"\n  📂 Loading dataset: {dataset_path}...")
    raw_data = load_dataset(dataset_path)
    dataset = format_training_examples(raw_data, tokenizer)

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding="max_length",
        )

    tokenized_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.1)

    print(f"  ✓ Training examples: {len(tokenized_dataset['train'])}")
    print(f"  ✓ Validation examples: {len(tokenized_dataset['test'])}")

    # Step 4: Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        warmup_steps=WARMUP_STEPS,
        learning_rate=LEARNING_RATE,
        fp16=torch.cuda.is_available(),
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        evaluation_strategy="steps",
        eval_steps=SAVE_STEPS,
        load_best_model_at_end=True,
        report_to="none",  # No wandb/tensorboard — keep offline
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Step 5: Train
    print("\n  🚀 Starting training...\n")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=data_collator,
    )

    trainer.train()

    # Step 6: Save adapter
    print(f"\n  💾 Saving LoRA adapter to: {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("\n  ✅ Fine-tuning complete!")
    print(f"  LoRA adapter saved to: {OUTPUT_DIR}")
    print("\n  To use the fine-tuned model, set SLM_BACKEND='transformers'")
    print("  and point the model path to the adapter directory.")


# ─── CLI ENTRY ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SecureDocAI — LoRA Fine-tuning")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to training JSON file (default: finetuning/dataset/train.json)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    train(dataset_path=args.dataset)
