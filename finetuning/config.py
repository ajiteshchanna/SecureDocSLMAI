"""
SecureDocAI - Fine-tuning Configuration
LoRA (Low-Rank Adaptation) settings for behavioral fine-tuning.

IMPORTANT:
  - Fine-tuning is for BEHAVIOR improvement (tone, structure, reasoning quality)
  - RAG handles KNOWLEDGE — never use fine-tuning to memorize documents
"""

from config import LORA_CONFIG, TRAINING_CONFIG, LORA_ADAPTERS_DIR

# ─── BASE MODEL FOR FINE-TUNING ───────────────────────────────────────────────

# Use a small model compatible with PEFT / LoRA on CPU
FINETUNE_BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# ─── LORA HYPERPARAMETERS ─────────────────────────────────────────────────────

LORA_R = LORA_CONFIG["r"]                          # Rank of adaptation matrices
LORA_ALPHA = LORA_CONFIG["lora_alpha"]             # Scaling factor
LORA_DROPOUT = LORA_CONFIG["lora_dropout"]         # Regularization
LORA_TARGET_MODULES = LORA_CONFIG["target_modules"] # Layers to adapt

# ─── TRAINING HYPERPARAMETERS ─────────────────────────────────────────────────

NUM_EPOCHS = TRAINING_CONFIG["num_train_epochs"]
BATCH_SIZE = TRAINING_CONFIG["per_device_train_batch_size"]
GRAD_ACCUM = TRAINING_CONFIG["gradient_accumulation_steps"]
LEARNING_RATE = TRAINING_CONFIG["learning_rate"]
WARMUP_STEPS = TRAINING_CONFIG["warmup_steps"]
SAVE_STEPS = TRAINING_CONFIG["save_steps"]
LOGGING_STEPS = TRAINING_CONFIG["logging_steps"]

OUTPUT_DIR = TRAINING_CONFIG["output_dir"]

# ─── DATASET FORMAT ───────────────────────────────────────────────────────────

# Expected JSON format for training data:
# [
#   {
#     "instruction": "What is habeas corpus?",
#     "context": "Habeas corpus is a legal principle...",
#     "response": "Habeas corpus is a fundamental right..."
#   },
#   ...
# ]

DATASET_PATH = "finetuning/dataset/train.json"
DATASET_FORMAT = "instruction_context_response"

# ─── PROMPT TEMPLATE FOR FINE-TUNING ─────────────────────────────────────────

FINETUNE_PROMPT_TEMPLATE = """<|system|>
You are a precise document intelligence assistant. Answer questions based only on the provided context.
</s>
<|user|>
Context: {context}

Question: {instruction}
</s>
<|assistant|>
{response}"""
