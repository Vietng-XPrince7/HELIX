"""
04_train_lora.py - Distributed Cloud LoRA Fine-Tuning (Unsloth + TRL)
======================================================================
Executes fast, memory-efficient LoRA SFT and DPO alignment on Modal.com
using an NVIDIA A100 GPU (40GB/80GB).

Key Highlights:
1. Powered by Unsloth FastLanguageModel (2x faster, 70% less VRAM with 4-bit QLoRA).
2. Two-Stage Adaptation:
   - Stage 1 (SFT): Fine-tunes Qwen2.5-Coder-7B on clean, verified successful trajectories.
   - Stage 2 (DPO): Aligns model using contrastive preference pairs (Clean/Complex vs Degraded/Minimal).
3. Checkpoint Persistence:
   Saves merged/standalone LoRA adapter weights directly to a persistent Modal Volume:
   `/vol/checkpoints/helix_lora_final`.
4. Closes the Loop:
   Once committed, `01_modal_vllm.py` automatically detects and mounts these weights
   to initiate the REBUILD evaluation phase.

Usage:
  # Run on Modal cloud GPU:
  modal run closed-loop/scripts/04_train_lora.py --stage sft_then_dpo --epochs 3
"""

import os
from pathlib import Path
import modal

# ------------------------------------------------------------------------------
# 1. Modal App & Persistent Volume Configuration
# ------------------------------------------------------------------------------
APP_NAME = "helix-lora-training"
VOLUME_NAME = "helix-checkpoints-volume"
MOUNT_PATH = "/vol/checkpoints"
BASE_MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# Container image optimized with Unsloth and HuggingFace TRL stack
training_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.4.0",
        "transformers>=4.45.0",
        "datasets>=3.0.0",
        "accelerate>=0.34.0",
        "peft>=0.13.0",
        "trl>=0.11.0",
        "bitsandbytes>=0.44.0",
        "unsloth>=2024.9",
        "pyyaml>=6.0.2",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# Mount local processed data into container
LOCAL_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
data_mount = modal.Mount.from_local_dir(LOCAL_DATA_DIR, remote_path="/data/processed")


# ------------------------------------------------------------------------------
# 2. Remote Training Function on A100 GPU
# ------------------------------------------------------------------------------
@app.function(
    image=training_image,
    gpu=modal.gpu.A100(size="40GB"),
    volumes={MOUNT_PATH: volume},
    mounts=[data_mount],
    timeout=7200,  # 2 hours max
)
def train_lora_pipeline(
    stage: str = "sft_then_dpo",
    epochs: int = 3,
    lr: float = 2.0e-4,
    dpo_lr: float = 5.0e-6,
    r: int = 16,
    lora_alpha: int = 32,
):
    import json
    import torch
    from datasets import Dataset
    from transformers import TrainingArguments
    from trl import SFTTrainer, DPOTrainer, DPOConfig
    from unsloth import FastLanguageModel, is_bfloat16_supported

    print("=" * 70)
    print("STARTING HELIX CLOSED-LOOP LORA FINE-TUNING ON MODAL A100")
    print(f"Base Model: {BASE_MODEL_NAME} | Stage: {stage} | LoRA r={r}, alpha={lora_alpha}")
    print("=" * 70)

    # 1. Initialize Base Model & Tokenizer with Unsloth 4-bit optimization
    max_seq_length = 4096
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL_NAME,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        dtype=torch.bfloat16 if is_bfloat16_supported() else torch.float16,
    )

    # 2. Attach LoRA Adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # Paths
    sft_file = Path("/data/processed/sft_data.jsonl")
    dpo_file = Path("/data/processed/dpo_pairs.jsonl")
    final_adapter_dir = os.path.join(MOUNT_PATH, "helix_lora_final")

    # --------------------------------------------------------------------------
    # Stage 1: Supervised Fine-Tuning (SFT)
    # --------------------------------------------------------------------------
    if stage in ["sft", "sft_then_dpo"] and sft_file.exists():
        print(f"\n[Stage 1: SFT] Loading training trajectories from {sft_file}...")
        sft_samples = []
        with open(sft_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    # Convert messages into tokenizer chat template string
                    prompt_text = tokenizer.apply_chat_template(
                        rec["messages"], tokenize=False, add_generation_prompt=False
                    )
                    sft_samples.append({"text": prompt_text})

        print(f"[Stage 1: SFT] Loaded {len(sft_samples)} samples. Beginning SFTTrainer...")
        sft_dataset = Dataset.from_list(sft_samples)

        sft_training_args = TrainingArguments(
            output_dir=os.path.join(MOUNT_PATH, "sft_checkpoints"),
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_ratio=0.1,
            num_train_epochs=epochs,
            learning_rate=lr,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=5,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=42,
            save_strategy="no",
        )

        sft_trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=sft_dataset,
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            args=sft_training_args,
        )
        sft_trainer.train()
        print("[Stage 1: SFT] Training complete!")

    # --------------------------------------------------------------------------
    # Stage 2: Direct Preference Optimization (DPO)
    # --------------------------------------------------------------------------
    if stage in ["dpo", "sft_then_dpo"] and dpo_file.exists():
        print(f"\n[Stage 2: DPO] Loading contrastive pairs from {dpo_file}...")
        dpo_samples = []
        with open(dpo_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    dpo_samples.append({
                        "prompt": f"<|im_start|>system\n{rec['system']}<|im_end|>\n<|im_start|>user\n{rec['prompt']}<|im_end|>\n<|im_start|>assistant\n",
                        "chosen": f"{rec['chosen']}<|im_end|>",
                        "rejected": f"{rec['rejected']}<|im_end|>",
                    })

        print(f"[Stage 2: DPO] Loaded {len(dpo_samples)} preference pairs. Beginning DPOTrainer...")
        dpo_dataset = Dataset.from_list(dpo_samples)

        dpo_config = DPOConfig(
            output_dir=os.path.join(MOUNT_PATH, "dpo_checkpoints"),
            beta=0.1,
            learning_rate=dpo_lr,
            num_train_epochs=2,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=5,
            optim="adamw_8bit",
            seed=42,
            save_strategy="no",
        )

        dpo_trainer = DPOTrainer(
            model=model,
            ref_model=None,  # Implicit reference model in PEFT/Unsloth
            tokenizer=tokenizer,
            train_dataset=dpo_dataset,
            args=dpo_config,
            max_prompt_length=2048,
            max_length=max_seq_length,
        )
        dpo_trainer.train()
        print("[Stage 2: DPO] Alignment complete!")

    # --------------------------------------------------------------------------
    # 3. Save Final Adapter Weights to Persistent Modal Volume
    # --------------------------------------------------------------------------
    print(f"\n[Persisting Weights] Saving final LoRA adapter to {final_adapter_dir}...")
    os.makedirs(final_adapter_dir, exist_ok=True)
    model.save_pretrained(final_adapter_dir)
    tokenizer.save_pretrained(final_adapter_dir)

    # Commit changes to Modal Volume so 01_modal_vllm.py immediately detects them
    volume.commit()
    print("[✓] Modal Volume committed successfully!")
    print(f"[✓] Checkpoint ready for REBUILD serving at: {final_adapter_dir}")


# ------------------------------------------------------------------------------
# 4. Local Entrypoint
# ------------------------------------------------------------------------------
@app.local_entrypoint()
def main(stage: str = "sft_then_dpo", epochs: int = 3, lr: float = 2.0e-4):
    print("Triggering remote LoRA training on Modal A100 GPU...")
    train_lora_pipeline.remote(
        stage=stage,
        epochs=epochs,
        lr=lr,
    )
    print("Training job finished.")

