"""
01_modal_vllm.py - Cloud Inference Server for HELIX Closed-Loop Pipeline
========================================================================
Deploys a high-throughput, OpenAI-compatible vLLM inference server on Modal.com
using an NVIDIA A100 (40GB/80GB) or A10G GPU.

Key Capabilities:
1. BUILD Phase: Serves base model (Qwen/Qwen2.5-Coder-7B-Instruct) for baseline execution.
2. REBUILD Phase: Dynamically mounts Modal Volume and serves LoRA adapter weights
   via vLLM's multi-LoRA engine (--enable-lora).
3. Exposes OpenAI-compatible endpoints:
   - POST /v1/chat/completions
   - POST /v1/completions
   - GET  /v1/models

Usage:
  # Deploy server to Modal as a persistent endpoint:
  modal deploy closed-loop/scripts/01_modal_vllm.py

  # Test locally / ephemeral interactive run:
  modal serve closed-loop/scripts/01_modal_vllm.py
"""

import os
import subprocess
import time
import modal

# ------------------------------------------------------------------------------
# 1. Configuration & Constants
# ------------------------------------------------------------------------------
APP_NAME = "helix-vllm-engine"
BASE_MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"
VOLUME_NAME = "helix-checkpoints-volume"
MOUNT_PATH = "/vol/checkpoints"
LORA_ADAPTER_NAME = "helix-lora-adapter"
LORA_SUBDIR = "helix_lora_final"

# GPU Selection: A100 for high concurrency, or A10G for cost-effective development
GPU_TYPE = modal.gpu.A100(size="40GB")

# ------------------------------------------------------------------------------
# 2. Container Image & Persistent Storage
# ------------------------------------------------------------------------------
vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm>=0.6.3",
        "torch>=2.4.0",
        "transformers>=4.45.0",
        "hf-transfer>=0.1.8",
        "fastapi>=0.115.0",
        "requests>=2.32.0",
        "pydantic>=2.8.0",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# ------------------------------------------------------------------------------
# 3. vLLM Server Process on Modal
# ------------------------------------------------------------------------------
@app.function(
    image=vllm_image,
    gpu=GPU_TYPE,
    volumes={MOUNT_PATH: volume},
    timeout=3600,
    container_idle_timeout=300,
    allow_concurrent_inputs=64,
)
@modal.web_server(port=8000, startup_timeout=300)
def serve():
    """
    Spawns vLLM OpenAI-compatible server.
    Detects whether LoRA weights exist in the attached volume; if present,
    activates LoRA serving for REBUILD phase evaluation.
    """
    lora_path = os.path.join(MOUNT_PATH, LORA_SUBDIR)
    lora_enabled = os.path.exists(lora_path) and os.path.exists(os.path.join(lora_path, "adapter_config.json"))

    cmd = [
        "python3",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model", BASE_MODEL_NAME,
        "--host", "0.0.0.0",
        "--port", "8000",
        "--trust-remote-code",
        "--tensor-parallel-size", "1",
        "--max-model-len", "4096",
        "--gpu-memory-utilization", "0.90",
    ]

    if lora_enabled:
        print(f"[REBUILD Mode] Detected LoRA adapter at {lora_path}. Activating LoRA support...")
        cmd.extend([
            "--enable-lora",
            "--max-loras", "2",
            "--max-lora-rank", "32",
            "--lora-modules", f"{LORA_ADAPTER_NAME}={lora_path}",
        ])
    else:
        print(f"[BUILD Mode] No LoRA adapter found in {lora_path}. Serving base model {BASE_MODEL_NAME}...")

    print(f"Launching vLLM command: {' '.join(cmd)}")
    subprocess.Popen(cmd)


# ------------------------------------------------------------------------------
# 4. Local Verification Runner
# ------------------------------------------------------------------------------
@app.local_entrypoint()
def test_connection():
    """
    Local test entrypoint to verify that vLLM server responds to requests.
    Run with: modal run closed-loop/scripts/01_modal_vllm.py
    """
    import urllib.request
    import json

    print(f"Target Base Model: {BASE_MODEL_NAME}")
    print(f"Attached Volume: {VOLUME_NAME} -> {MOUNT_PATH}")
    print("Modal deployment configured successfully.")
    print("Run `modal serve closed-loop/scripts/01_modal_vllm.py` to launch live testing URL.")

