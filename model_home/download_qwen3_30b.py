import os

from huggingface_hub import snapshot_download

repo_id = "Qwen/Qwen3-30B-A3B"
repo_type = "model"
local_dir = "model_home/qwen3-30b-a3b"

# Tuning knobs:
# - HF_MAX_WORKERS: control concurrent file downloads (default 16)
# - HF_ENDPOINT: set a mirror endpoint when needed
# - HF_HUB_ENABLE_HF_TRANSFER=1: use hf_transfer accelerator if installed
max_workers = int(os.getenv("HF_MAX_WORKERS", "16"))
hf_endpoint = os.getenv("HF_ENDPOINT")

if hf_endpoint:
    os.environ["HF_ENDPOINT"] = hf_endpoint

# Prefer downloading only files needed for inference/training.
allow_patterns = [
    "*.safetensors",
    "*.json",
    "*.py",
    "*.model",
    "*.txt",
    "tokenizer*",
]
ignore_patterns = [
    "*.bin",
    "*.pt",
    "*.pth",
    "*.onnx",
]

print(f"[download] repo_id={repo_id}")
print(f"[download] local_dir={local_dir}")
print(f"[download] max_workers={max_workers}")
print(f"[download] HF_ENDPOINT={os.getenv('HF_ENDPOINT')}")
print(f"[download] HF_HUB_ENABLE_HF_TRANSFER={os.getenv('HF_HUB_ENABLE_HF_TRANSFER')}")

snapshot_download(
    repo_id=repo_id,
    repo_type=repo_type,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    allow_patterns=allow_patterns,
    ignore_patterns=ignore_patterns,
    resume_download=True,
    max_workers=max_workers,
)
