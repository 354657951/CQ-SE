# Download models from HuggingFace Hub for cross-query SE experiments.
# Models: Qwen2.5-7B-Instruct, bge-large-en-v1.5, deberta-v2-xlarge-mnli
import os
import sys
from huggingface_hub import snapshot_download

HF_HOME = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
print(f"HF_HOME: {HF_HOME}")

models = [
    "Qwen/Qwen2.5-7B-Instruct",
    "BAAI/bge-large-en-v1.5",
    "microsoft/deberta-v2-xlarge-mnli",
]

if len(sys.argv) > 1:
    models = sys.argv[1:]

for model_id in models:
    print(f"\nDownloading {model_id} ...")
    local_dir = snapshot_download(
        repo_id=model_id,
        cache_dir=os.path.join(HF_HOME, "hub"),
    )
    print(f"  -> {local_dir}")

print("\nAll downloads complete.")
