#!/bin/bash
# Download Qwen2.5-72B-Instruct from HuggingFace Hub
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="${EXP_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
WORKSPACE_DIR="$(dirname "$EXP_DIR")"

source "$EXP_DIR/.venv/bin/activate"

export HF_HUB_DISABLE_XET=1
# Use HF_HOME from environment if set, otherwise derive from workspace layout
export HF_HOME="${HF_HOME:-$WORKSPACE_DIR/hf_cache}"

echo "HF_HOME: $HF_HOME"
echo "Starting download of Qwen2.5-72B-Instruct..."

python3 -c "
import os
from huggingface_hub import snapshot_download
hf_home = os.environ['HF_HOME']
local_dir = snapshot_download(
    repo_id='Qwen/Qwen2.5-72B-Instruct',
    cache_dir=os.path.join(hf_home, 'hub'),
)
print(f'Downloaded to: {local_dir}')
"

echo "Download complete."
