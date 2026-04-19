#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage:"
  echo "  $0 <model_id> <pre_rope_codebook_dir> <post_rope_codebook_dir> [task] [device] [batch_size]"
  echo ""
  echo "Example:"
  echo "  $0 meta-llama/Llama-3.1-8B \\"
  echo "     /path/to/pre_rope/centroids \\"
  echo "     /path/to/post_rope/centroids \\"
  echo "     winogrande cuda:0 auto"
  exit 1
fi

MODEL_ID="$1"
PRE_CB="$2"
POST_CB="$3"
TASK="${4:-winogrande}"
DEVICE="${5:-cuda:0}"
BATCH_SIZE="${6:-auto}"

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="${ROOT_DIR}/results/prepost_rope_cmp"
mkdir -p "${OUT_DIR}"

TS="$(date +%Y%m%d_%H%M%S)"
PRE_JSON="${OUT_DIR}/pre_rope_${TASK}_${TS}.json"
POST_JSON="${OUT_DIR}/post_rope_${TASK}_${TS}.json"

echo "================================================"
echo "CQ pre/post RoPE comparison"
echo "Model : ${MODEL_ID}"
echo "Task  : ${TASK}"
echo "Device: ${DEVICE}"
echo "================================================"

run_eval() {
  local codebook_dir="$1"
  local out_json="$2"
  local tag="$3"
  echo ""
  echo "[${tag}] codebook_dir=${codebook_dir}"
  python -m lm_eval.run_models \
    --model hf \
    --model_args "pretrained=${MODEL_ID},cq_codebook_dir=${codebook_dir},attn_implementation=eager" \
    --tasks "${TASK}" \
    --batch_size "${BATCH_SIZE}" \
    --device "${DEVICE}" \
    --verbosity INFO \
    --output_path "${out_json}"
}

run_eval "${PRE_CB}" "${PRE_JSON}" "pre_rope"
run_eval "${POST_CB}" "${POST_JSON}" "post_rope"

python - <<PY
import json
from pathlib import Path

pre_path = Path(r"${PRE_JSON}")
post_path = Path(r"${POST_JSON}")
task = r"${TASK}"

def load_metric(path: Path):
    data = json.loads(path.read_text())
    task_result = data.get("results", {}).get(task, {})
    if "acc,none" in task_result:
        return "acc,none", task_result["acc,none"]
    for key in task_result:
        if key.startswith("acc"):
            return key, task_result[key]
    # fallback: first numeric metric
    for key, value in task_result.items():
        if isinstance(value, (int, float)):
            return key, value
    raise RuntimeError(f"Cannot find metric in {path}")

pre_key, pre_val = load_metric(pre_path)
post_key, post_val = load_metric(post_path)

print("\\n================ Comparison ================")
print(f"pre_rope  ({pre_key}): {pre_val:.6f}")
print(f"post_rope ({post_key}): {post_val:.6f}")
print(f"delta(post - pre): {post_val - pre_val:+.6f}")
print("===========================================")
print(f"pre result : {pre_path}")
print(f"post result: {post_path}")
PY
