#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=5
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-output/llama-3.1-8b-4c8b}"

cd "${PROJECT_ROOT}"

python export_kv_and_fisher.py \
  --model "meta-llama/Meta-Llama-3.1-8B" \
  --output_dir "${OUTPUT_DIR}" \
  --num_samples 16 \
  --max_seq_len 2048 \
  --key_export_domain pre_rope \
  --num_coupled_channels 4 \
  --num_bits 8 \
  --dataset "wikitext" \
  --dataset_config "wikitext-2-raw-v1"