#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=5
set -euo pipefail

OUTPUT_DIR="/home/yx277/workspace-vq/LLMSim-CQ/output/llama-3.1-8b"

python export_kv_and_fisher.py \
  --model "meta-llama/Meta-Llama-3.1-8B" \
  --output_dir "${OUTPUT_DIR}" \
  --num_samples 16 \
  --max_seq_len 2048 \
  --num_coupled_channels 4 \
  --num_bits 8 \
  --dataset "wikitext" \
  --dataset_config "wikitext-2-raw-v1"