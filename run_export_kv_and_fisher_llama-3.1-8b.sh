#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=7
set -euo pipefail

OUTPUT_DIR="/home/zz359/workspace-CQ-zzy/LLMSim-CQ-zzy/output/llama-3.1-8b-2c4b"

python export_kv_and_fisher.py \
  --model "meta-llama/Meta-Llama-3.1-8B" \
  --output_dir "${OUTPUT_DIR}" \
  --num_samples 16 \
  --max_seq_len 2048 \
  --key_export_domain post_rope \
  --num_coupled_channels 2 \
  --num_bits 4 \
  --dataset "wikitext" \
  --dataset_config "wikitext-2-raw-v1"