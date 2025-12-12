#!/bin/bash

# FP baseline (no CQ) for winogrande
# 不传递 cq_codebook_dir 参数即为 FP 基线
python -m lm_eval.run_models --model hf \
    --model_args pretrained=meta-llama/Llama-3.1-8B \
    --tasks winogrande \
    --batch_size 1 \
    --device cuda:1 \
    --output_path results/llama-3.1-8b/fp_winogrande_full.json

