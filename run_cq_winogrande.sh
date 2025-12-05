#!/bin/bash
# CQ 量化测试 - WinoGrande

python -m lm_eval.run_models --model hf \
    --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct,attn_implementation=eager,cq_codebook_dir=/home/zz359/workspace-vq/fisher_weighted_codebook/llama-3.1-8b/4c8b \
    --tasks winogrande \
    --apply_chat_template \
    --fewshot_as_multiturn \
    --batch_size 1 \
    --device cuda:5 \
    --output_path results/llama-3.1-8b/cq_4c8b_winogrande_full.json

