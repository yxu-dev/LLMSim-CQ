#!/bin/bash
# Baseline 测试（无量化）- WinoGrande

python -m lm_eval.run_models --model hf \
    --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct,attn_implementation=eager \
    --tasks winogrande \
    --apply_chat_template \
    --fewshot_as_multiturn \
    --batch_size 1 \
    --device cuda:0 \
    --output_path results/llama-3.1-8b/baseline_winogrande_full.json






