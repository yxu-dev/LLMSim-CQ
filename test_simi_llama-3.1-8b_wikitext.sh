python -m lm_eval.run_models --model hf \
    --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct \
    --tasks wikitext \
    --apply_chat_template \
    --fewshot_as_multiturn \
    --batch_size 1 \
    --device cuda:4 \
    --simi \
    --output_path results/llama-3.1-8b/baseline_wikitext_full.json