python -m lm_eval --model hf \
    --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct \
    --tasks gsm8k \
    --apply_chat_template \
    --fewshot_as_multiturn \
    --batch_size 1 \
    --limit 10 \
    --device cuda:0 \
    # --log_samples \
    # --output_path output/llama3/gsm8k