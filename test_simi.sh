python -m lm_eval.run_models --model hf \
    --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct,attn_implementation=eager \
    --tasks gsm8k \
    --apply_chat_template \
    --fewshot_as_multiturn \
    --batch_size 1 \
    --limit 3 \
    --device cuda:0 \
    --simi \
    --extract_hooks \
    # --log_samples \
    # --output_path output/llama3/gsm8k