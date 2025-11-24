lm_eval --model hf \
    --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct,dtype=bfloat16,parallelize=True \
    --tasks gsm8k \
    --apply_chat_template \
    --fewshot_as_multiturn \
    --batch_size 16 \
    --limit 10 \
    # --device cuda:0 \