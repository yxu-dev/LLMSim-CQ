python -m lm_eval.run_models --model hf \
    --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct,cq_codebook_dir=/home/yx277/workspace-vq/fisher_weighted_codebook/llama-3.1-8b/4c8b \
    --tasks wikitext \
    --apply_chat_template \
    --fewshot_as_multiturn \
    --batch_size 1 \
    --device cuda:6 \
    --simi \
    --output_path results/llama-3.1-8b/cq_4c8b_wikitext_full.json