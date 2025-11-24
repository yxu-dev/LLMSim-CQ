python -u collect_Fisher_gradients.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --calib_txt /home/yx277/workspace-vq/LLMSim/output/kv-simi-llama-3.1-8b-wikitext/prefill_long_sequences.txt \
  --max_seq_len 2048 \
  --num_samples 16 \
  --batch_size 1 \
  --dtype bfloat16 \
  --num_coupled_channels 4 \
  --num_bits 8 \
  --save_dir fisher_runs_output/run_llama-3.1-8b-4c8b