# LLMSim
1. google/gemma-3-12b-it
2. meta-llama/Llama-3.1-8B-Instruct
3. Qwen/Qwen3-4B-Instruct-2507
4. Qwen/Qwen3-4B-Thinking-2507
5. openai/gpt-oss-20b

Run on slurm with:
```bash
srun -p athena-genai -t 24:00:00 -w node5 --pty bash
```

## CQ KV-cache quantization workflow

1. Collect activations and Fisher diagonals (see `generate_all_fisher_codebooks_*.sh`).
2. Train per-layer Fisher-weighted CQ codebooks with `run_weighted_kmeans.py` (already automated in the helper script).
3. Enable the runtime patch and evaluate perplexity / downstream metrics with the new helper:

```bash
python scripts/run_cq_eval.py \
	--model meta-llama/Llama-3.1-8B-Instruct \
	--codebook-dir /path/to/fisher_weighted_codebook/llama-3.1-8b/4c8b \
	--dataset wikitext --dataset-config wikitext-2-raw-v1 \
	--limit 1 \
	--max-eval-tokens 131072
```

Add `--disable-cq` to obtain the FP baseline for comparison. The script exercises the quantized KV-cache during loss computation by forcing `use_cache=True`, matching the coupled quantization paper setup.

### Using CQ within `lm_eval.run_models`

`lm_eval` will now enable the CQ cache whenever `cq_codebook_dir` is supplied via `--model_args` (omit it for FP baselines):

```bash
python -m lm_eval.run_models --model hf \
	--model_args pretrained=meta-llama/Llama-3.1-8B-Instruct,attn_implementation=eager,cq_codebook_dir=/path/to/fisher_weighted_codebook/llama-3.1-8b/4c8b \
	--tasks wikitext \
	--device cuda:6 \
	--limit 10
```

Use `cq_layer_prefix` if your codebook filenames deviate from the default `k_centroids_fisher_layer{i}.npy` pattern.