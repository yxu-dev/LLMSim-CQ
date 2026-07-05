# LLMSim-CQ

This repository is an engineering reproduction of the NeurIPS 2024 paper. It is provided on a best-effort basis, and the reproduced implementation, pipeline, and results may be incomplete, imperfect, or not fully correct relative to the original work:

KV Cache is 1 Bit Per Channel: Efficient Large Language Model Inference with Coupled Quantization  
Tianyi Zhang, Jonah Yi, Zhaozhuo Xu, Anshumali Shrivastava  
Paper URL: https://proceedings.neurips.cc/paper_files/paper/2024/file/05d6b5b6901fb57d2c287e1d3ce6d63c-Paper-Conference.pdf

At the time of writing, this project assumes there is no official public code release linked from the paper page, so this repo provides an end-to-end practical reproduction pipeline.

---

## What This Repo Reproduces

- Coupled Quantization (CQ) for KV-cache compression
- RTN per-tensor baseline for comparison
- FP (full precision) baseline for comparison
- End-to-end runs on real evaluation workloads (for example Winogrande and perplexity)

Main target model family in current scripts:

- meta-llama/Meta-Llama-3.1-8B (export stage)
- meta-llama/Llama-3.1-8B (evaluation stage)

---

## Current Default Configuration In Scripts

The checked-in shell scripts are currently configured for 4c8b runs by default:

- num_coupled_channels = 4
- num_bits = 8
- key_export_domain = pre_rope
- export/data root default = output/llama-3.1-8b-4c8b

If you want different settings (for example 2c4b), change the script arguments or override environment variables described below.

---

## Environment Setup

### 1) Enter the project

```bash
cd LLMSim-CQ
```

### 2) Create and activate a Conda environment

```bash
conda create -n vq python=3.10 -y
conda activate vq
```

### 3) Install dependencies

```bash
pip install -U pip setuptools wheel
pip install -e .
pip install torch transformers datasets accelerate sentencepiece
```

Notes:

- pip install -e . installs project dependencies from pyproject.toml.
- Extra installs above ensure runtime packages required by the reproduction scripts are present.

### 4) Optional Slurm interactive session

```bash
srun -p athena-genai -t 24:00:00 -w node5 --pty bash
```

---

## Environment Check (Recommended Before Long Runs)

Run these checks once before expensive jobs:

```bash
which python
python --version
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -c "import transformers, datasets, accelerate; print('imports ok')"
python -c "import lm_eval; print('lm_eval import ok')"
nvidia-smi
```

Expected outcome:

- Python is from your vq environment.
- CUDA is available if you plan to run GPU experiments.
- All core imports succeed.

---

## Reproduction Workflow

Use this sequence:

1. Export KV activations + Fisher diagonal
2. Learn Fisher-weighted centroids per layer
3. Run CQ evaluation
4. Run FP baseline
5. (Optional) Run RTN baseline

### Step 1) Export KV + Fisher

```bash
bash run_export_kv_and_fisher_llama-3.1-8b.sh
```

Current script defaults:

- output_dir: output/llama-3.1-8b-4c8b
- dataset: wikitext / wikitext-2-raw-v1
- num_samples: 16
- max_seq_len: 2048
- key_export_domain: pre_rope

Produced artifacts:

- output/.../kv_cache/sample*_layer*_key.pt
- output/.../kv_cache/sample*_layer*_value.pt
- output/.../fisher_diag.pt

### Step 2) Generate centroids

```bash
bash run_generate_centroids_llama-3.1-8b.sh
```

This loops over all 32 layers and writes:

- k_centroids_fisher_layer{i}.npy
- v_centroids_fisher_layer{i}.npy

Default save path:

- output/llama-3.1-8b-4c8b/centroids

### Step 3) CQ evaluation (Winogrande)

```bash
bash test_cq_llama-3.1-8b_optimized.sh
```

Check logs for:

- Enabled CQ KV-cache quantization

Default outputs:

- results/llama-3.1-8b/cq_4c8b_winogrande_optimized.json
- cq_test_optimized_log.txt

### Step 4) FP baseline

```bash
bash test_baseline_winogrande_llama3.1-8b.sh
```

Default outputs:

- result/llama-3.1-8b/baseline_winogrande_optimized.json
- baseline_test_optimized_log.txt

### Step 5) RTN per-tensor baseline (optional)

```bash
bash test_rtn_winogrande_llama3.1-8b.sh
```

This run passes rtn_pertensor_bits=4 via model_args.

Default outputs:

- result/llama-3.1-8b/rtn_pertensor_winogrande.json
- result/llama-3.1-8b/rtn_pertensor_winogrande.log

---

## Optional: Perplexity Comparison Using scripts/run_cq_eval.py

CQ run:

```bash
python scripts/run_cq_eval.py \
  --model meta-llama/Llama-3.1-8B \
  --codebook-dir output/llama-3.1-8b-4c8b/centroids \
  --dataset wikitext \
  --dataset-config wikitext-2-raw-v1 \
  --limit 1 \
  --max-eval-tokens 131072
```

FP baseline run (disable CQ):

```bash
python scripts/run_cq_eval.py \
  --model meta-llama/Llama-3.1-8B \
  --codebook-dir output/llama-3.1-8b-4c8b/centroids \
  --disable-cq \
  --dataset wikitext \
  --dataset-config wikitext-2-raw-v1 \
  --limit 1 \
  --max-eval-tokens 131072
```

---

## Path and Runtime Notes

- Some scripts write to result/... while others write to results/.... Both directories are used in this repo.
- GPU selection is hardcoded in several scripts (for example cuda:5 or cuda:3). Update them to match your machine.
- You can override these environment variables where supported:
  - OUTPUT_DIR for export
  - DATA_ROOT for centroid generation
  - CODEBOOK_DIR for CQ evaluation
  - RESULT_DIR for RTN evaluation

---

## Minimal End-to-End Quickstart

```bash
bash run_export_kv_and_fisher_llama-3.1-8b.sh
bash run_generate_centroids_llama-3.1-8b.sh
bash test_cq_llama-3.1-8b_optimized.sh
bash test_baseline_winogrande_llama3.1-8b.sh
```

---

## Models Referenced In This Repo

1. google/gemma-3-12b-it
2. meta-llama/Llama-3.1-8B-Instruct
3. Qwen/Qwen3-4B-Instruct-2507
4. Qwen/Qwen3-4B-Thinking-2507
5. openai/gpt-oss-20b
