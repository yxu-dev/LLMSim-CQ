step1: export_kv.py  run_export_kv_llama-3.1-8b.sh
step2: collect_Fisher_gradients.py  fisher_gradients_llama-3.1-8b.sh
step3: run_weighted_kmeans.py  workspace-vq/LLMSim-CQ/generate_all_fisher_codebooks_llama-3.1-8b.sh