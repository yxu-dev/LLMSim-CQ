import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
import os
from lm_eval.similarity.utils import get_attr_by_name
from lm_eval.similarity.models.llama3_model import repeat_kv
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- 1. Configuration ---
# All the key parameters are defined here for easy modification.
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
TARGET_LAYER_IDX = 9  # This is the 10th layer (zero-indexed).

ACTIVATIONS_FILE = "output/extracted_data/meta-llama_Llama-3.1-8B-Instruct_activations.pt"
ATTN_WEIGHTS_FILE = "output/extracted_data/meta-llama_Llama-3.1-8B-Instruct_attn_weights.pt"

# --- 2. Load Data from Files ---
print(f"Loading data from files...")
if not (os.path.exists(ACTIVATIONS_FILE) and os.path.exists(ATTN_WEIGHTS_FILE)):
    raise FileNotFoundError("One or both data files not found. Please check the paths.")

# Load the dictionaries containing activations, weights, and attention scores
activations_data = torch.load(ACTIVATIONS_FILE, map_location='cpu')

attn_weights_data = torch.load(ATTN_WEIGHTS_FILE, map_location='cpu')
print("Data files loaded successfully.")


# --- 3. Load Model from Hugging Face to get fresh weights ---
print(f"Loading model '{MODEL_ID}' to get weights...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="cpu"  # Load to CPU to save GPU memory; we only need weights.
)
print("Model loaded successfully.")


# --- 4. Extract All Required Values ---
print(f"\nExtracting all requested values for Layer {TARGET_LAYER_IDX}...")
extracted_values = {}

# Define the linear layers we want to inspect in the target layer
linear_modules = {
    "self_attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "mlp": ["gate_proj", "up_proj", "down_proj"]
}

# --- Task A & B: Get Prefill Inputs and Weights for ALL Linear Layers ---
for block_name, modules in linear_modules.items():
    for module_name in modules:
        # Construct the base key for the module
        input_key = f"model.layers.{TARGET_LAYER_IDX}.{block_name}.{module_name}"

        if input_key in activations_data:
            extracted_values[f"input_{module_name}"] = activations_data[input_key][0]
        
        try:
            extracted_values[f"weight_{module_name}"] = get_attr_by_name(model, input_key).weight.detach().cpu()
        except AttributeError:
            print(f"Warning: Weight key '{input_key}' not found in model.")

# # --- Task C: Get Prefill Q, K, V values (outputs of projection layers) ---
# qkv_modules = ["q_proj", "k_proj", "v_proj"]
# for module_name in qkv_modules:
#     output_key = f"model.layers.{TARGET_LAYER_IDX}.self_attn.{module_name}_out"
#     if output_key in activations_data:
#         # The variable name Q, K, or V is the first letter of the module name
#         tensor_name = module_name[0]
#         extracted_values[f"{tensor_name}"] = activations_data[output_key][0]
        
# # --- Task D: Get Prefill Attention Score ---
# attn_score_key = f'model.layers.{TARGET_LAYER_IDX}.self_attn'
# if attn_score_key in attn_weights_data:
#     extracted_values["attn_score"] = attn_weights_data[attn_score_key][0]

# print("Extraction complete.")


# --- 5. Verification ---
# Print the keys and shapes of everything we just extracted.
print("\n--- Summary of Extracted Data ---")
for name, tensor in extracted_values.items():
    if tensor is not None:
        print(f"  - {name}: shape={tensor.shape}, dtype={tensor.dtype}")
    else:
        print(f"  - {name}: Not found.")
        
print("\nScript finished. All data is in the `extracted_values` dictionary.")

# --- 6. Save Extracted Values ---
OUTPUT_FILE = f"output/extracted_data/extracted_values_layer_{TARGET_LAYER_IDX}.pt"
torch.save(extracted_values, OUTPUT_FILE)
print(f"Extracted values saved to '{OUTPUT_FILE}'.")