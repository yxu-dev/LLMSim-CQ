import torch
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- 1. Configuration ---
TARGET_LAYER_IDX = 9
# --- Point to the file saved by the first script ---
CONSOLIDATED_FILE = f"output/extracted_data/extracted_values_layer_{TARGET_LAYER_IDX}.pt"


# --- 2. Load PRE-EXTRACTED Data from File ---
print(f"Loading consolidated data from '{CONSOLIDATED_FILE}'...")
if not os.path.exists(CONSOLIDATED_FILE):
    raise FileNotFoundError(f"Data file not found. Please run 'extract_and_save.py' first.")

# This is much faster than loading the whole model
extracted_values = torch.load(CONSOLIDATED_FILE, map_location='cpu')
print("Consolidated data loaded successfully.")


# --- 3. Verification ---
print("\n--- Summary of Loaded Data ---")
for name, tensor in extracted_values.items():
    if tensor is not None:
        print(f"  - {name}: shape={tensor.shape}, dtype={tensor.dtype}")
    else:
        print(f"  - {name}: Not found.")


# ==============================================================================
# --- 4. Tiled Similarity Analysis & Visualization ---
# (This part is identical to the previous analysis code)
# ==============================================================================

print("\n--- Starting Tiled Similarity Analysis ---")

# --- 4a. Analysis Configuration ---
ANALYSIS_CONFIG = {
    "tile_m": 64,
    "tile_k": 64,
    "tile_n": 64,
    "output_dir": "output/similarity_analysis",
    # --- NEW: Parameters to select a specific tile ---
    "tile_row_idx": 0,  # 0 = first tile, 1 = second tile, etc.
    "tile_col_idx": 2   # 0 = first tile, 1 = second tile, etc.
}
os.makedirs(ANALYSIS_CONFIG["output_dir"], exist_ok=True)
print(f"Configuration: tile_m={ANALYSIS_CONFIG['tile_m']}, tile_k={ANALYSIS_CONFIG['tile_k']}, tile_n={ANALYSIS_CONFIG['tile_n']}")
print(f"Plots will be saved to: {ANALYSIS_CONFIG['output_dir']}\n")

# --- 4b. Helper Functions ---
def calculate_and_plot_similarity(tensor, analysis_type, name, config):
    if tensor is None or tensor.numel() == 0:
        print(f"Skipping {name} - {analysis_type} similarity, tensor is empty.")
        return

    tensor = tensor.to(torch.float32)

    if analysis_type == 'row':
        sim_vectors = tensor
        dim_label, num_vectors = "Rows", sim_vectors.shape[0]
        title = f'Row Similarity for {name}\n(Comparing {num_vectors} vectors of size {sim_vectors.shape[1]})'
    elif analysis_type == 'col':
        sim_vectors = tensor.T
        dim_label, num_vectors = "Columns", sim_vectors.shape[0]
        title = f'Column Similarity for {name}\n(Comparing {num_vectors} vectors of size {sim_vectors.shape[1]})'
    else:
        return

    # sim_vectors_norm = F.normalize(sim_vectors, p=2, dim=1)
    # sim_matrix = torch.matmul(sim_vectors_norm, sim_vectors_norm.T).cpu().numpy()
    cos = torch.nn.CosineSimilarity(dim=2)
    sim_matrix = cos(sim_vectors.unsqueeze(1), sim_vectors.unsqueeze(0)).cpu().numpy()
    
    avg_sim = np.mean(sim_matrix[~np.eye(num_vectors, dtype=bool)])
    print(f"  - Analyzing {name} [{dim_label}]: Avg similarity = {avg_sim:.4f}")
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(sim_matrix, cmap='viridis', vmin=-1, vmax=1)
    plt.title(title)
    plt.xlabel(f"Vector Index")
    plt.ylabel(f"Vector Index")
    
    filename = f"{name}_{analysis_type}_similarity.png"
    output_path = os.path.join(config["output_dir"], filename)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

# --- 4c. Main Analysis Loop ---
layers_to_analyze = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
# layers_to_analyze = ["q_proj"]
tile_m, tile_k, tile_n = ANALYSIS_CONFIG["tile_m"], ANALYSIS_CONFIG["tile_k"], ANALYSIS_CONFIG["tile_n"]
row_idx, col_idx = ANALYSIS_CONFIG["tile_row_idx"], ANALYSIS_CONFIG["tile_col_idx"]

for layer_name in layers_to_analyze:
    input_tensor = extracted_values.get(f"input_{layer_name}")
    weight_tensor = extracted_values.get(f"weight_{layer_name}")

    if input_tensor is None and weight_tensor is None:
        continue

    print(f"--- Analyzing Layer: {layer_name} ---")

    # --- A. Analyze Input Activation Tensor ---
    if input_tensor is not None:
        input_tensor_2d = input_tensor.squeeze(0)
        
        # Calculate the slice boundaries for the selected tile
        start_row = row_idx * tile_m
        end_row = start_row + tile_m
        start_col = col_idx * tile_k
        end_col = start_col + tile_k

        # Boundary check
        if start_row >= input_tensor_2d.shape[0] or start_col >= input_tensor_2d.shape[1]:
            print(f"  - Skipping input_{layer_name}: Tile index ({row_idx}, {col_idx}) is out of bounds for shape {input_tensor_2d.shape}")
        else:
            # Slice the specific tile
            tile = input_tensor_2d[start_row:end_row, start_col:end_col]
            
            # Update the name to include tile index for unique filenames
            plot_name = f'input_{layer_name}_tile_r{row_idx}_c{col_idx}'
            
            # Perform and plot analysis on this tile
            calculate_and_plot_similarity(tile, 'col', plot_name, ANALYSIS_CONFIG)
            calculate_and_plot_similarity(tile, 'row', plot_name, ANALYSIS_CONFIG)

    # --- B. Analyze Weight Tensor ---
    if weight_tensor is not None:
        # Calculate the slice boundaries for the selected tile (using tile_n for rows)
        start_row = row_idx * tile_n
        end_row = start_row + tile_n
        start_col = col_idx * tile_k
        end_col = start_col + tile_k

        # Boundary check
        if start_row >= weight_tensor.shape[0] or start_col >= weight_tensor.shape[1]:
            print(f"  - Skipping weight_{layer_name}: Tile index ({row_idx}, {col_idx}) is out of bounds for shape {weight_tensor.shape}")
        else:
            # Slice the specific tile
            tile = weight_tensor[start_row:end_row, start_col:end_col]

            # Update the name to include tile index for unique filenames
            plot_name = f'weight_{layer_name}_tile_r{row_idx}_c{col_idx}'
            
            # Perform and plot analysis on this tile
            calculate_and_plot_similarity(tile, 'col', plot_name, ANALYSIS_CONFIG)
            calculate_and_plot_similarity(tile, 'row', plot_name, ANALYSIS_CONFIG)