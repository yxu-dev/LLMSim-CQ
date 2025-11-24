import os
# 设置环境变量，强制单线程执行以避免BLAS错误
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch
import numpy as np
import argparse
from tqdm import tqdm

# ---------------------------------------------------------------------------- #
#         加权 K-Means (GPU Implementation using PyTorch)         #
# ---------------------------------------------------------------------------- #

def weighted_kmeans_torch(data, weights, K, max_iter=100, device='cuda'):
    """
    使用PyTorch在GPU上执行加权K-Means。
    
    Args:
    - data (torch.Tensor): [N, D] 维的输入数据 (例如 [32768, 4])
    - weights (torch.Tensor): [D] 维的权重 (例如 [4])
    - K (int): 聚类数量 (例如 256)
    - max_iter (int): 最大迭代次数
    
    Returns:
    - torch.Tensor: [K, D] 维的最终质心
    """
    N, D = data.shape
    if device == 'cuda' and torch.cuda.is_available():
        data = data.to(device)
        weights = weights.to(device)
    else:
        device = 'cpu'
        if not torch.cuda.is_available():
            print("警告: 未检测到 CUDA. 回退到 CPU 执行 (速度可能较慢)。")

    # 初始化质心：从数据点中随机选择K个
    indices = torch.randperm(N, device=device)[:K]
    centroids = data[indices]

    weights_expanded = weights[None, :] # Shape: [1, D]
    
    last_assignments = None

    for i in range(max_iter):
        # 1. 分配步骤 (Assignment Step)
        data_expanded = data.unsqueeze(1)
        centroids_expanded = centroids.unsqueeze(0)
        diff_sq = (data_expanded - centroids_expanded) ** 2
        weighted_diff_sq = diff_sq * weights_expanded
        distance_matrix = weighted_diff_sq.sum(dim=2)
        assignments = torch.argmin(distance_matrix, dim=1)

        # 检查是否收敛
        if last_assignments is not None and torch.all(assignments == last_assignments):
            # print(f"  -> Converged in {i+1} iterations.")
            break
        last_assignments = assignments

        # 2. 更新步骤 (Update Step)
        new_centroids = torch.zeros_like(centroids)
        counts = torch.zeros(K, device=device, dtype=data.dtype)
        
        assignments_expanded = assignments.unsqueeze(1).expand(-1, D)
        new_centroids.scatter_add_(0, assignments_expanded, data)
        counts.scatter_add_(0, assignments, torch.ones_like(assignments, dtype=data.dtype))

        # 处理空簇（避免除零）
        empty_clusters = (counts < 1e-6)
        counts = torch.clamp(counts, min=1.0) # 避免除零
        new_centroids /= counts.unsqueeze(1)
        
        # 重新初始化空簇 (如果存在)
        if torch.any(empty_clusters):
            num_empty = empty_clusters.sum().item()
            # print(f"  -> Warning: {num_empty} empty clusters found. Re-initializing.")
            # 从数据点中随机选择新的质心来替换空簇
            random_indices = torch.randint(0, N, (num_empty,), device=device)
            new_centroids[empty_clusters] = data[random_indices]

        centroids = new_centroids

    return centroids

# ---------------------------------------------------------------------------- #
#                质心学习主函数                 #
# ---------------------------------------------------------------------------- #

def learn_weighted_cq_centroids(tensors, fisher_weights_all_groups, num_bits=8):
    """
    为给定的张量列表学习 *加权* 耦合量化质心。
    """
    if not tensors:
        print("  错误: 激活张量列表为空。")
        return None

    # 1. 准备数据：将所有张量合并并重塑
    try:
        combined_tensor = torch.cat(tensors, dim=1) # shape: [1, N_total, D]
        data_all_channels = combined_tensor.squeeze(0) # shape: [N_total, D]
    except Exception as e:
        print(f"  错误: 合并张量时出错: {e}")
        return None
    
    N_total, D_total = data_all_channels.shape
    num_groups, num_coupled_channels = fisher_weights_all_groups.shape
    num_centroids = 2**num_bits

    if D_total != num_groups * num_coupled_channels:
        print(f"  错误: 总通道数 {D_total} 与 Fisher 权重形状 {fisher_weights_all_groups.shape} 不匹配。")
        return None

    print(f"  总通道数: {D_total}")
    print(f"  耦合通道数 (C): {num_coupled_channels}")
    print(f"  分组数量: {num_groups}")
    print(f"  每个分组的质心数 (K): {num_centroids}")

    all_centroids_list = []

    # 2. 对每个通道组独立运行 *加权* K-means
    for i in tqdm(range(num_groups), desc="    学习质心", ncols=100, leave=False):
        start_channel = i * num_coupled_channels
        end_channel = start_channel + num_coupled_channels
        
        group_data = data_all_channels[:, start_channel:end_channel]
        group_weights = fisher_weights_all_groups[i, :]
        
        group_centroids = weighted_kmeans_torch(
            group_data.to(torch.float32), 
            group_weights.to(torch.float32),
            K=num_centroids,
            max_iter=100
        )
        
        all_centroids_list.append(group_centroids.cpu())
        
    return torch.stack(all_centroids_list, dim=0).numpy().astype(np.float32)

# ---------------------------------------------------------------------------- #
#                         主执行流程                                   #
# ---------------------------------------------------------------------------- #

def main(args):
    # --- 1. 数据加载与预处理 ---
    print(f"--- 步骤 1: 加载第 {args.layer_idx} 层的激活 (Activations) 数据 ---")
    print(f"  数据来源: {args.data_path}")
    K_tensors = []
    V_tensors = []
    # 固定的样本列表
    samples_to_load = [2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 14, 17, 19, 20, 21, 23] #Llama-3.1-8b, Llama-3.2-1b and Llama-3.2-3b
    
    for i in samples_to_load:
        k_path = os.path.join(args.data_path, f"sample{i}_Prefill_step0_layer{args.layer_idx}_key.pt")
        v_path = os.path.join(args.data_path, f"sample{i}_Prefill_step0_layer{args.layer_idx}_value.pt")
        
        try:
            k_tensor = torch.load(k_path, map_location='cpu')
            v_tensor = torch.load(v_path, map_location='cpu')
            
            # 截断到 2048 (如果需要)
            k_truncated = k_tensor[:, :2048, :]
            v_truncated = v_tensor[:, :2048, :]
            
            K_tensors.append(k_truncated)
            V_tensors.append(v_truncated)
        except FileNotFoundError:
            print(f"  警告: 找不到文件 sample{i} (layer {args.layer_idx})。跳过。")
            continue
        except Exception as e:
            print(f"  警告: 加载 sample{i} (layer {args.layer_idx}) 时出错: {e}。跳过。")
            continue

    if not K_tensors or not V_tensors:
        print("错误: 未能加载任何激活数据。请检查 --data_path 和 --layer_idx。")
        return
        
    print(f"  成功加载了 {len(K_tensors)} 个样本用于训练。\n")

    # --- 2. 加载 Fisher 权重 ---
    print(f"--- 步骤 2: 加载 Fisher 权重 ({args.fisher_path}) ---")
    try:
        fisher_data = torch.load(args.fisher_path, map_location='cpu')['fisher']
    except FileNotFoundError:
        print(f"错误: 找不到 Fisher 权重文件: {args.fisher_path}")
        print("请先运行 collect_Fisher_gradients.py 脚本。")
        return
    except KeyError:
        print(f"错误: {args.fisher_path} 文件中没有 'fisher' 键。")
        return

    LAYER_IDX = args.layer_idx
    k_fisher_weights = fisher_data.get((LAYER_IDX, 'k'))
    v_fisher_weights = fisher_data.get((LAYER_IDX, 'v'))

    if k_fisher_weights is None or v_fisher_weights is None:
        print(f"错误: 无法从 Fisher 文件中找到第 {LAYER_IDX} 层的 'k' 或 'v' 权重。")
        print("请确保 collect_Fisher_gradients.py 脚本已正确运行并包含了该层。")
        return

    print(f"  成功加载第 {LAYER_IDX} 层的 Fisher 权重。")
    print(f"    K 权重 shape: {k_fisher_weights.shape}")
    print(f"    V 权重 shape: {v_fisher_weights.shape}\n")

    # --- 3. 运行加权 K-Means ---
    print("--- 步骤 3: 运行 Fisher 引导的加权 K-Means ---")
    B = 8  # 比特数 (固定为 8b)
    os.makedirs(args.output_dir, exist_ok=True)

    print("--> 开始为Key张量学习质心 (使用加权K-Means)...")
    k_centroids = learn_weighted_cq_centroids(K_tensors, k_fisher_weights, num_bits=B)
    if k_centroids is not None:
        key_filename = f"k_centroids_fisher_layer{args.layer_idx}.npy"
        key_save_path = os.path.join(args.output_dir, key_filename)
        np.save(key_save_path, k_centroids)
        print(f"  Key质心学习完成。Shape: {k_centroids.shape}")
        print(f"  已保存到: {key_save_path}\n")

    print("--> 开始为Value张量学习质心 (使用加权K-Means)...")
    v_centroids = learn_weighted_cq_centroids(V_tensors, v_fisher_weights, num_bits=B)
    if v_centroids is not None:
        value_filename = f"v_centroids_fisher_layer{args.layer_idx}.npy"
        value_save_path = os.path.join(args.output_dir, value_filename)
        np.save(value_save_path, v_centroids)
        print(f"  Value质心学习完成。Shape: {v_centroids.shape}")
        print(f"  已保存到: {value_save_path}\n")
    
    print("--- 流程完成 ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="运行加权K-Means来学习Fisher引导的CQ质心。")
    parser.add_argument(
        "--fisher_path", 
        type=str, 
        required=True, 
        help="由 collect_Fisher_gradients.py 生成的 'fisher_diag.pt' 文件路径。"
    )
    parser.add_argument(
        "--data_path", 
        type=str, 
        required=True, 
        help="包含 .pt 激活文件 (kv-simi) 的目录路径。"
    )
    parser.add_argument(
        "--layer_idx", 
        type=int, 
        required=True, 
        help="要处理的目标层索引 (例如 15)。"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=".", 
        help="保存 .npy 质心文件的输出目录。 (默认: 当前目录)"
    )
    
    args = parser.parse_args()
    main(args)