import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import torch
import numpy as np
import argparse
import glob
from tqdm import tqdm


# K-Means++ 初始化 (PyTorch GPU)
def kmeans_plusplus_init(data, weights, K, device='cuda'):
    """
    实现加权距离下的 K-Means++ 初始化 (论文标准)
    """
    N, D = data.shape
    # 必须使用 float32 进行距离计算，防止溢出
    data = data.to(torch.float32)
    weights = weights.to(torch.float32)
    
    centroids = torch.empty((K, D), device=device, dtype=torch.float32)
    
    # 1. 随机选择第一个中心
    first_idx = torch.randint(0, N, (1,), device=device)
    centroids[0] = data[first_idx]
    
    weights_expanded = weights[None, :] # [1, D]
    
    # 2. 选择剩余 K-1 个中心
    # dist_sq: [N]
    curr_dist_sq = ((data - centroids[0]) ** 2 * weights_expanded).sum(dim=1)
    
    for i in range(1, K):
        # 按距离平方概率采样
        probs = curr_dist_sq / curr_dist_sq.sum()
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 采样下一个索引
        next_idx = torch.multinomial(probs, 1)
        centroids[i] = data[next_idx]
        
        # 更新距离
        new_dist_sq = ((data - centroids[i]) ** 2 * weights_expanded).sum(dim=1)
        curr_dist_sq = torch.minimum(curr_dist_sq, new_dist_sq)
        
    return centroids

# 加权 K-Means (GPU Implementation)
def weighted_kmeans_torch(data, weights, K, max_iter=100, device='cuda'):
    """
    运行加权 K-Means
    输入 data 应该是 [N, D]，类型 float32
    """
    N, D = data.shape
    if device == 'cuda' and torch.cuda.is_available():
        data = data.to(device).to(torch.float32)
        weights = weights.to(device).to(torch.float32)
    else:
        device = 'cpu'
        data = data.to(torch.float32)

    # 初始化
    centroids = kmeans_plusplus_init(data, weights, K, device=device)

    weights_expanded = weights[None, :] # Shape: [1, D]
    last_assignments = None

    for i in range(max_iter):
        # 1. Assignment Step (Weighted Euclidean)
        # 距离公式: sum(w * (x - c)^2)
        # 展开维度: data[N, 1, D], centroids[1, K, D]
        # 注意：显存优化点。如果 N 很大 (如 16*2048=32k)，直接广播 [N, K, D] 
        # 32768 * 256 * 128 * 4bytes ≈ 4GB 显存。
        diff_sq = (data.unsqueeze(1) - centroids.unsqueeze(0)) ** 2
        weighted_diff_sq = diff_sq * weights_expanded
        distance_matrix = weighted_diff_sq.sum(dim=2) # [N, K]
        
        assignments = torch.argmin(distance_matrix, dim=1)

        # 收敛检查
        if last_assignments is not None and torch.equal(assignments, last_assignments):
            break
        last_assignments = assignments

        # Update Step
        new_centroids = torch.zeros_like(centroids)
        counts = torch.zeros(K, device=device, dtype=torch.float32)
        
        assignments_expanded = assignments.unsqueeze(1).expand(-1, D)
        new_centroids.scatter_add_(0, assignments_expanded, data)
        counts.scatter_add_(0, assignments, torch.ones_like(assignments, dtype=torch.float32))

        # 处理空簇
        empty_clusters = (counts < 1e-6)
        counts = torch.clamp(counts, min=1.0)
        new_centroids /= counts.unsqueeze(1)
        
        if torch.any(empty_clusters):
            num_empty = empty_clusters.sum().item()
            random_indices = torch.randint(0, N, (num_empty,), device=device)
            new_centroids[empty_clusters] = data[random_indices]

        centroids = new_centroids

    return centroids

# 质心学习主函数
def learn_weighted_cq_centroids(tensors, fisher_weights_all_groups, num_bits=8):
    if not tensors:
        print("  错误: 激活张量列表为空。")
        return None

    # 合并数据: [Total_N, D]
    combined_tensor = torch.cat(tensors, dim=0).to(torch.float32)
    
    N_total, D_total = combined_tensor.shape
    num_groups, num_coupled_channels = fisher_weights_all_groups.shape
    num_centroids = 2**num_bits

    # 简单校验
    # 注意：导出的 Fisher 可能是 [Groups, C]，但也可能因为 Head 对齐问题有细微差别
    # 这里我们只检查总通道数是否吻合
    if D_total != num_groups * num_coupled_channels:
         print(f"  错误: 数据通道总数 {D_total} != Fisher分组 {num_groups} * 耦合通道 {num_coupled_channels}")
         return None

    all_centroids_list = []

    # 对每个组进行聚类
    # 使用 tqdm 显示进度
    for i in tqdm(range(num_groups), desc="    Clustering Groups", ncols=100, leave=False):
        start_channel = i * num_coupled_channels
        end_channel = start_channel + num_coupled_channels
        
        group_data = combined_tensor[:, start_channel:end_channel]
        group_weights = fisher_weights_all_groups[i, :]
        
        # 异常值保护
        if torch.isnan(group_data).any() or torch.isinf(group_data).any():
             # 简单的 fallback: 均值为 0，或者随机
             group_data = torch.nan_to_num(group_data, nan=0.0)
        
        group_centroids = weighted_kmeans_torch(
            group_data, 
            group_weights,
            K=num_centroids,
            max_iter=100
        )
        
        all_centroids_list.append(group_centroids.cpu())
        
    return torch.stack(all_centroids_list, dim=0).numpy().astype(np.float32)


def main(args):
    # 路径适配逻辑
    # 检查 data_path 是否直接包含 .pt，或者包含在 data_path/kv_cache 中
    search_dir = args.data_path
    if os.path.isdir(os.path.join(args.data_path, "kv_cache")):
        search_dir = os.path.join(args.data_path, "kv_cache")
        print(f"检测到 kv_cache 子目录，将从这里读取数据: {search_dir}")
    
    # 同样检查 Fisher 文件
    fisher_file = os.path.join(args.data_path, "fisher_diag.pt")
    if not os.path.exists(fisher_file):
        # 尝试用户传入的 fisher_path 参数（如果有）
        fisher_file = args.fisher_path
        
    if not os.path.exists(fisher_file):
         print(f"错误: 找不到 Fisher 文件。请检查路径: {fisher_file}")
         return

    print(f"--- 步骤 1: 加载 Layer {args.layer_idx} 激活数据 ---")
    
    # 动态匹配文件: sample*_layer{layer}_key.pt
    key_pattern = os.path.join(search_dir, f"sample*_layer{args.layer_idx}_key.pt")
    value_pattern = os.path.join(search_dir, f"sample*_layer{args.layer_idx}_value.pt")
    
    key_files = sorted(glob.glob(key_pattern))
    value_files = sorted(glob.glob(value_pattern))
    
    if len(key_files) == 0:
        print(f"错误: 在 {search_dir} 未找到层 {args.layer_idx} 的 key 文件。")
        print(f"搜索模式: {key_pattern}")
        return

    max_samples = 16
    key_files = key_files[:max_samples]
    value_files = value_files[:max_samples]

    K_tensors = []
    V_tensors = []

    print(f"  加载 {len(key_files)} 个样本文件...")
    for kf, vf in zip(key_files, value_files):
        try:
            # 加载并转为 CPU float32
            k_t = torch.load(kf, map_location='cpu').float()
            v_t = torch.load(vf, map_location='cpu').float()
            
            # 展平: [Batch, Seq, Dim] -> [N, D]
            if k_t.dim() == 3:
                k_t = k_t.reshape(-1, k_t.shape[-1])
                v_t = v_t.reshape(-1, v_t.shape[-1])
                
            K_tensors.append(k_t)
            V_tensors.append(v_t)
        except Exception as e:
            print(f"  警告: 加载 {kf} 失败: {e}")

    # 加载 Fisher 权重
    print(f"--- 步骤 2: 加载 Fisher 权重 ---")
    try:
        loaded = torch.load(fisher_file, map_location='cpu')
        fisher_data = loaded['fisher']
    except Exception as e:
        print(f"错误: 加载 Fisher 文件失败: {e}")
        return

    # 获取对应层的权重
    # key 是 (layer_idx, 'k') 或 (layer_idx, 'v')
    k_fisher_weights = fisher_data.get((args.layer_idx, 'k'))
    v_fisher_weights = fisher_data.get((args.layer_idx, 'v'))

    if k_fisher_weights is None:
        print(f"错误: Fisher 数据中缺少 Layer {args.layer_idx}。Available keys example: {list(fisher_data.keys())[0]}")
        return

    print(f"  Fisher Shape: {k_fisher_weights.shape}")

    # 运行加权 K-Means
    print(f"--- 步骤 3: 学习质心 (Layer {args.layer_idx}) ---")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"--> Learning KEY Centroids...")
    k_centroids = learn_weighted_cq_centroids(K_tensors, k_fisher_weights, num_bits=args.num_bits)
    if k_centroids is not None:
        save_path = os.path.join(args.output_dir, f"k_centroids_fisher_layer{args.layer_idx}.npy")
        np.save(save_path, k_centroids)
        print(f"  [Saved] {save_path}")

    print(f"--> Learning VALUE Centroids...")
    v_centroids = learn_weighted_cq_centroids(V_tensors, v_fisher_weights, num_bits=args.num_bits)
    if v_centroids is not None:
        save_path = os.path.join(args.output_dir, f"v_centroids_fisher_layer{args.layer_idx}.npy")
        np.save(save_path, v_centroids)
        print(f"  [Saved] {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, 
                        help="包含 Fisher 文件和 kv_cache 文件夹的根目录 (Step 2 的 output_dir)")
    parser.add_argument("--fisher_path", type=str, default="", 
                        help="可选：手动指定 fisher_diag.pt 路径，如果不指定则在 data_path 下找")
    parser.add_argument("--layer_idx", type=int, required=True, help="目标层索引")
    parser.add_argument("--output_dir", type=str, default="centroids", help="质心保存目录")
    parser.add_argument("--num_bits", type=int, default=8, help="每组码本位宽，质心数为 2^num_bits")
    args = parser.parse_args()
    main(args)