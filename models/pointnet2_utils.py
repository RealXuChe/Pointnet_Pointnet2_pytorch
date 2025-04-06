import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np
from torch_geometric.nn import fps

def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    根据 idx 的维度选择不同的优化策略。
    - 如果 idx 是 2D ([B, S]), 使用 torch.gather。
    - 如果 idx 是 3D ([B, S, K]), 使用高级索引，但优化 batch_indices 创建。

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S] or [B, S, K]
    Return:
        new_points:, indexed points data, [B, S, C] or [B, S, K, C]
    """
    device = points.device
    B, N, C = points.shape

    if idx.dim() == 2:
        # --- Case 1: idx is [B, S] ---
        # Output shape should be [B, S, C]
        # gather 要求 index 和 input 维度数相同 (都为 3)
        S = idx.shape[1]
        # 创建形状为 [B, S, C] 的索引张量，用于 gather
        # idx: [B, S] -> unsqueeze: [B, S, 1] -> expand: [B, S, C]
        idx_expanded = idx.unsqueeze(-1).expand(B, S, C)
        # 沿 dim=1 (N 维度) 收集
        # new_points[b, s, c] = points[b, idx_expanded[b, s, c], c]
        new_points = torch.gather(points, 1, idx_expanded)

    elif idx.dim() == 3:
        # --- Case 2: idx is [B, S, K] ---
        # Output shape should be [B, S, K, C]
        # gather 不适用 (维度不匹配) -> 使用高级索引
        S, K = idx.shape[1], idx.shape[2]

        # 优化 batch_indices 的创建，使用 expand 替代 repeat
        # 目标形状: [B, S, K]
        # 1. 创建 [B, 1, 1] 的 arange
        # 2. expand 到 [B, S, K]
        batch_indices = torch.arange(B, dtype=torch.long, device=device).view(B, 1, 1).expand(B, S, K)

        # 使用高级索引: points[batch_indices, idx, :]
        # batch_indices 提供 B 维索引，idx 提供 N 维索引，: 提供 C 维索引
        # 输出形状由 batch_indices 和 idx 广播决定，为 [B, S, K, C]
        new_points = points[batch_indices, idx, :]

    else:
        raise ValueError(f"Unsupported index dimensions: {idx.dim()}. Expected 2 or 3.")

    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    
    xyz_reshaped = xyz.reshape(B * N, C)
    batch = torch.arange(B, dtype=torch.long, device=device).repeat_interleave(N)
    
    index = fps(xyz_reshaped, batch, ratio=npoint / N, random_start=True, batch_size=B)
    
    centroids_global = index.view(B, npoint)
    batch_offsets = torch.arange(B, dtype=torch.long, device=device).unsqueeze(1) * N
    centroids_local = centroids_global - batch_offsets

    return centroids_local

from torch_geometric.nn.pool import radius as radius_cluster

def query_ball_point(radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    """
    使用 torch_geometric.nn.pool.radius (torch_cluster) 的 query_ball_point。

    Input:
        radius: local region radius (局部区域半径)
        nsample: max sample number in local region (局部区域最大采样点数)
        xyz: all points, [B, N, 3] (所有点)
        new_xyz: query points, [B, S, 3] (查询点/中心点)
    Return:
        group_idx: grouped points index, [B, S, nsample] (分组点的索引)
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape

    if B == 0 or N == 0 or S == 0:
        return torch.empty(B, S, nsample, dtype=torch.long, device=device)

    # 1. 数据格式转换以适配 torch_cluster.radius
    xyz_flat = xyz.reshape(B * N, C)
    new_xyz_flat = new_xyz.reshape(B * S, C)

    # 创建 batch 索引向量
    batch_x = torch.arange(B, dtype=torch.long, device=device).view(B, 1).repeat(1, N).flatten() # B*N
    batch_y = torch.arange(B, dtype=torch.long, device=device).view(B, 1).repeat(1, S).flatten() # B*S

    # 2. 调用 torch_geometric.nn.pool.radius (底层是 torch_cluster)
    # 它返回的是 [2, num_pairs] 的索引张量，row[0]是y(查询点)的索引, row[1]是x(邻居点)的索引
    # 注意：这里的索引是相对于 xyz_flat 和 new_xyz_flat 的全局索引
    # 我们将 nsample 传递给 max_num_neighbors
    # 注意：torch_cluster.radius 可能使用不同的策略处理超过 max_num_neighbors 的情况（例如随机采样），
    # 而原始实现是排序后取前 nsample 个。这是一个潜在的行为差异。
    assign_index = radius_cluster(
        x=xyz_flat,                # 源点云 (邻居候选)
        y=new_xyz_flat,            # 查询点
        r=radius,                  # 半径
        batch_x=batch_x,           # 源点云的 batch 索引
        batch_y=batch_y,           # 查询点的 batch 索引
        max_num_neighbors=nsample, # 最大邻居数
        num_workers=1             # 通常在 GPU 上此参数无效，设为1
    )

    # 3. 结果格式转换与填充，以匹配原始输出 [B, S, nsample]

    # 获取查询点索引 (相对于 new_xyz_flat) 和邻居点索引 (相对于 xyz_flat)
    # y_idx_flat: 范围 [0, B*S - 1]
    # x_idx_flat: 范围 [0, B*N - 1]
    y_idx_flat = assign_index[0]
    x_idx_flat = assign_index[1]

    # 如果没有找到任何邻居对
    if assign_index.shape[1] == 0:
        # 原始实现会用第一个点（如果存在）填充，但如果根本找不到点（都在半径外），
        # 它会填充 N。这里我们遵循后一种情况，因为 radius_cluster 找不到就不会返回。
        # 但原始代码有一个特殊情况：如果半径内有点，但少于nsample，会用第一个点填充。
        # 为了模拟这一点，我们先用 N 填充。
        # 稍后会处理用第一个点填充的情况。
        group_idx = torch.full((B, S, nsample), N, dtype=torch.long, device=device)
        return group_idx

    # 将邻居点的全局索引转换为 batch 内的索引
    # x_idx_in_batch: 范围 [0, N - 1]
    x_idx_in_batch = x_idx_flat % N

    # 获取查询点的 batch 索引和 batch 内索引
    # b_idx: 范围 [0, B - 1]
    # s_idx_in_batch: 范围 [0, S - 1]
    b_idx = y_idx_flat // S
    s_idx_in_batch = y_idx_flat % S

    # 创建最终的输出张量，并初始化为一个特殊值（例如 -1 或 N）
    # 使用 N 作为初始填充值，与原始代码中半径外的点的标记一致
    group_idx = torch.full((B, S, nsample), N, dtype=torch.long, device=device)

    # 计算每个查询点找到了多少个邻居
    # unique_y_idx: 每个查询点的唯一全局索引
    # y_inverse_indices: assign_index 中的每个条目对应 unique_y_idx 中的哪个索引
    # counts: 每个 unique_y_idx (查询点) 对应的邻居数量 (不超过 nsample)
    unique_y_idx, y_inverse_indices, counts = torch.unique(y_idx_flat, return_inverse=True, return_counts=True)

    # 计算每个邻居在其所属查询点的邻居列表中的序号 k (0 到 count-1)
    # 这可以通过 y_inverse_indices 创建一个从0开始的计数器来实现
    k_indices = torch.arange(assign_index.shape[1], device=device)
    # 计算每个 unique_y_idx 起始位置的偏移量
    offsets = torch.zeros_like(unique_y_idx, dtype=torch.long)
    offsets[1:] = torch.cumsum(counts[:-1], dim=0)
    # k = 当前全局索引 - 该查询点第一个邻居的全局索引
    k_indices -= offsets[y_inverse_indices]

    # 将找到的邻居索引填入 group_idx
    # group_idx[b, s, k] = n
    group_idx[b_idx, s_idx_in_batch, k_indices] = x_idx_in_batch

    # 4. 处理邻居数量不足 nsample 的情况 (模拟原始代码的填充逻辑)
    # 原始代码: 用找到的第一个邻居的索引填充空位 (值为 N 的位置)
    # 找到 group_idx 中每个查询点的第一个有效邻居 (索引 < N)
    # 如果 group_idx[b, s, 0] 是有效索引 (< N), 则它就是第一个邻居
    first_neighbor_indices = group_idx[:, :, 0].clone() # 形状 [B, S]

    # 创建一个 mask 标记需要填充的位置 (值为 N 且 k > 0，因为 k=0 不需要填充自己)
    mask = (group_idx == N) & (torch.arange(nsample, device=device).view(1, 1, nsample) > 0)

    # 扩展 first_neighbor_indices 以匹配 group_idx 的形状进行广播
    first_neighbor_expanded = first_neighbor_indices.unsqueeze(-1).expand(B, S, nsample)

    # 使用第一个邻居的索引填充 mask 标记的位置
    group_idx[mask] = first_neighbor_expanded[mask]

    # 最后一步特殊情况：如果一个查询点连第一个邻居都找不到 (first_neighbor_indices[b, s] == N)，
    # 那么原始代码的行为是用 N 填充所有 nsample 个位置。
    # 我们当前的 group_idx 在这种情况下已经是全 N 了，所以不需要额外处理。

    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points


class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.npoint
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points =  F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points

