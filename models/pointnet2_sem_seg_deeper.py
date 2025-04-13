# models/pointnet2_sem_seg.py (6-Layer Version)

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pointnet2_utils import PointNetSetAbstraction,PointNetFeaturePropagation

class get_model(nn.Module):
    def __init__(self, num_classes, input_channels=3): # 默认为 3 (XYZ)，如果使用强度则传入 4
        super(get_model, self).__init__()
        C_in = input_channels # 输入特征维度 (e.g., 3 for XYZ, 4 for XYZI)

        # --- SA Layers ---
        # Layer 1: N -> 4096 points
        self.sa1 = PointNetSetAbstraction(npoint=4096, radius=0.1, nsample=32, in_channel=C_in + 3, mlp=[32, 32, 64], group_all=False)
        c_out_1 = 64
        # Layer 2: 4096 -> 1024 points
        self.sa2 = PointNetSetAbstraction(npoint=1024, radius=0.2, nsample=32, in_channel=c_out_1 + 3, mlp=[64, 64, 128], group_all=False)
        c_out_2 = 128
        # Layer 3: 1024 -> 256 points
        self.sa3 = PointNetSetAbstraction(npoint=256, radius=0.4, nsample=32, in_channel=c_out_2 + 3, mlp=[128, 128, 256], group_all=False)
        c_out_3 = 256
        # Layer 4: 256 -> 64 points
        self.sa4 = PointNetSetAbstraction(npoint=64, radius=0.8, nsample=32, in_channel=c_out_3 + 3, mlp=[256, 256, 512], group_all=False)
        c_out_4 = 512
        # Layer 5: 64 -> 16 points
        self.sa5 = PointNetSetAbstraction(npoint=16, radius=1.6, nsample=32, in_channel=c_out_4 + 3, mlp=[512, 512, 1024], group_all=False)
        c_out_5 = 1024
        # Layer 6: 16 -> 4 points (或者 npoint=1 进行全局抽象)
        self.sa6 = PointNetSetAbstraction(npoint=4, radius=3.2, nsample=16, in_channel=c_out_5 + 3, mlp=[1024, 1024, 1024], group_all=False)
        c_out_6 = 1024

        # --- FP Layers ---
        # FP Layer 6: Upsample sa6 -> sa5 resolution
        # Input: sa5_features (c_out_5=1024) + interpolated sa6_features (c_out_6=1024)
        self.fp6 = PointNetFeaturePropagation(in_channel=c_out_5 + c_out_6, mlp=[512, 512])
        c_fp6_out = 512
        # FP Layer 5: Upsample fp6 -> sa4 resolution
        # Input: sa4_features (c_out_4=512) + interpolated fp6_features (c_fp6_out=512)
        self.fp5 = PointNetFeaturePropagation(in_channel=c_out_4 + c_fp6_out, mlp=[512, 512])
        c_fp5_out = 512
        # FP Layer 4: Upsample fp5 -> sa3 resolution
        # Input: sa3_features (c_out_3=256) + interpolated fp5_features (c_fp5_out=512)
        self.fp4 = PointNetFeaturePropagation(in_channel=c_out_3 + c_fp5_out, mlp=[256, 256])
        c_fp4_out = 256
        # FP Layer 3: Upsample fp4 -> sa2 resolution
        # Input: sa2_features (c_out_2=128) + interpolated fp4_features (c_fp4_out=256)
        self.fp3 = PointNetFeaturePropagation(in_channel=c_out_2 + c_fp4_out, mlp=[256, 256])
        c_fp3_out = 256
        # FP Layer 2: Upsample fp3 -> sa1 resolution
        # Input: sa1_features (c_out_1=64) + interpolated fp3_features (c_fp3_out=256)
        self.fp2 = PointNetFeaturePropagation(in_channel=c_out_1 + c_fp3_out, mlp=[256, 128])
        c_fp2_out = 128
        # FP Layer 1: Upsample fp2 -> original resolution
        # Input: l0_features (None, effectively 0 channels) + interpolated fp2_features (c_fp2_out=128)
        self.fp1 = PointNetFeaturePropagation(in_channel=c_fp2_out, mlp=[128, 128, 128])
        c_fp1_out = 128

        # --- Final Classification Head ---
        self.conv1 = nn.Conv1d(c_fp1_out, 128, 1) # Input channel matches fp1 output
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz):
        # xyz: 输入点云, 形状 [B, C, N], C = input_channels (e.g., 3 or 4)
        l0_points = xyz # 保留所有特征
        l0_xyz = xyz[:,:3,:] # 仅提取前 3 维作为坐标

        # --- Encoder (SA Layers) ---
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points) # N -> 4096
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points) # 4096 -> 1024
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points) # 1024 -> 256
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points) # 256 -> 64
        l5_xyz, l5_points = self.sa5(l4_xyz, l4_points) # 64 -> 16
        l6_xyz, l6_points = self.sa6(l5_xyz, l5_points) # 16 -> 4

        # --- Decoder (FP Layers) ---
        l5_points_fp = self.fp6(l5_xyz, l6_xyz, l5_points, l6_points)
        l4_points_fp = self.fp5(l4_xyz, l5_xyz, l4_points, l5_points_fp)
        l3_points_fp = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points_fp)
        l2_points_fp = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points_fp)
        l1_points_fp = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points_fp)
        l0_points_out = self.fp1(l0_xyz, l1_xyz, None, l1_points_fp)

        # --- Head ---
        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points_out))))
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1) # 输出 Log-Probabilities
        x = x.permute(0, 2, 1) # [B, N, num_classes]
        return x, None # 返回 None 以保持接口一致性


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()
        # NLLLoss for LogSoftmax output from the model
        # Reduction is 'mean' by default, which averages over the batch *and* points (after ignoring indices)

    def forward(self, pred, target, class_weights=None, ignore_index=None):
        """
        计算 NLL Loss

        Args:
            pred (torch.Tensor): 模型预测的 Log-Probabilities.
                                 训练脚本中 reshape 后的形状: [B*N, num_classes]
            target (torch.Tensor): 真实标签.
                                   训练脚本中 reshape 后的形状: [B*N]
            class_weights (torch.Tensor, optional): 类别的权重, 形状 [num_classes]。默认为 None。
            ignore_index (int, optional): 指定 target 中要忽略的标签索引。默认为 None (-100 used by default in F.nll_loss if not specified)

        Returns:
            torch.Tensor: 计算得到的 loss (标量)
        """
        # F.nll_loss 期望输入形状 [minibatch, C], target [minibatch]
        # 或者 [N, C, d1, ...], target [N, d1, ...]
        # 我们的 pred 和 target 已经被 reshape 成 [B*N, C] 和 [B*N]
        total_loss = F.nll_loss(pred, target, weight=class_weights, ignore_index=ignore_index if ignore_index is not None else -100)

        return total_loss