import torch.nn as nn
import torch.nn.functional as F
from models.pointnet2_utils import PointNetSetAbstraction,PointNetFeaturePropagation

class get_model(nn.Module):
    def __init__(self, num_classes, input_channels=3): # 默认为 3 (XYZ)，如果使用强度则传入 4
        super(get_model, self).__init__()
        # KITTI XYZ (3) -> 3 + 3 = 6
        # KITTI XYZI (4) -> 4 + 3 = 7
        self.sa1 = PointNetSetAbstraction(npoint=1024, radius=0.1, nsample=32, in_channel=input_channels + 3, mlp=[32, 32, 64], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=256, radius=0.2, nsample=32, in_channel=64 + 3, mlp=[64, 64, 128], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=64, radius=0.4, nsample=32, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa4 = PointNetSetAbstraction(npoint=16, radius=0.8, nsample=32, in_channel=256 + 3, mlp=[256, 256, 512], group_all=False)
        self.fp4 = PointNetFeaturePropagation(in_channel=768, mlp=[256, 256]) # 512 + 256
        self.fp3 = PointNetFeaturePropagation(in_channel=384, mlp=[256, 256]) # 128 + 256
        self.fp2 = PointNetFeaturePropagation(in_channel=320, mlp=[256, 128]) # FP2 输入应为 l1_points(64) + l2_points_interpolated(来自fp3, 256) = 320.
        self.fp1 = PointNetFeaturePropagation(in_channel=128, mlp=[128, 128, 128]) # l0_points(None) + l1_points_interpolated(来自fp2, 128) = 128

        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz):
        # xyz: 输入点云, 形状 [B, C, N], C = input_channels (e.g., 3 or 4)
        l0_points = xyz # 保留所有特征
        l0_xyz = xyz[:,:3,:] # 仅提取前 3 维作为坐标

        # 注意：sa1 的输入 points 是 l0_points (包含所有特征 C 维)
        #       sa1 内部计算时 new_points = concat(relative_xyz(3), grouped_points(C)) -> C+3 维送入第一个 MLP
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points) # l1_points 是 64 维
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points) # l2_points 是 128 维
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points) # l3_points 是 256 维
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points) # l4_points 是 512 维

        # FP 模块的 in_channel 参数是拼接后的维度
        # fp4: l3_points(256) + interpolated l4_points(512) = 768
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points) # 输出 256 维
        # fp3: l2_points(128) + interpolated l3_points(256) = 384
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points) # 输出 256 维
        # fp2: l1_points(64) + interpolated l2_points(256) = 320
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points) # 输出 128 维
        # fp1: l0_points(None) + interpolated l1_points(128) = 128
        l0_points_out = self.fp1(l0_xyz, l1_xyz, None, l1_points) # 输出 128 维

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points_out))))
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1) # 输出 Log-Probabilities
        x = x.permute(0, 2, 1)
        return x, None # 返回 None 以保持接口一致性，或者直接返回 x


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()
        # 可以选择在这里初始化 NLLLoss，如果需要存储状态，但目前不需要

    def forward(self, pred, target, class_weights=None, ignore_index=None):
        """
        计算 NLL Loss

        Args:
            pred (torch.Tensor): 模型预测的 Log-Probabilities, 形状 [B, N, num_classes] 或 [B*N, num_classes]
                                 注意：原始模型输出是 [B, N, C]，但训练脚本中会 view 成 [B*N, C]
                                 这里假设 pred 是 [B*N, C] 或 [B, C, N] 格式
            target (torch.Tensor): 真实标签, 形状 [B, N] 或 [B*N]
            class_weights (torch.Tensor, optional): 类别的权重, 形状 [num_classes]。默认为 None。
            ignore_index (int, optional): 指定 target 中要忽略的标签索引。默认为 None。

        Returns:
            torch.Tensor: 计算得到的 loss (标量)
        """
        # F.nll_loss 期望输入形状为 [minibatch, C] 和 target [minibatch]
        # 或者 [minibatch, C, d1, d2,...] 和 target [minibatch, d1, d2,...]
        # 我们的 pred 在进入损失函数前通常被 reshape 成 [B*N, C]
        # 我们的 target 在进入损失函数前通常被 reshape 成 [B*N]

        total_loss = F.nll_loss(pred, target, weight=class_weights, ignore_index=ignore_index if ignore_index is not None else -100)
        # 注意: F.nll_loss 的 ignore_index 默认为 -100，所以如果传入 None，需要转为 -100 或确保训练脚本传入有效值

        return total_loss