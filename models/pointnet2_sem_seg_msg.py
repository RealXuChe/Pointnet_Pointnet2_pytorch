import torch.nn as nn
import torch.nn.functional as F
from models.pointnet2_utils import PointNetSetAbstractionMsg,PointNetFeaturePropagation

class get_model(nn.Module):
    def __init__(self, num_classes, input_channels=3): # 默认为 3 (XYZ)
        super(get_model, self).__init__()

        # KITTI XYZ (3) -> in_channel=3
        # KITTI XYZI (4) -> in_channel=4
        self.sa1 = PointNetSetAbstractionMsg(npoint=1024, radius_list=[0.05, 0.1], nsample_list=[16, 32], in_channel=input_channels, mlp_list=[[16, 16, 32], [32, 32, 64]])
        # 后续 SA 层的 in_channel 是上一层 MLP 输出的拼接维度
        c_out_sa1 = 32 + 64
        self.sa2 = PointNetSetAbstractionMsg(npoint=256, radius_list=[0.1, 0.2], nsample_list=[16, 32], in_channel=c_out_sa1, mlp_list=[[64, 64, 128], [64, 96, 128]])
        c_out_sa2 = 128 + 128
        self.sa3 = PointNetSetAbstractionMsg(npoint=64, radius_list=[0.2, 0.4], nsample_list=[16, 32], in_channel=c_out_sa2, mlp_list=[[128, 196, 256], [128, 196, 256]])
        c_out_sa3 = 256 + 256
        self.sa4 = PointNetSetAbstractionMsg(npoint=16, radius_list=[0.4, 0.8], nsample_list=[16, 32], in_channel=c_out_sa3, mlp_list=[[256, 256, 512], [256, 384, 512]])
        c_out_sa4 = 512 + 512

        # FP 层的 in_channel 是拼接后的维度
        # fp4: l3_points(c_out_sa3=512) + interpolated l4_points(c_out_sa4=1024) = 1536
        self.fp4 = PointNetFeaturePropagation(in_channel=c_out_sa3 + c_out_sa4, mlp=[256, 256]) # 512 + 1024 = 1536
        # fp3: l2_points(c_out_sa2=256) + interpolated l3_points(fp4_out=256) = 512
        self.fp3 = PointNetFeaturePropagation(in_channel=c_out_sa2 + 256, mlp=[256, 256]) # 256 + 256 = 512
        # fp2: l1_points(c_out_sa1=96) + interpolated l2_points(fp3_out=256) = 352
        self.fp2 = PointNetFeaturePropagation(in_channel=c_out_sa1 + 256, mlp=[256, 128]) # 96 + 256 = 352
        # fp1: l0_points(None) + interpolated l1_points(fp2_out=128) = 128
        self.fp1 = PointNetFeaturePropagation(in_channel=128, mlp=[128, 128, 128]) # 输入只有来自上一层 FP 的 128 维

        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz):
        l0_points = xyz
        l0_xyz = xyz[:,:3,:]

        # 注意：sa1 的输入 points 是 l0_points (包含所有特征 C 维)
        #       sa1 内部计算时 new_points = concat(relative_xyz(3), grouped_points(C)) -> C+3 维送入第一个 MLP
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points_out = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points_out))))
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x, None


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, class_weights=None, ignore_index=None):
        """
        计算 NLL Loss

        Args:
            pred (torch.Tensor): 模型预测的 Log-Probabilities, 形状 [B*N, C] 或 [B, C, N]
            target (torch.Tensor): 真实标签, 形状 [B*N] 或 [B, N]
            class_weights (torch.Tensor, optional): 类别的权重, 形状 [num_classes]。默认为 None。
            ignore_index (int, optional): 指定 target 中要忽略的标签索引。默认为 None。

        Returns:
            torch.Tensor: 计算得到的 loss (标量)
        """
        total_loss = F.nll_loss(pred, target, weight=class_weights, ignore_index=ignore_index if ignore_index is not None else -100)
        return total_loss