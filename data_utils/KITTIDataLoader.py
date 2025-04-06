import os
import yaml
import numpy as np
import random
import torch
from torch.utils.data import Dataset
import time
import fpsample

from .auxiliary.laserscan import SemLaserScan

class SemanticKittiDataset(Dataset):
    """
    用于加载 SemanticKITTI 或类似格式数据集的 PyTorch Dataset 类。
    处理 .bin 点云文件和 .label 标签文件。
    """
    def __init__(self,
                 root_dir='/home/dy/FYProject/dataset/extracted/dataset', 
                 config_path='kitti_config/semantic-kitti.yaml',
                 split='train',
                 sequences=None, # 可选，用于覆盖 config 文件中的 split 定义
                 num_point=4096,
                 use_intensity=True,
                 transform=None,
                 remap_labels=True):
        """
        初始化数据集加载器。

        Args:
            root_dir (str): 数据集根目录。
            config_path (str): dataset-config.yaml 文件的路径。
            split (str): 'train', 'valid', 或 'test'。将使用 config 文件中定义的序列。
            sequences (list[str], optional): 如果提供，将覆盖 config 文件中 'split' 定义的序列列表。
            num_point (int): 每个样本采样到的点数。
            use_intensity (bool): 是否将强度 (remission) 作为第四维特征。
            transform (callable, optional): 应用于样本(点云, 标签)的可选转换函数。
            remap_labels (bool): 是否使用 config 文件中的 learning_map 转换标签。
        """
        self.root_dir = root_dir
        self.config_path = config_path
        self.split = split
        self.req_sequences = sequences
        self.num_point = num_point
        self.use_intensity = use_intensity
        self.transform = transform
        self.remap_labels = remap_labels

        # --- 1. 加载配置文件 ---
        self.CFG = yaml.safe_load(open(config_path, 'r'))

        # --- 2. 获取标签映射和类别数 ---
        self.learning_map_lut = None
        self.num_classes = 0
        if self.remap_labels:
            print(f"[{self.__class__.__name__}] Enabling label remapping.")
            # 创建从原始 ID 到学习 ID 的查找表 (LUT)
            learning_map_dict = self.CFG['learning_map']
            max_key = max(learning_map_dict.keys())
            self.learning_map_lut = np.zeros((max_key + 1), dtype=np.int32)
            for key, value in learning_map_dict.items():
                 # 确保 key 是非负整数
                 if isinstance(key, int) and key >= 0:
                    self.learning_map_lut[key] = value
                 else:
                     print(f"警告: learning_map 中发现无效的 key: {key}，将被忽略。")

            # 从 learning_map_inv 推断类别数 (0 到 max_learning_id)
            self.num_classes = len(self.CFG['learning_map_inv'])
            print(f"[{self.__class__.__name__}] Number of classes (incl. ignore): {self.num_classes}")
        else:
            print(f"[{self.__class__.__name__}] Label remapping disabled.")
            # 如果不重映射，类别数可能需要手动设置或从 labels 推断 (可能不准确)
            # self.num_classes = len(self.CFG['labels']) # 这是一个粗略估计
            print("警告：未进行标签重映射，类别数未明确定义，可能导致问题！")


        # 获取颜色字典 (SemLaserScan 初始化需要)
        self.color_dict = self.CFG.get("color_map", {})
        if not self.color_dict:
             print(f"警告: 配置文件 {config_path} 中未找到 'color_map'。")


        # --- 3. 初始化 SemLaserScan (用于解析文件) ---
        # DataLoader 通常处理 3D 点，project=False 较合适
        print(f"[{self.__class__.__name__}] Initializing SemLaserScan (project=False).")
        self.scan_parser = SemLaserScan(self.color_dict, project=False)

        # --- 4. 确定要加载的序列 ---
        if self.req_sequences:
            print(f"[{self.__class__.__name__}] Using provided sequences: {self.req_sequences}")
            self.sequences = self.req_sequences
        elif self.split in self.CFG['split']:
            self.sequences = self.CFG['split'][self.split]
            print(f"[{self.__class__.__name__}] Using sequences from config for split '{self.split}': {self.sequences}")
        else:
            raise ValueError(f"错误：配置文件 {config_path} 中未定义 split '{self.split}' 或未提供 'sequences' 参数。")

        # --- 5. 查找所有相关的扫描和标签文件 ---
        self.scan_files = []
        self.label_files = []
        print(f"[{self.__class__.__name__}] Finding files for sequences: {self.sequences}...")
        for seq in self.sequences:
            # 格式化序列号为两位数（例如 0 -> "00", 8 -> "08"）
            seq_str = '{0:02d}'.format(int(seq))
            scan_path_seq = os.path.join(self.root_dir, "sequences", seq_str, "velodyne")
            label_path_seq = os.path.join(self.root_dir, "sequences", seq_str, "labels")

            if not os.path.isdir(scan_path_seq):
                print(f"警告：扫描路径不存在 {scan_path_seq}，跳过序列 {seq_str}")
                continue
            if not os.path.isdir(label_path_seq):
                 print(f"警告：标签路径不存在 {label_path_seq}，跳过序列 {seq_str}")
                 continue

            # 使用 os.scandir 获取文件路径
            try:
                seq_scan_files = sorted([f.path for f in os.scandir(scan_path_seq) if f.name.endswith('.bin')])
                seq_label_files = sorted([f.path for f in os.scandir(label_path_seq) if f.name.endswith('.label')])
            except Exception as e:
                print(f"错误：扫描序列 {seq_str} 中的文件时出错: {e}")
                continue

            # 检查并匹配文件
            if len(seq_scan_files) != len(seq_label_files):
                print(f"警告：序列 {seq_str} 的扫描 ({len(seq_scan_files)}) 和标签 ({len(seq_label_files)}) 文件数量不匹配！将尝试匹配。")
                scan_basenames = {os.path.splitext(os.path.basename(f))[0] for f in seq_scan_files}
                label_basenames = {os.path.splitext(os.path.basename(f))[0] for f in seq_label_files}
                common_basenames = sorted(list(scan_basenames.intersection(label_basenames)))

                seq_scan_files = [os.path.join(scan_path_seq, f"{name}.bin") for name in common_basenames]
                seq_label_files = [os.path.join(label_path_seq, f"{name}.label") for name in common_basenames]
                print(f"  匹配后，序列 {seq_str} 有 {len(common_basenames)} 个文件对。")

            self.scan_files.extend(seq_scan_files)
            self.label_files.extend(seq_label_files)

        if not self.scan_files:
             raise RuntimeError(f"错误：在指定路径和序列中未找到任何有效的 .bin/.label 文件对。检查 root_dir 和 sequences/split 设置。")

        print(f"[{self.__class__.__name__}] Found {len(self.scan_files)} scan/label pairs for split '{self.split}'.")
            
        # --- 6. 创建标签到类别的映射 ---
        self.seg_label_to_cat = {}
        learning_map_inv_dict = self.CFG['learning_map_inv']
        labels_dict = self.CFG['labels']
        # 遍历所有学习后的类别索引 (0 到 num_classes-1)
        for learning_idx in range(self.num_classes):
            original_label_id = learning_map_inv_dict[learning_idx]
            self.seg_label_to_cat[learning_idx] = labels_dict[original_label_id]

    def __len__(self):
        """返回数据集中的样本总数。"""
        return len(self.scan_files)

    def __getitem__(self, index):
        """
        获取指定索引的样本。

        Args:
            index (int): 样本索引。

        Returns:
            tuple: (points, labels)
                points (torch.Tensor): 点云数据 (形状: [num_point, D])，D=3 (XYZ) 或 4 (XYZI)。
                labels (torch.Tensor): 语义标签 (形状: [num_point])。
        """
        scan_path = self.scan_files[index]
        label_path = self.label_files[index]

        try:
            # --- 1. 加载扫描和标签 ---
            # **重要：先加载扫描，再加载标签** (因为 set_label 需要 self.points)
            self.scan_parser.open_scan(scan_path)
            self.scan_parser.open_label(label_path) # 会自动调用 set_label

            # --- 2. 获取原始数据 ---
            points_raw = self.scan_parser.points        # 获取 XYZ (N, 3)
            remissions_raw = self.scan_parser.remissions  # 获取 intensity (N,)
            labels_raw = self.scan_parser.sem_label     # 获取原始 semantic label (N,)
            
            N = points_raw.shape[0] # 获取原始点数
                    
        except Exception as e:
            print(f"错误：加载或解析文件失败: {scan_path} 或 {label_path}")
            print(f"  错误信息: {e}")
            # 返回 None 或 虚拟数据，让 collate_fn 处理 (如果配置了)
            # 或者更简单地，重新抛出异常
            print(f"  跳过索引 {index}")
            # 尝试加载下一个样本（如果这是在 DataLoader worker 中）
            # 返回一个虚拟样本可能导致后续处理问题，所以抛出异常更安全
            raise RuntimeError(f"无法加载索引 {index}") from e


        # --- 4. 标签重映射 (如果启用) ---
        if self.remap_labels:
            # 检查原始标签是否在 LUT 范围内
            max_raw_label = np.max(labels_raw)
            if max_raw_label >= len(self.learning_map_lut):
                # 处理超出范围的标签：通常映射到忽略类 0
                print(f"警告: 文件 {label_path} 中存在超出 learning_map LUT 范围的标签 ID ({max_raw_label})。将映射到 0。")
                valid_mask = labels_raw < len(self.learning_map_lut)
                labels_remapped = np.zeros_like(labels_raw) # 默认为 0
                labels_remapped[valid_mask] = self.learning_map_lut[labels_raw[valid_mask]]
            else:
                labels_remapped = self.learning_map_lut[labels_raw]
            labels = labels_remapped.astype(np.int64) # 使用 int64 匹配 torch.long()
        else:
            labels = labels_raw.astype(np.int64)

        # --- 5. 特征选择与组合 ---
        if self.use_intensity:
            # 将 (N, 3) 和 (N,) 合并为 (N, 4)
            points = np.hstack((points_raw, remissions_raw.reshape(-1, 1)))
        else:
            points = points_raw # 仅使用 XYZ (N, 3)

        # --- 6. 点采样/填充 ---
        if N >= self.num_point:
            # 点数足够多，进行 FPS 采样 
            choice_idx = fpsample.bucket_fps_kdline_sampling(points_raw, self.num_point, h=8)
        else:
            # 点数不足，进行填充 (重复采样)
            choice_idx = np.random.choice(N, self.num_point, replace=True)
            print(f"警告: 样本 {index} 只有 {N} 个点，少于要求的 {self.num_point}。将重复采样。")

        points_sampled = points[choice_idx, :]
        labels_sampled = labels[choice_idx]

        # --- 7. 转换为 PyTorch Tensors ---
        points_tensor = torch.from_numpy(points_sampled).float()
        labels_tensor = torch.from_numpy(labels_sampled).long() # CrossEntropyLoss 需要 Long 类型

        # --- 8. 数据增强 (如果提供了 transform) ---
        if self.transform:
            points_tensor, labels_tensor = self.transform(points_tensor, labels_tensor)

        return points_tensor, labels_tensor


def worker_init_fn(worker_id):
    """配合 DataLoader 使用，为每个 worker 设置不同的随机种子"""
    seed = int(time.time()) + worker_id
    np.random.seed(seed)
    random.seed(seed)