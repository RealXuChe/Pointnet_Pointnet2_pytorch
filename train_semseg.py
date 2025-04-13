import argparse
import os
from data_utils.KITTIDataLoader import SemanticKittiDataset, worker_init_fn
import torch
import datetime
import logging
from pathlib import Path
import sys
import importlib
import shutil
from tqdm import tqdm
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


logger = logging.getLogger("Model")
logger.setLevel(logging.INFO)

class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnet_sem_seg', help='model name [default: pointnet_sem_seg]')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
    parser.add_argument('--epoch', default=32, type=int, help='Epoch to run [default: 32]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='Adam, AdamW or SGD [default: AdamW]')
    parser.add_argument('--log_dir', type=str, default=None, help='Log path [default: None, creates YYYY-MM-DD_HH-MM]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='Weight decay [default: 1e-4]')
    parser.add_argument('--npoint', type=int, default=65536, help='Number of points to sample [default: 65536]')
    parser.add_argument('--step_size', type=int, default=10, help='Decay step for lr decay [default: every 10 epochs]')
    parser.add_argument('--lr_decay', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')

    # --- KITTI Specific Args ---
    parser.add_argument('--data_root', type=str, required=True, help='Root directory of SemanticKITTI dataset')
    parser.add_argument('--config_path', type=str, default='kitti_config/semantic-kitti.yaml', help='Path to semantic-kitti.yaml config file')
    parser.add_argument('--train_sequences', nargs='+', type=str, default=None, help='Specify training sequences (e.g., 00 01 02...) to override config')
    parser.add_argument('--val_sequences', nargs='+', type=str, default=None, help='Specify validation sequences (e.g., 08) to override config')
    # parser.add_argument('--test_sequences', nargs='+', type=str, default=None, help='Specify test sequences (e.g., 11 12...) to override config') # 如果需要测试集推理
    parser.add_argument('--use_intensity', action='store_true', help='Use intensity as a feature')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading [default: 4]')
    parser.add_argument('--ignore_index', type=int, default=0, help='Label index to ignore during loss calculation and evaluation [default: 0]')

    return parser.parse_args()


def log_string(str):
    logger.info(str)
    print(str)

def new_optimizer(model, optimizer_type, learning_rate, decay_rate):
    """
    Create a new optimizer based on the specified type
    
    Args:
        model: The model whose parameters will be optimized
        optimizer_type: Type of optimizer ('AdamW', 'Adam', or 'SGD')
        learning_rate: Learning rate for the optimizer
        decay_rate: Weight decay rate
        
    Returns:
        The created optimizer
    """
    log_string(f"Creating optimizer: {optimizer_type}")
    if optimizer_type == 'AdamW':  # 推荐使用 AdamW
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),  # 可以调整
            eps=1e-08,
            weight_decay=decay_rate  # AdamW 直接处理权重衰减
        )
    elif optimizer_type == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=decay_rate  # Adam 的 weight decay 不同于 AdamW
        )
    else:  # SGD
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,  # SGD 通常需要 momentum
            weight_decay=decay_rate
        )
    return optimizer


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1: # 同时处理 Conv1d 和 Conv2d
        if hasattr(m, 'weight') and m.weight is not None:
            torch.nn.init.xavier_normal_(m.weight.data)
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            torch.nn.init.xavier_normal_(m.weight.data)
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            torch.nn.init.constant_(m.weight.data, 1.0) # 通常初始化为 1
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0) # 通常初始化为 0



def bn_momentum_adjust(m, momentum):
    if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
        m.momentum = momentum
        
        
def log_parameters(args):
    log_string('PARAMETER SETTINGS')
    log_string('=' * 40)
    param_str = ''
    for key, value in vars(args).items():
        param_str += f"{key.ljust(20)}: {value}\n"
    log_string(param_str[:-1])  # 去掉最后一个换行符
    log_string('=' * 40)

def main(args):
    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('sem_seg')
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    log_parameters(args)

    NUM_POINT = args.npoint
    BATCH_SIZE = args.batch_size

    # --- 数据集加载 ---
    print("Start loading training data ...")
    TRAIN_DATASET = SemanticKittiDataset(
        root_dir=args.data_root,
        config_path=args.config_path,
        split='train',
        sequences=args.train_sequences, # 使用命令行参数覆盖
        num_point=args.npoint,
        use_intensity=args.use_intensity,
        remap_labels=True, # 通常训练时需要重映射
        transform=None # 可以后续添加数据增强 transform
    )

    print("Start loading validation data ...")
    VAL_DATASET = SemanticKittiDataset(
        root_dir=args.data_root,
        config_path=args.config_path,
        split='valid',
        sequences=args.val_sequences, # 使用命令行参数覆盖
        num_point=args.npoint,
        use_intensity=args.use_intensity,
        remap_labels=True, # 验证时也需要重映射到相同的标签空间
        transform=None
    )
    
    # --- 获取数据集信息 ---
    NUM_CLASSES = TRAIN_DATASET.num_classes
    seg_label_to_cat = TRAIN_DATASET.seg_label_to_cat # 获取标签到名称的映射
    log_string(f"Number of classes: {NUM_CLASSES}")

    assert 0 <= args.ignore_index < NUM_CLASSES, f"ignore_index ({args.ignore_index}) 必须在类别范围 [0, {NUM_CLASSES-1}] 内"
    ignore_index = args.ignore_index
    log_string(f"Using ignore index: {ignore_index} ({seg_label_to_cat.get(ignore_index, 'Unknown')})")

    # --- 创建 DataLoader ---
    trainDataLoader = MultiEpochsDataLoader(
        TRAIN_DATASET,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_init_fn
    )
    valDataLoader = MultiEpochsDataLoader(
        VAL_DATASET,
        batch_size=args.batch_size, # 验证时 batch size 可以不同
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False # 验证集通常不丢弃
    )
    
    # 如果需要类别权重，需要单独计算，例如：
    weights = None # 暂时不使用权重
    # weights = compute_class_weights(TRAIN_DATASET, NUM_CLASSES, ignore_index).cuda() # 需要实现 compute_class_weights 函数
    log_string(f"Class weights: {'Not used' if weights is None else 'Computed'}")

    log_string(f"The number of training data is: {len(TRAIN_DATASET)}")
    log_string(f"The number of validation data is: {len(VAL_DATASET)}")

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    shutil.copy('models/%s.py' % args.model, str(experiment_dir))
    shutil.copy('models/pointnet2_utils.py', str(experiment_dir))

    input_channels = 4 if args.use_intensity else 3
    log_string(f"Model input channels: {input_channels}")

    # 实例化模型
    classifier = MODEL.get_model(num_classes=NUM_CLASSES, input_channels=input_channels).cuda()

    # 获取损失函数类并实例化
    LossClass = MODEL.get_loss
    criterion = LossClass().cuda() # 实例化时不传入 ignore_index 或 weights

    classifier.apply(inplace_relu) # 应用 inplace ReLU (可选)

    optimizer = None
    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
        # 加载优化器状态 (如果需要继续之前的优化过程)
        if 'optimizer_state_dict' in checkpoint and args.optimizer == checkpoint.get('optimizer_type', args.optimizer): # 检查优化器类型是否匹配
            optimizer = new_optimizer(classifier, args.optimizer, args.learning_rate, args.decay_rate)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            log_string('Optimizer state loaded.')
        else:
            log_string('Optimizer state not loaded (not found or type mismatch). Creating new optimizer.')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0
        classifier = classifier.apply(weights_init)

    # 创建优化器 (如果尚未从 checkpoint 加载)
    if optimizer is None:
        optimizer = new_optimizer(classifier, args.optimizer, args.learning_rate, args.decay_rate)

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    global_epoch = 0
    best_iou = 0

    for epoch in range(start_epoch, args.epoch):
        '''Train on scenes'''
        log_string('**** Epoch %d (%d/%s) ****' % (global_epoch + 1, epoch + 1, args.epoch))
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log_string('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
        num_batches = len(trainDataLoader)
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        classifier = classifier.train()

        for i, (points, target) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.2, desc=f"Train Epoch {epoch+1}"):
            optimizer.zero_grad()

            assert points.dtype == torch.float32, "points should be float32"
            assert target.dtype == torch.long, "target should be long"

            points, target = points.cuda(), target.cuda()
            points = points.transpose(2, 1) # [B, C, N]

            # 模型前向传播 (忽略第二个返回值)
            # seg_pred 形状: [B, N, num_classes], 是 log_softmax 输出
            seg_pred, _ = classifier(points)
            # Reshape 用于 NLLLoss: [B*N, num_classes]
            seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)

            # Reshape target 用于 NLLLoss: [B*N]
            target = target.view(-1)

            # 计算损失，传入 ignore_index 和 weights (如果计算了)
            loss = criterion(seg_pred, target, class_weights=weights, ignore_index=ignore_index)

            loss.backward()
            optimizer.step()

            # --- 训练准确率计算 (包含忽略点) ---
            # 从 log_softmax 输出获取预测类别
            pred_choice = seg_pred.cpu().data.max(1)[1].numpy() # [B*N]
            # 获取 batch 标签 (需要 target 在 reshape 前的副本，或者用 reshape 后的 target)
            batch_label = target.cpu().data.numpy() # [B*N]
            correct = np.sum(pred_choice == batch_label) # 计算所有点的正确预测数
            total_correct += correct
            total_seen += (BATCH_SIZE * NUM_POINT) # BATCH_SIZE * NUM_POINT = B*N，总点数
            loss_sum += loss.item() # 使用 .item() 获取标量值

        log_string('Training mean loss: %f' % (loss_sum / num_batches))
        # 注意：此准确率包含 ignore_index 的点，仅供参考
        log_string('Training accuracy (all points): %f' % (total_correct / float(total_seen)))

        if epoch % 5 == 0 or epoch == args.epoch - 1: # 每 5 个 epoch 或最后一个 epoch 保存
            logger.info('Save model...')
            savepath = checkpoints_dir / f'model_epoch_{epoch}.pth' # 可以包含 epoch 数
            log_string(f'Saving intermediate model at {savepath}')
            state = {
                'epoch': epoch,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'optimizer_type': args.optimizer, # 保存优化器类型
            }
            torch.save(state, savepath)
            log_string('Saving model....')

        '''Evaluate on validation set'''
        with torch.no_grad(): # 确保没有梯度计算
            num_batches = len(valDataLoader)
            total_correct = 0
            total_seen = 0 # 只统计有效点 (非 ignore_index)
            loss_sum = 0
            total_seen_class = [0 for _ in range(NUM_CLASSES)]
            total_correct_class = [0 for _ in range(NUM_CLASSES)]
            total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]
            classifier = classifier.eval() # 设置为评估模式

            log_string('---- EPOCH %03d VALIDATION ----' % (global_epoch + 1))
            for i, (points, target) in tqdm(enumerate(valDataLoader), total=len(valDataLoader), smoothing=0.9, desc=f"Val Epoch {epoch+1}"):
                # points 和 target 已经是 Tensor，不需要 .data.numpy() 再转回
                points, target = points.float().cuda(), target.long().cuda()
                points = points.transpose(2, 1) # [B, C, N]

                # 模型前向传播
                seg_pred, _ = classifier(points) # [B, N, num_classes]
                # 用于损失计算的 reshape
                seg_pred_loss = seg_pred.contiguous().view(-1, NUM_CLASSES) # [B*N, num_classes]
                target_loss = target.view(-1) # [B*N]

                # 计算损失
                loss = criterion(seg_pred_loss, target_loss, class_weights=weights, ignore_index=ignore_index)
                loss_sum += loss.item()

                # --- 指标计算 ---
                # 获取预测类别 [B, N]
                pred_val = seg_pred.cpu().data.numpy() # [B, N, C]
                pred_choice = np.argmax(pred_val, 2) # [B, N]
                # 获取真实标签 [B, N]
                batch_label = target.cpu().data.numpy() # [B, N]

                # 创建有效点掩码 (不等于 ignore_index)
                valid_mask = (batch_label != ignore_index) # [B, N]

                # 计算有效点的正确预测数
                correct = np.sum((pred_choice[valid_mask] == batch_label[valid_mask]))
                total_correct += correct
                total_seen += np.sum(valid_mask) # 累加有效点的数量

                # 计算每个类别的指标 (忽略 ignore_index)
                for l in range(NUM_CLASSES):
                    if l == ignore_index:
                        continue # 跳过忽略类别
                    # 找到真实标签为 l 且有效的点
                    label_mask_l = (batch_label == l) & valid_mask
                    # 找到预测标签为 l 且有效的点
                    pred_mask_l = (pred_choice == l) & valid_mask

                    total_seen_class[l] += np.sum(label_mask_l) # TP + FN for class l
                    total_correct_class[l] += np.sum(label_mask_l & pred_mask_l) # TP for class l
                    total_iou_deno_class[l] += np.sum(label_mask_l | pred_mask_l) # TP + FP + FN for class l

            # --- 计算 mIoU 和 mAcc (忽略 ignore_index) ---
            iou_per_class = np.zeros(NUM_CLASSES, dtype=np.float32)
            acc_per_class = np.zeros(NUM_CLASSES, dtype=np.float32)
            valid_classes_count = 0 # 统计参与计算的有效类别数

            for l in range(NUM_CLASSES):
                if l == ignore_index:
                    iou_per_class[l] = np.nan # 标记为 NaN
                    acc_per_class[l] = np.nan # 标记为 NaN
                    continue

                if total_iou_deno_class[l] > 0:
                    iou_per_class[l] = total_correct_class[l] / float(total_iou_deno_class[l])
                else:
                    iou_per_class[l] = 0.0 # 如果该类别的并集为0，IoU为0

                if total_seen_class[l] > 0:
                    acc_per_class[l] = total_correct_class[l] / float(total_seen_class[l])
                    # 只有当这个类别实际出现过，才算作有效类别参与 mAcc 和 mIoU 计算
                    valid_classes_count += 1
                else:
                    acc_per_class[l] = 0.0 # 如果该类别从未出现，准确率为0

            # 使用 np.nansum 忽略 NaN 值计算总和，然后除以有效类别数
            mIoU = np.nansum(iou_per_class) / valid_classes_count if valid_classes_count > 0 else 0.0
            mAcc = np.nansum(acc_per_class) / valid_classes_count if valid_classes_count > 0 else 0.0

            # 记录评估日志
            log_string('eval mean loss: %f' % (loss_sum / float(num_batches)))
            log_string('eval point avg class IoU: %f' % mIoU)
            log_string('eval point accuracy (valid points): %f' % (total_correct / float(total_seen) if total_seen > 0 else 0.0))
            log_string('eval point avg class acc: %f' % mAcc)

            # 打印每类 IoU (跳过 ignore_index)
            iou_per_class_str = '------- IoU Per Class --------\n'
            for l in range(NUM_CLASSES):
                if l == ignore_index: continue
                cat_name = seg_label_to_cat.get(l, f'Class_{l}')
                # 移除 weight 显示
                iou_per_class_str += 'class %s: %.4f \n' % (
                    cat_name + ' ' * (14 - len(cat_name)), iou_per_class[l])
            log_string(iou_per_class_str)

            # 保存最佳模型 (基于 mIoU)
            if mIoU >= best_iou:
                best_iou = mIoU
                logger.info('Save best model...')
                savepath = checkpoints_dir / 'best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': epoch,
                    'class_avg_iou': mIoU, # 保存 mIoU 值
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'optimizer_type': args.optimizer, # 保存优化器类型
                }
                torch.save(state, savepath)
                log_string('Saving model....')
            log_string('Best mIoU: %f' % best_iou)
            
            
        global_epoch += 1


if __name__ == '__main__':
    args = parse_args()
    main(args)
