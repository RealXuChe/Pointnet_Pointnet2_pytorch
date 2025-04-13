# test_semseg.py (Modified for SemanticKITTI with Voting & Visualization)
import argparse
import os
import yaml # For loading KITTI config
from data_utils.KITTIDataLoader import SemanticKittiDataset # Use KITTI Loader
# Removed: from data_utils.S3DISDataLoader import ScannetDatasetWholeScene
# Removed: from data_utils.indoor3d_util import g_label2color
import torch
import logging
from pathlib import Path
import sys
import importlib
from tqdm import tqdm
import numpy as np
# Removed: import provider

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

# Removed: Hardcoded S3DIS classes and labels

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Model Evaluation')
    # --- Essential Args ---
    parser.add_argument('--gpu', type=str, default='0', help='Specify gpu device')
    parser.add_argument('--log_dir', type=str, required=True, help='Experiment root directory (e.g., YYYY-MM-DD_HH-MM)')
    parser.add_argument('--model', type=str, required=True, help='Model name to load (e.g., pointnet2_sem_seg)')

    # --- Data Args ---
    parser.add_argument('--data_root', type=str, required=True, help='Root directory of SemanticKITTI dataset')
    parser.add_argument('--config_path', type=str, default='kitti_config/semantic-kitti.yaml', help='Path to semantic-kitti.yaml config file')
    parser.add_argument('--test_sequences', nargs='+', type=str, required=True, help='Specify test sequences (e.g., 08 09)')
    parser.add_argument('--split', type=str, default='valid', choices=['train', 'valid', 'test'], help='Dataset split corresponding to test_sequences [default: valid]')
    parser.add_argument('--num_point', type=int, default=65536, help='Number of points to sample per vote [default: 65536, must match training]')
    parser.add_argument('--use_intensity', action='store_true', default=False, help='Use intensity as a feature (must match training)')
    parser.add_argument('--ignore_index', type=int, default=0, help='Label index to ignore (must match training) [default: 0]')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading [default: 4]') # Used by KITTIDataLoader instance, not directly here

    # --- Evaluation Args ---
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for inference within voting loop [default: 16]') # Used if processing samples in batches
    parser.add_argument('--num_votes', type=int, default=5, help='Number of votes per scan [default: 5]')
    parser.add_argument('--visual', action='store_true', default=False, help='Visualize result by saving obj files [default: False]')

    # Removed: --test_area
    return parser.parse_args()

# Removed: add_vote function (logic moved inline)

def main(args):
    def log_string(str_):
        logger.info(str_)
        print(str_)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    experiment_dir = Path('./log/sem_seg') / args.log_dir # Use Path object
    if not experiment_dir.exists():
         log_string(f"Error: Log directory {experiment_dir} not found!")
         sys.exit(1)

    visual_dir = experiment_dir / 'visual/'
    visual_dir.mkdir(exist_ok=True)

    '''LOG'''
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = experiment_dir / 'eval.txt'
    file_handler = logging.FileHandler(str(log_file))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(f"Args: {args}") # Log all args

    # --- Load KITTI Config for labels/colors ---
    try:
        kitti_cfg = yaml.safe_load(open(args.config_path, 'r'))
        log_string(f"Loaded KITTI config from: {args.config_path}")
    except Exception as e:
        log_string(f"Error loading KITTI config {args.config_path}: {e}")
        sys.exit(1)

    # --- Dataset Loading ---
    log_string("Start loading evaluation dataset...")
    # We instantiate the dataset mainly to access its properties and file lists
    # The actual data loading for evaluation will happen manually inside the loop
    TEST_DATASET = SemanticKittiDataset(
        root_dir=args.data_root,
        config_path=args.config_path,
        split=args.split, # Use the split argument
        sequences=args.test_sequences, # Use the sequence argument
        num_point=args.num_point, # Needed for dataset init, though we sample manually later
        use_intensity=args.use_intensity,
        remap_labels=True, # Evaluation needs same remapping
        transform=None
    )
    log_string(f"The number of scans to evaluate is: {len(TEST_DATASET)}")

    # --- Get dataset info ---
    NUM_CLASSES = TEST_DATASET.num_classes
    seg_label_to_cat = TEST_DATASET.seg_label_to_cat # Dynamically get names
    kitti_color_map = kitti_cfg.get('color_map', {})   # Maps ORIGINAL labels to colors
    learning_map_inv = kitti_cfg.get('learning_map_inv', {}) # Maps LEARNING labels back to ORIGINAL
    log_string(f"Number of classes: {NUM_CLASSES}")
    assert 0 <= args.ignore_index < NUM_CLASSES, f"ignore_index ({args.ignore_index}) must be within class range [0, {NUM_CLASSES-1}]"
    log_string(f"Using ignore index: {args.ignore_index} ({seg_label_to_cat.get(args.ignore_index, 'Unknown')})")
    if not kitti_color_map:
        log_string("Warning: KITTI color map not found in config, visualization might use defaults.")
    if not learning_map_inv:
        log_string("Warning: KITTI inverse learning map not found in config, visualization might fail.")
        learning_map_inv = {i: i for i in range(NUM_CLASSES)} # Basic fallback

    # --- Derived parameters ---
    NUM_POINT_PER_VOTE = args.num_point
    BATCH_SIZE = args.batch_size # How many points to process at once during inference if needed (not used in current simple voting loop)

    '''MODEL LOADING'''
    log_string(f"Loading model: {args.model}")
    MODEL = importlib.import_module(args.model)
    input_channels = 4 if args.use_intensity else 3
    classifier = MODEL.get_model(num_classes=NUM_CLASSES, input_channels=input_channels).cuda()

    checkpoint_path = experiment_dir / 'checkpoints/best_model.pth'
    if not checkpoint_path.exists():
        log_string(f"Error: Checkpoint {checkpoint_path} not found!")
        sys.exit(1)

    checkpoint = torch.load(str(checkpoint_path), weights_only=False)
    try:
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string(f"Loaded model state dict from epoch {checkpoint.get('epoch', 'N/A')}")
    except Exception as e:
        log_string(f"Error loading state dict: {e}")
        # Attempt to load with strict=False if necessary, though it might indicate issues
        try:
            log_string("Attempting to load with strict=False")
            classifier.load_state_dict(checkpoint['model_state_dict'], strict=False)
        except Exception as e2:
            log_string(f"Error loading state dict even with strict=False: {e2}")
            sys.exit(1)

    classifier = classifier.eval()
    log_string("Model loaded and set to evaluation mode.")


    with torch.no_grad():
        num_scans = len(TEST_DATASET)

        # --- Initialize Global Metrics ---
        total_correct_global = 0
        total_seen_global = 0 # Counts valid points (not ignored)
        total_seen_class_global = np.zeros(NUM_CLASSES, dtype=np.int64)
        total_correct_class_global = np.zeros(NUM_CLASSES, dtype=np.int64)
        total_iou_deno_class_global = np.zeros(NUM_CLASSES, dtype=np.int64)

        log_string('---- EVALUATION WITH VOTING ----')

        # Loop through each scan file in the dataset
        for scan_idx in range(num_scans):
            scan_basename = os.path.splitext(os.path.basename(TEST_DATASET.scan_files[scan_idx]))[0]
            log_string(f"Processing scan [{scan_idx + 1}/{num_scans}]: {scan_basename}")

            # --- Load Full Scan Data for this Scan ---
            scan_file = TEST_DATASET.scan_files[scan_idx]
            label_file = TEST_DATASET.label_files[scan_idx]
            try:
                # Use the dataset's parser instance to load
                TEST_DATASET.scan_parser.open_scan(scan_file)
                TEST_DATASET.scan_parser.open_label(label_file)
                whole_scene_points_xyz = TEST_DATASET.scan_parser.points # Nx3
                whole_scene_remissions = TEST_DATASET.scan_parser.remissions # N,
                whole_scene_labels_raw = TEST_DATASET.scan_parser.sem_label # N, (original labels)
                N_total = whole_scene_points_xyz.shape[0]

                # Remap the full ground truth labels
                if TEST_DATASET.remap_labels:
                    max_raw_label = np.max(whole_scene_labels_raw)
                    if max_raw_label >= len(TEST_DATASET.learning_map_lut):
                        print(f"Warning: Scan {scan_basename} has out-of-bounds raw labels (max={max_raw_label}). Clamping.")
                        valid_mask_remap = whole_scene_labels_raw < len(TEST_DATASET.learning_map_lut)
                        whole_scene_labels = np.zeros_like(whole_scene_labels_raw, dtype=np.int64)
                        whole_scene_labels[valid_mask_remap] = TEST_DATASET.learning_map_lut[whole_scene_labels_raw[valid_mask_remap]]
                    else:
                        whole_scene_labels = TEST_DATASET.learning_map_lut[whole_scene_labels_raw].astype(np.int64)
                else:
                     whole_scene_labels = whole_scene_labels_raw.astype(np.int64) # Should be remapped for eval

                # Combine features based on args.use_intensity
                if args.use_intensity:
                    whole_scene_data = np.hstack((whole_scene_points_xyz, whole_scene_remissions.reshape(-1, 1))) # N x 4
                else:
                    whole_scene_data = whole_scene_points_xyz # N x 3

            except Exception as e:
                log_string(f"Error loading scan {scan_basename} ({scan_file}): {e}. Skipping.")
                continue

            # --- Voting Loop ---
            vote_label_pool = np.zeros((N_total, NUM_CLASSES), dtype=np.float32) # Use float for potential weighted voting later
            for _ in tqdm(range(args.num_votes), desc=f"Voting on {scan_basename}", leave=False):
                # Randomly sample indices for this vote
                replace_sample = N_total < NUM_POINT_PER_VOTE
                if replace_sample:
                    log_string(f"Warning: Scan {scan_basename} has {N_total} points, less than num_point {NUM_POINT_PER_VOTE}. Using replacement.")
                current_indices = np.random.choice(N_total, NUM_POINT_PER_VOTE, replace=replace_sample)

                # Get points for the current sample
                current_points_features = whole_scene_data[current_indices, :] # [N_sample, C]

                # Prepare for PyTorch model
                points_tensor = torch.from_numpy(current_points_features).float().unsqueeze(0) # [1, N_sample, C]
                points_tensor = points_tensor.cuda()
                points_tensor = points_tensor.transpose(2, 1) # Model expects [B, C, N] -> [1, C, N_sample]

                # Inference
                seg_pred, _ = classifier(points_tensor) # Output: [1, N_sample, num_classes] (log-softmax)
                pred_logits_or_logprobs = seg_pred.squeeze(0).cpu().data # [N_sample, num_classes]

                # Get predicted labels for the sample
                # We vote with the predicted label index, not probabilities here
                pred_sampled_labels = pred_logits_or_logprobs.max(1)[1].numpy() # [N_sample]

                # Add votes to the pool using original indices
                for i in range(NUM_POINT_PER_VOTE):
                    original_idx = current_indices[i]
                    predicted_label = pred_sampled_labels[i]
                    vote_label_pool[original_idx, predicted_label] += 1 # Simple count voting

            # Final prediction for the scan based on votes
            pred_label_scan = np.argmax(vote_label_pool, axis=1) # [N_total]

            # --- Calculate Metrics for this Scan ---
            correct_scan = 0
            seen_scan = 0
            total_seen_class_tmp = np.zeros(NUM_CLASSES, dtype=np.int64)
            total_correct_class_tmp = np.zeros(NUM_CLASSES, dtype=np.int64)
            total_iou_deno_class_tmp = np.zeros(NUM_CLASSES, dtype=np.int64)

            valid_mask_scan = (whole_scene_labels != args.ignore_index) # [N_total]

            correct_scan = np.sum((pred_label_scan[valid_mask_scan] == whole_scene_labels[valid_mask_scan]))
            seen_scan = np.sum(valid_mask_scan)

            for l in range(NUM_CLASSES):
                if l == args.ignore_index:
                    continue
                label_mask_l = (whole_scene_labels == l) & valid_mask_scan
                pred_mask_l = (pred_label_scan == l) & valid_mask_scan

                total_seen_class_tmp[l] = np.sum(label_mask_l)
                total_correct_class_tmp[l] = np.sum(label_mask_l & pred_mask_l)
                total_iou_deno_class_tmp[l] = np.sum(label_mask_l | pred_mask_l)

            # --- Accumulate Global Metrics ---
            total_correct_global += correct_scan
            total_seen_global += seen_scan
            total_seen_class_global += total_seen_class_tmp
            total_correct_class_global += total_correct_class_tmp
            total_iou_deno_class_global += total_iou_deno_class_tmp

            # Log per-scan IoU (optional)
            iou_map_scan = np.zeros(NUM_CLASSES)
            present_classes_mask = total_iou_deno_class_tmp > 0
            iou_map_scan[present_classes_mask] = total_correct_class_tmp[present_classes_mask] / total_iou_deno_class_tmp[present_classes_mask]
            # Calculate mIoU for this scan (ignoring ignore_index and absent classes)
            valid_iou_mask = (np.arange(NUM_CLASSES) != args.ignore_index) & (total_seen_class_tmp > 0)
            mIoU_scan = np.mean(iou_map_scan[valid_iou_mask]) if np.any(valid_iou_mask) else 0.0
            log_string(f'Scan {scan_basename}: mIoU={mIoU_scan:.4f}, Acc={(correct_scan / seen_scan if seen_scan > 0 else 0.0):.4f}')
            # print(f"  IoU per class: { {seg_label_to_cat.get(l,'?'): iou_map_scan[l] for l in range(NUM_CLASSES) if l != args.ignore_index} }")

            # --- Visualization Output ---
            pred_output_file = visual_dir / (scan_basename + '_pred.txt')
            try:
                 # Save predicted labels (learning IDs) to text file
                 np.savetxt(str(pred_output_file), pred_label_scan, fmt='%d')
                 # log_string(f"Saved predictions to {pred_output_file}")
            except Exception as e:
                 log_string(f"Error saving prediction file {pred_output_file}: {e}")


            if args.visual:
                obj_pred_file = visual_dir / (scan_basename + '_pred.obj')
                obj_gt_file = visual_dir / (scan_basename + '_gt.obj')
                try:
                    fout_pred = open(str(obj_pred_file), 'w')
                    fout_gt = open(str(obj_gt_file), 'w')

                    for i in range(N_total):
                        # Map learning label back to original label for color lookup
                        predicted_learning_label = pred_label_scan[i]
                        gt_learning_label = whole_scene_labels[i]

                        original_pred_label = learning_map_inv.get(predicted_learning_label, 0) # Default to 0 (unlabeled)
                        original_gt_label = learning_map_inv.get(gt_learning_label, 0)

                        # Get color from KITTI map, default to black [0,0,0] if not found
                        pred_color_raw = kitti_color_map.get(original_pred_label, [0, 0, 0])
                        gt_color_raw = kitti_color_map.get(original_gt_label, [0, 0, 0])

                        # Ensure color values are integers 0-255
                        pred_color = [int(c * 255) if isinstance(c, float) and c <= 1.0 else int(c) for c in pred_color_raw]
                        gt_color = [int(c * 255) if isinstance(c, float) and c <= 1.0 else int(c) for c in gt_color_raw]

                        # Write vertex line: v x y z r g b
                        fout_pred.write('v %f %f %f %d %d %d\n' % (
                            whole_scene_points_xyz[i, 0], whole_scene_points_xyz[i, 1], whole_scene_points_xyz[i, 2],
                            pred_color[0], pred_color[1], pred_color[2]))
                        fout_gt.write('v %f %f %f %d %d %d\n' % (
                            whole_scene_points_xyz[i, 0], whole_scene_points_xyz[i, 1], whole_scene_points_xyz[i, 2],
                            gt_color[0], gt_color[1], gt_color[2]))

                    fout_pred.close()
                    fout_gt.close()
                    # log_string(f"Saved visualization OBJ files for {scan_basename}")

                except Exception as e:
                    log_string(f"Error during visualization for scan {scan_basename}: {e}")
                    if 'fout_pred' in locals() and not fout_pred.closed: fout_pred.close()
                    if 'fout_gt' in locals() and not fout_gt.closed: fout_gt.close()


        # --- Calculate Final Global Metrics ---
        log_string('---- FINAL EVALUATION RESULTS ----')
        iou_per_class = np.zeros(NUM_CLASSES, dtype=np.float32)
        acc_per_class = np.zeros(NUM_CLASSES, dtype=np.float32)
        valid_classes_count = 0 # Counts classes present in the ground truth across all scans

        # Calculate IoU and Acc per class
        for l in range(NUM_CLASSES):
            if l == args.ignore_index:
                iou_per_class[l] = np.nan
                acc_per_class[l] = np.nan
                continue

            if total_iou_deno_class_global[l] > 0:
                iou_per_class[l] = total_correct_class_global[l] / float(total_iou_deno_class_global[l])
            else:
                iou_per_class[l] = 0.0 # Or np.nan

            if total_seen_class_global[l] > 0:
                acc_per_class[l] = total_correct_class_global[l] / float(total_seen_class_global[l])
                valid_classes_count += 1 # Count this class as valid for mIoU/mAcc calculation
            else:
                 acc_per_class[l] = 0.0 # Or np.nan

        # Calculate means, ignoring NaN values (from ignore_index) and absent classes
        mIoU = np.nansum(iou_per_class) / valid_classes_count if valid_classes_count > 0 else 0.0
        mAcc = np.nansum(acc_per_class) / valid_classes_count if valid_classes_count > 0 else 0.0
        overall_acc = total_correct_global / float(total_seen_global) if total_seen_global > 0 else 0.0

        # --- Log Final Results ---
        log_string(f'Evaluation finished after {num_scans} scans.')
        log_string(f'Overall Accuracy (valid points): {overall_acc:.6f}')
        log_string(f'Mean Accuracy (over valid classes): {mAcc:.6f}')
        log_string(f'Mean IoU (over valid classes): {mIoU:.6f}')

        iou_per_class_str = '------- IoU Per Class --------\n'
        for l in range(NUM_CLASSES):
            if l == args.ignore_index: continue
            cat_name = seg_label_to_cat.get(l, f'Class_{l}')
            iou_val = iou_per_class[l]
            # Format NaN nicely
            iou_str = f'{iou_val:.4f}' if not np.isnan(iou_val) else 'NaN'
            # Include TP / (TP+FP+FN) counts for detail
            iou_counts_str = f'({total_correct_class_global[l]}/{total_iou_deno_class_global[l]})'
            iou_per_class_str += f'  {cat_name:<15}: {iou_str} {iou_counts_str}\n'
        log_string(iou_per_class_str)

        log_string("Evaluation Done!")


if __name__ == '__main__':
    args = parse_args()
    main(args)