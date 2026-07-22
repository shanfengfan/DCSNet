# -*- coding: utf-8 -*-
import torch
import matplotlib.pyplot as plt
import random
import swanlab
import os
import numpy as np
from torch.utils.data import DataLoader
from net import FasterRCNNUNet
import torch.nn.functional as F
import cv2
from dataset import create_train_val_test_datasets
from tqdm import tqdm

# Check if FLOPs calculation tools are available (for calculate_flops in metrics.py)
try:
    from thop import profile as thop_profile
    _has_thop = True
except Exception:
    _has_thop = False
try:
    from fvcore.nn import FlopCountAnalysis
    _has_fvcore = True
except Exception:
    _has_fvcore = False

# All metric functions imported from metrics.py
from metrics import (
    dice_loss, calculate_iou, combined_loss, dice_coefficient, iou_score, precision_score, hd95_score, mae_score,
    count_parameters, calculate_flops, calculate_inference_time, format_model_statistics
)

def custom_collate_fn(batch):
    images = []
    targets = []
    for img, target in batch:
        images.append(img)
        targets.append(target)
    images = torch.stack(images, 0)
    return images, targets

def evaluate_test_with_metrics(model, test_loader, device, threshold=0.7):
    model.eval()
    test_loss = 0.0
    test_acc = 0.0
    test_dice = 0.0
    test_iou = 0.0
    test_precision = 0.0
    test_hd95 = 0.0
    test_mae = 0.0
    detection_coverage = 0.0
    batch_count = 0

    test_pbar = tqdm(enumerate(test_loader), total=len(test_loader),
                     desc="Test Evaluation", leave=False, ncols=100)
    with torch.no_grad():
        for batch_idx, (images, targets) in test_pbar:
            images = images.to(device)

            true_masks_list = []
            for t in targets:
                mask = t['masks']
                if mask.ndim == 3:
                    mask = mask[0:1, :, :]
                else:
                    mask = mask.unsqueeze(0)
                true_masks_list.append(mask)
            true_masks = torch.stack(true_masks_list, dim=0).to(device)

            moved_targets = []
            for t in targets:
                target_dict = {}
                for k, v in t.items():
                    if isinstance(v, torch.Tensor):
                        target_dict[k] = v.to(device)
                moved_targets.append(target_dict)

            model.train()
            losses = model(images, moved_targets)
            model.eval()

            valid_losses = []
            for loss_val in losses.values():
                if isinstance(loss_val, torch.Tensor) and loss_val.ndim == 0:
                    valid_losses.append(loss_val)
                else:
                    valid_losses.append(torch.tensor(0.0, device=device))
            total_loss = sum(valid_losses)
            test_loss += total_loss.item()

            pred_dict = model(images, targets=None)
            seg_logits = pred_dict["segmentations"]
            proposals = [res["boxes"] for res in pred_dict["detections"]]

            B = images.size(0)
            full_pred_masks = torch.zeros_like(true_masks).to(device)
            roi_idx = 0

            for img_idx in range(B):
                boxes = proposals[img_idx]
                if boxes.numel() == 0:
                    continue
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.tolist())
                    x1 = max(0, min(x1, 511))
                    y1 = max(0, min(y1, 511))
                    x2 = max(x1+1, min(x2, 512))
                    y2 = max(y1+1, min(y2, 512))
                    if roi_idx >= len(seg_logits):
                        break
                    pred_64 = torch.sigmoid(seg_logits[roi_idx:roi_idx+1])
                    pred_crop = F.interpolate(
                        pred_64, size=(y2-y1, x2-x1),
                        mode="bilinear", align_corners=False
                    )
                    full_pred_masks[img_idx, :, y1:y2, x1:x2] = torch.max(
                        full_pred_masks[img_idx, :, y1:y2, x1:x2],
                        pred_crop
                    )
                    roi_idx += 1

            binary_pred_full = (full_pred_masks > threshold).float()
            binary_true_full = true_masks

            dice = dice_coefficient(binary_pred_full, binary_true_full).item()
            iou = iou_score(binary_pred_full, binary_true_full).item()
            acc = (binary_pred_full == binary_true_full).float().mean().item()
            prec = precision_score(binary_pred_full, binary_true_full).item()
            hd95 = hd95_score(binary_pred_full, binary_true_full)
            mae = mae_score(binary_pred_full, binary_true_full) if mae_score is not None else 0.0

            test_dice += dice
            test_iou += iou
            test_acc += acc
            test_precision += prec
            test_hd95 += hd95
            test_mae += mae

            SAVE_DIR = "/root/zhangshanfeng/UNet-Medical-master/dataset/small_tumor/result-loss64-dice512"
            os.makedirs(SAVE_DIR, exist_ok=True)
            if not hasattr(evaluate_test_with_metrics, "global_idx"):
                evaluate_test_with_metrics.global_idx = 0

            for img_idx in range(B):
                img_name = targets[img_idx]["image_name"]
                pred_mask = (full_pred_masks[img_idx] > threshold).squeeze().cpu().numpy() * 255
                save_path = os.path.join(SAVE_DIR, img_name)
                cv2.imwrite(save_path, pred_mask)

            evaluate_test_with_metrics.global_idx += B

            coverage = 0.0
            det_results = model.detector(images)
            for i in range(len(images)):
                pred_boxes = det_results[i].get("boxes", torch.empty((0, 4), device=device))
                gt_boxes = moved_targets[i]["boxes"]
                if pred_boxes.numel() > 0 and gt_boxes.numel() > 0:
                    x1, y1, x2, y2 = pred_boxes[0].round().int().tolist()
                    h, w = true_masks[i].shape[-2:]
                    x1 = max(0, min(x1, w - 1))
                    y1 = max(0, min(y1, h - 1))
                    x2 = max(x1 + 1, min(x2, w))
                    y2 = max(y1 + 1, min(y2, h))
                    gt_crop = true_masks[i][:, y1:y2, x1:x2]
                    total = true_masks[i].sum().item()
                    coverage += gt_crop.sum().item() / total if total > 0 else 0.0
                else:
                    coverage += 0.0
            detection_coverage += coverage / len(images)

            batch_count += 1
            test_pbar.set_postfix({"loss": f"{total_loss.item():.3f}", "dice": f"{dice:.3f}"})

    n = batch_count if batch_count > 0 else 1
    avg_loss = test_loss / n
    avg_acc = test_acc / n
    avg_dice = test_dice / n
    avg_iou = test_iou / n
    avg_prec = test_precision / n
    avg_hd95 = test_hd95 / n
    avg_mae = test_mae / n
    avg_cov = detection_coverage / n

    print(f"\n================ Test Set Evaluation Results ================")
    print(f"Test Loss:        {avg_loss:.4f}")
    print(f"Test Accuracy:    {avg_acc:.4f}")
    print(f"Test Dice Score:  {avg_dice:.4f}")
    print(f"Test IoU:         {avg_iou:.4f}")
    print(f"Test Precision:   {avg_prec:.4f}")
    print(f"Test HD95:        {avg_hd95:.4f}")
    print(f"Test MAE:         {avg_mae:.4f}")
    print(f"Detection Coverage:{avg_cov:.4f}")

    total_params, trainable_params = count_parameters(model)
    sample_images, _ = next(iter(test_loader))
    per_forward_flops, flops_tool = calculate_flops(model, sample_images.to(device), _has_thop, _has_fvcore)
    inference_time = calculate_inference_time(model, sample_images.to(device), 1)
    stats = format_model_statistics(total_params, trainable_params, per_forward_flops, flops_tool, 0, inference_time, len(test_loader))
    for line in stats:
        print(line)
    print("=====================================================================")

    swanlab.log({
        "test/loss": avg_loss, "test/acc": avg_acc,
        "test/dice": avg_dice, "test/iou": avg_iou,
        "test/precision": avg_prec, "test/hd95": avg_hd95,
        "test/mae": avg_mae, "test/cov": avg_cov
    })

    return avg_loss, avg_acc, avg_dice, avg_iou, avg_prec, avg_hd95, avg_mae, avg_cov

def visualize_predictions_with_metrics(model, test_loader, device, num_samples=4, threshold=0.7):
    model.eval()
    with torch.no_grad():
        images, targets = next(iter(test_loader))
        images = images.to(device)
        true_masks = torch.stack([t['masks'] for t in targets], dim=0).to(device)

        pred_masks = model.infer_full_mask(images, threshold=threshold)
        img_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        img_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        denorm_images = images.cpu() * img_std + img_mean
        denorm_images = denorm_images.clip(0, 1)

        sample_idx = random.sample(range(len(images)), min(num_samples, len(images)))
        fig, axes = plt.subplots(len(sample_idx), 3, figsize=(15, 5 * len(sample_idx)))
        fig.suptitle(f'Prediction Visualization (threshold={threshold})', fontsize=16)

        for i, idx in enumerate(sample_idx):
            ax1 = axes[i, 0] if len(sample_idx) > 1 else axes[0]
            img_to_show = denorm_images[idx].permute(1, 2, 0).numpy()
            ax1.imshow(img_to_show)
            ax1.set_title('Original Image')
            ax1.axis('off')

            ax2 = axes[i, 1] if len(sample_idx) > 1 else axes[1]
            ax2.imshow(true_masks[idx].cpu().squeeze(), cmap='gray')
            ax2.set_title('Ground Truth Mask')
            ax2.axis('off')

            ax3 = axes[i, 2] if len(sample_idx) > 1 else axes[2]
            pred = pred_masks[idx].cpu().squeeze()
            true = true_masks[idx].cpu().squeeze()
            dice = dice_coefficient(pred, true).item()
            iou = iou_score(pred, true).item()
            ax3.imshow(pred, cmap='gray')
            ax3.set_title(f'Predicted Mask\nDice: {dice:.3f} | IoU: {iou:.3f}')
            ax3.axis('off')

        plt.tight_layout()
        swanlab.log({"predictions/with_metrics": swanlab.Image(fig)})
        print("Prediction results saved to SwanLab")

def main():
    swanlab.init(
        project="Unet-Medical-Segmentation",
        experiment_name="fasterrcnn-unet-test",
        config={
            "batch_size": 4,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "data_dir": "/root/zhangshanfeng/UNet-Medical-master/dataset/small_tumor",
            "image_size": (512, 512),
            "threshold": 0.65,
            "model_path": 'models/best_model_loss64_dice512_2.pth'
        }
    )
    device = torch.device(swanlab.config["device"])
    print(f"Using device: {device}")

    data_root = "/root/zhangshanfeng/UNet-Medical-master/dataset/small_tumor"
    _, _, test_ds = create_train_val_test_datasets(
        data_root=data_root,
        target_size=swanlab.config["image_size"]
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=swanlab.config["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )

    print("\nInitializing model and loading weights...")
    model = FasterRCNNUNet(num_classes=1).to(device)

    if not os.path.exists(swanlab.config["model_path"]):
        raise FileNotFoundError(f"Model file not found at {swanlab.config['model_path']}. Please run train.py first.")

    model.load_state_dict(torch.load(swanlab.config["model_path"], map_location=device))
    print("Model loaded successfully.")

    print("\nStarting test evaluation...")
    evaluate_test_with_metrics(model, test_loader, device, threshold=swanlab.config["threshold"])

    print("\nGenerating visualizations...")
    visualize_predictions_with_metrics(model, test_loader, device, num_samples=4, threshold=swanlab.config["threshold"])

    print("\nTesting and evaluation completed!")

if __name__ == '__main__':
    main()