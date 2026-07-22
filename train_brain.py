# -*- coding: utf-8 -*-
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import random
import swanlab
import os
import numpy as np
from torch.utils.data import DataLoader
from net import FasterRCNNUNet
import torch.nn.functional as F
from PIL import Image
from dataset import create_train_val_test_datasets

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
from tqdm import tqdm

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

def _is_finite_tensor(x: torch.Tensor) -> bool:
    return torch.isfinite(x).all().item() if isinstance(x, torch.Tensor) else True

def train_model(model, train_loader, val_loader, optimizer, scheduler, num_epochs, device, patience=20, grad_clip_norm: float = 1.0):
    best_val_loss = float('inf')
    patience_counter = 0
    model.total_epochs = num_epochs
    model.current_epoch = 0
    scaler = torch.amp.GradScaler('cuda', enabled=model.use_amp)
    
    for epoch in range(num_epochs):
        model.current_epoch = epoch
        model.train()
        train_loss = 0.0
        
        train_seg_loss_accum = {
            "level_0": 0.0, "level_1": 0.0, "level_2": 0.0, "level_3": 0.0,
            "main": 0.0, "consistency": 0.0, "edge":0.0, "total": 0.0
        }
        train_det_loss_accum = {
            "loss_classifier": 0.0,
            "loss_box_reg": 0.0,
            "loss_objectness": 0.0,
            "loss_rpn_box_reg": 0.0,
            "total": 0.0
        }
        train_seg_batch_count = 0
        train_det_batch_count = 0
        
        train_pbar = tqdm(enumerate(train_loader), total=len(train_loader), 
                          desc=f"Epoch {epoch+1}/{num_epochs} [Train]", 
                          leave=True, dynamic_ncols=False, ncols=120, ascii=False)
        
        for batch_idx, (images, targets) in train_pbar:
            images = images.to(device)
            moved_targets = []
            for t in targets:
                target_dict = {}
                for k, v in t.items():
                    if isinstance(v, torch.Tensor):
                        target_dict[k] = v.to(device)
                moved_targets.append(target_dict)
            
            model.train()
            model.current_epoch = epoch
            optimizer.zero_grad()
            losses = model(images, moved_targets)
            
            seg_loss_detail = losses.get('seg_loss_detail', {
                "level_0": 0.0, "level_1": 0.0, "level_2": 0.0, "level_3": 0.0,
                "main": 0.0, "consistency": 0.0, "edge":0.0, "total": 0.0
            })
            
            for key in train_seg_loss_accum.keys():
                train_seg_loss_accum[key] += seg_loss_detail.get(key, 0.0)
            train_seg_batch_count += 1
            
            det_loss_detail = losses.get('det_loss_detail', {
                "loss_classifier": 0.0,
                "loss_box_reg": 0.0,
                "loss_objectness": 0.0,
                "loss_rpn_box_reg": 0.0,
                "total": 0.0
            })
            for key in train_det_loss_accum.keys():
                if key in det_loss_detail:
                    train_det_loss_accum[key] += det_loss_detail[key]
                elif key == "total":
                    det_total = sum([det_loss_detail.get(k, 0.0) for k in ["loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"]])
                    train_det_loss_accum[key] += det_total
            train_det_batch_count += 1
            
            valid_losses = []
            for loss_val in losses.values():
                if isinstance(loss_val, torch.Tensor) and loss_val.ndim == 0 and torch.isfinite(loss_val):
                    valid_losses.append(loss_val)
                else:
                    valid_losses.append(torch.tensor(0.0, device=device))
            
            total_loss = sum(valid_losses)
            if (not torch.isfinite(total_loss)) or (not total_loss.requires_grad):
                optimizer.zero_grad(set_to_none=True)
                train_pbar.set_postfix({"Avg Loss": "NaN/Inf/NoGrad, skipped"})
                continue
            
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            train_loss += total_loss.item()
            avg_loss = train_loss / (batch_idx + 1)
            train_pbar.set_postfix({"Avg Loss": f"{avg_loss:.4f}"})
        
        train_avg_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0.0
        train_seg_avg = {}
        if train_seg_batch_count > 0:
            for key in train_seg_loss_accum:
                train_seg_avg[key] = train_seg_loss_accum[key] / train_seg_batch_count
        else:
            train_seg_avg = {k: 0.0 for k in train_seg_loss_accum.keys()}
            
        train_det_avg = {}
        if train_det_batch_count > 0:
            for key in train_det_loss_accum:
                train_det_avg[key] = train_det_loss_accum[key] / train_det_batch_count
        else:
            train_det_avg = {k: 0.0 for k in train_det_loss_accum.keys()}

        model.eval()
        val_loss = 0.0
        val_seg_loss_accum = {
            "level_0": 0.0, "level_1": 0.0, "level_2": 0.0, "level_3": 0.0,
            "main": 0.0, "consistency": 0.0, "edge":0.0, "total": 0.0
        }
        
        val_det_loss_accum = {
            "loss_classifier": 0.0,
            "loss_box_reg": 0.0,
            "loss_objectness": 0.0,
            "loss_rpn_box_reg": 0.0,
            "total": 0.0
        }
        
        val_seg_batch_count = 0
        val_det_batch_count = 0
        
        val_pbar = tqdm(enumerate(val_loader), total=len(val_loader),
                        desc=f"Epoch {epoch+1}/{num_epochs} [Val]",
                        leave=True, dynamic_ncols=False, ncols=120, ascii=False)
        
        with torch.no_grad():
            for batch_idx, (images, targets) in val_pbar:
                images = images.to(device)
                moved_targets = []
                for t in targets:
                    target_dict = {}
                    for k, v in t.items():
                        if isinstance(v, torch.Tensor):
                            target_dict[k] = v.to(device)
                    moved_targets.append(target_dict)
                
                model.train()
                model.current_epoch = epoch
                losses = model(images, moved_targets)
                model.eval()
                    
                seg_loss_detail = losses.get('seg_loss_detail', {
                    "level_0": 0.0, "level_1": 0.0, "level_2": 0.0, "level_3": 0.0,
                    "main": 0.0, "consistency": 0.0, "edge":0.0, "total": 0.0
                })
                
                det_loss_detail = losses.get('det_loss_detail', {
                    "loss_classifier": 0.0,
                    "loss_box_reg": 0.0,
                    "loss_objectness": 0.0,
                    "loss_rpn_box_reg": 0.0,
                    "total": 0.0
                })
                
                for key in val_seg_loss_accum.keys():
                    val_seg_loss_accum[key] += seg_loss_detail.get(key, 0.0)
                val_seg_batch_count += 1
                
                for key in val_det_loss_accum.keys():
                    if key in det_loss_detail:
                        val_det_loss_accum[key] += det_loss_detail[key]
                    elif key == "total":
                        det_total = sum([det_loss_detail.get(k, 0.0) for k in ["loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"]])
                        val_det_loss_accum[key] += det_total
                val_det_batch_count += 1
                
                valid_val_losses = []
                for loss_val in losses.values():
                    if isinstance(loss_val, torch.Tensor) and loss_val.ndim == 0 and torch.isfinite(loss_val):
                        valid_val_losses.append(loss_val)
                    else:
                        valid_val_losses.append(torch.tensor(0.0, device=device))
                
                total_val_loss = sum(valid_val_losses)
                if not torch.isfinite(total_val_loss):
                    val_pbar.set_postfix({"Avg Loss": "NaN/Inf"})
                    continue
                val_loss += total_val_loss.item()
                
                avg_val_loss = val_loss / (batch_idx + 1)
                val_pbar.set_postfix({"Avg Loss": f"{avg_val_loss:.4f}"})

        val_avg_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        
        val_seg_avg = {}
        if val_seg_batch_count > 0:
            for key in val_seg_loss_accum:
                val_seg_avg[key] = val_seg_loss_accum[key] / val_seg_batch_count
        else:
            val_seg_avg = {k: 0.0 for k in val_seg_loss_accum.keys()}
            
        val_det_avg = {}
        if val_det_batch_count > 0:
            for key in val_det_loss_accum:
                val_det_avg[key] = val_det_loss_accum[key] / val_det_batch_count
        else:
            val_det_avg = {k: 0.0 for k in val_det_loss_accum.keys()}
            
        print(f"\n Epoch {epoch+1}/{num_epochs} Detection Loss Details (Avg):")
        print("+----------------------+-----------+-----------+")
        print("|      Loss Type        |   Train   |    Val    |")
        print("+----------------------+-----------+-----------+")
        print(f"| Classifier           | {train_det_avg['loss_classifier']:.6f} | {val_det_avg['loss_classifier']:.6f} |")
        print(f"| Box Reg              | {train_det_avg['loss_box_reg']:.6f} | {val_det_avg['loss_box_reg']:.6f} |")
        print(f"| Objectness           | {train_det_avg['loss_objectness']:.6f} | {val_det_avg['loss_objectness']:.6f} |")
        print(f"| RPN Box Reg          | {train_det_avg['loss_rpn_box_reg']:.6f} | {val_det_avg['loss_rpn_box_reg']:.6f} |")
        print(f"| Total Detection      | {train_det_avg['total']:.6f} | {val_det_avg['total']:.6f} |")
        print("+----------------------+-----------+-----------+")

        print(f"\n Epoch {epoch+1}/{num_epochs} Segmentation Loss Details (Avg):")
        print("+--------------+-----------+-----------+")
        print("|  Loss Level  |   Train   |    Val    |")
        print("+--------------+-----------+-----------+")
        print(f"| level_0      | {train_seg_avg['level_0']:.4f} | {val_seg_avg['level_0']:.4f} |")
        print(f"| level_1      | {train_seg_avg['level_1']:.4f} | {val_seg_avg['level_1']:.4f} |")
        print(f"| level_2      | {train_seg_avg['level_2']:.4f} | {val_seg_avg['level_2']:.4f} |")
        print(f"| level_3      | {train_seg_avg['level_3']:.4f} | {val_seg_avg['level_3']:.4f} |")
        print(f"| main         | {train_seg_avg['main']:.4f} | {val_seg_avg['main']:.4f} |")
        print(f"| consistency  | {train_seg_avg['consistency']:.4f} | {val_seg_avg['consistency']:.4f} |")
        print(f"| edge         | {train_seg_avg['edge']:.4f} | {val_seg_avg['edge']:.4f} |")
        print(f"| Total Seg    | {train_seg_avg['total']:.4f} | {val_seg_avg['total']:.4f} |")
        print("+--------------+-----------+-----------+")
        
        print(f"\n Overall Summary:")
        print(f"Train Total Loss: {train_avg_loss:.6f} (Det: {train_det_avg['total']:.6f}, Seg: {train_seg_avg['total']:.6f})")
        print(f"Val Total Loss:   {val_avg_loss:.6f} (Det: {val_det_avg['total']:.6f}, Seg: {val_seg_avg['total']:.6f})")
        
        swanlab.log({
            "train/total_loss": train_avg_loss,
            "val/total_loss": val_avg_loss,
            "epoch": epoch + 1,
            "train/seg_level_0": train_seg_avg["level_0"],
            "train/seg_level_1": train_seg_avg["level_1"],
            "train/seg_level_2": train_seg_avg["level_2"],
            "train/seg_level_3": train_seg_avg["level_3"],
            "train/seg_main": train_seg_avg["main"],
            "train/seg_consistency": train_seg_avg["consistency"],
            "val/seg_level_0": val_seg_avg["level_0"],
            "val/seg_level_1": val_seg_avg["level_1"],
            "val/seg_level_2": val_seg_avg["level_2"],
            "val/seg_level_3": val_seg_avg["level_3"],
            "val/seg_main": val_seg_avg["main"],
            "val/seg_consistency": val_seg_avg["consistency"],
        }, step=epoch + 1)

        print(f"\nEpoch {epoch+1:02d}/{num_epochs} Summary:")
        print(f"Train Loss: {train_avg_loss:.4f} | Val Loss: {val_avg_loss:.4f}")
        
        level_weights = model.adjust_level_weights_during_training()
        consistency_weight = model.get_consistency_weight()
        print(f"[INFO] Current Weights - Levels: {level_weights}, Consistency: {consistency_weight:.3f}")
        
        current_lr = optimizer.param_groups[0]['lr']
        swanlab.log({"train/learning_rate": current_lr}, step=epoch + 1)
        
        if (val_avg_loss > 1e-6) and np.isfinite(val_avg_loss):
            scheduler.step(val_avg_loss)
            
        val_loss_valid = (val_avg_loss > 1e-6) and np.isfinite(val_avg_loss)
        if val_loss_valid and val_avg_loss < best_val_loss:
            best_val_loss = val_avg_loss
            patience_counter = 0
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), 'models/best_model_loss64_dice512_2.pth')
            print(f"[INFO] Best model updated (Best Val Loss: {best_val_loss:.4f})\n")
        else:
            if not val_loss_valid:
                print(f"[WARNING] Invalid val loss, early stop counter not increased\n")
                continue
            patience_counter += 1
            print(f"[INFO] Early stop counter: {patience_counter}/{patience}\n")
            if patience_counter >= patience:
                print(f"[INFO] Early stop triggered, training stopped\n")
                break


def main():
    swanlab.init(
        project="Unet-Medical-Segmentation",
        experiment_name="fasterrcnn-unet-final",
        config={
            "batch_size": 4,
            "lr": 2e-4,
            "weight_decay": 1e-4,
            "dropout_rate": 0.2,
            "label_smoothing": 0.1, 
            "epochs": 100,
            "patience": 20,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "data_dir": "/root/zhangshanfeng/UNet-Medical-master/dataset/small_tumor",
            "image_size": (512, 512)
        }
    )
    device = torch.device(swanlab.config["device"])
    print(f"Using device: {device}")

    data_root = "/root/zhangshanfeng/UNet-Medical-master/dataset/small_tumor"
    train_ds, val_ds, test_ds = create_train_val_test_datasets(
        data_root=data_root,
        target_size=swanlab.config["image_size"]
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=swanlab.config["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=swanlab.config["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )

    model = FasterRCNNUNet(num_classes=1).to(device)
    print("\nModel initialized (Enhanced Seg Head + RGB 3-Channel)")
    
    optimizer = optim.AdamW(model.parameters(), 
                            lr=swanlab.config["lr"], 
                            weight_decay=swanlab.config["weight_decay"])
                            
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7
    )                       

    print("\nStarting training...")
    train_model(
        model, 
        train_loader, 
        val_loader, 
        optimizer, 
        scheduler,
        swanlab.config["epochs"], 
        device,
        patience=swanlab.config["patience"]
    )

    print("\nTraining completed! Run test.py separately to evaluate best model.")

if __name__ == '__main__':
    main()  