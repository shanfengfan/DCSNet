import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math

from torchvision.ops import RoIAlign
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models import ResNet50_Weights
import torchvision
from torchvision.models.detection.transform import GeneralizedRCNNTransform

import torch.utils.checkpoint as checkpoint

try:
    from torch.amp import autocast
except ImportError:
    from torch.cuda.amp import autocast




class PixelAdaptiveFusion(nn.Module):
    def __init__(self, d_model, num_levels, reduction=4):
        super().__init__()
        self.num_levels = num_levels
        self.gate_generator = nn.Sequential(
            nn.Conv2d(d_model * num_levels, d_model // reduction, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_model // reduction, d_model // reduction, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_model // reduction, num_levels, kernel_size=1)
        )

    def forward(self, feats_list):
        combined = torch.cat(feats_list, dim=1)
        logits = self.gate_generator(combined)
        weights = F.softmax(logits, dim=1) 
        
        fused_feat = 0
        for i in range(self.num_levels):
            fused_feat += feats_list[i] * weights[:, i:i+1, :, :]
            
        return fused_feat, weights

class GatedFusionBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(d_model * 2, d_model, kernel_size=1),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True)
        )

    def forward(self, skip_feat, prev_feat):
        combined = torch.cat([skip_feat, prev_feat], dim=1)
        return self.fuse_conv(combined)

class MultiScaleUNetSegHead(nn.Module):
    def __init__(self, in_channels=256, out_channels=1, d_model=256, nhead=8, num_layers=4,
                 fpn_levels=("0", "1", "2", "3"), use_checkpoint: bool = False, dropout_rate: float = 0.2):
        super().__init__()
        self.fpn_levels = fpn_levels
        self.d_model = d_model
        self.use_checkpoint = use_checkpoint
        self.fpn_enhance = nn.ModuleDict({
            lvl: nn.Sequential(
                nn.Conv2d(in_channels, d_model, kernel_size=1),
                nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, groups=d_model),
                nn.ReLU(inplace=True)
            ) for lvl in self.fpn_levels
        })
        
        
        self.level_proj = nn.ModuleDict({lvl: nn.Conv2d(in_channels, d_model, kernel_size=1)
                                         for lvl in self.fpn_levels})
        
        self.level_to_size = {
            "0": (64, 64),
            "1": (32, 32),
            "2": (16, 16),
            "3": (8, 8)
        }
        level_to_scale = {"0": 1/4.0, "1": 1/8.0, "2": 1/16.0, "3": 1/32.0}
        self.roi_align = nn.ModuleDict({
            lvl: RoIAlign(
                output_size=self.level_to_size[lvl],
                spatial_scale=level_to_scale[lvl],
                sampling_ratio=2
            ) for lvl in self.fpn_levels
        })

        self.encoder_blocks = nn.ModuleList()
        for i, lvl in enumerate(self.fpn_levels):
            encoder_block = nn.Sequential(
                nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
                nn.BatchNorm2d(d_model),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout_rate),
                nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
                nn.BatchNorm2d(d_model),
                nn.ReLU(inplace=True)
            )
            self.encoder_blocks.append(encoder_block)

        self.level_to_patch = {
            "0": 8,
            "1": 4,
            "2": 2,
            "3": 1
        }
        self.patch_embed = nn.ModuleDict({
            lvl: nn.Conv2d(d_model, d_model, kernel_size=self.level_to_patch[lvl], stride=self.level_to_patch[lvl])
            for lvl in self.fpn_levels
        })
        self.tokens_per_level = 8 * 8
        total_tokens = self.tokens_per_level * len(self.fpn_levels)
        self.pos_embed = nn.Parameter(torch.zeros(1, total_tokens, d_model))
        self.level_embed = nn.Parameter(torch.zeros(len(self.fpn_levels), d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout_rate,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.decoder_blocks = nn.ModuleList()
        for i in range(len(self.fpn_levels)):
            skip_fusion = GatedFusionBlock(d_model)
            upsample = nn.Sequential(
                nn.ConvTranspose2d(d_model, d_model, kernel_size=2, stride=2),
                nn.BatchNorm2d(d_model),
                nn.ReLU(inplace=True)
            )
            self.decoder_blocks.append(nn.ModuleDict({
                'skip_fusion': skip_fusion,
                'upsample': upsample
            }))
        
        class DetailEnhance(nn.Module):
            def __init__(self, channels: int):
                super().__init__()
                self.depthwise = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
                self.pointwise = nn.Conv2d(channels, channels, kernel_size=1)
                self.relu = nn.ReLU(inplace=True)
                self.attention = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(channels, channels//4, kernel_size=1),
                    nn.ReLU(),
                    nn.Conv2d(channels//4, channels, kernel_size=1),
                    nn.Sigmoid()
                )
                self.shortcut = nn.Conv2d(channels, channels, kernel_size=1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                residual = self.shortcut(x)
                x = self.depthwise(x)
                x = self.pointwise(x)
                x = self.relu(x)
                attn = self.attention(x)
                x = x * attn
                return self.relu(x + residual)

        self.detail_enhance = DetailEnhance(d_model)

        self.decoder_level_heads = nn.ModuleList()
        for i in range(len(self.fpn_levels)):
            level_head = nn.Sequential(
                nn.Conv2d(d_model, d_model // 2, kernel_size=3, padding=1),
                nn.BatchNorm2d(d_model // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(d_model // 2, out_channels, kernel_size=1)
            )
            self.decoder_level_heads.append(level_head)

        self.main_head = nn.Sequential(
            nn.Conv2d(d_model, d_model // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model // 2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate),
            nn.Conv2d(d_model // 2, d_model // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model // 4),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate),
            nn.Conv2d(d_model // 4, out_channels, kernel_size=1)
        )
 
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.level_embed, std=0.02)
        
        self.pixel_fusion = PixelAdaptiveFusion(d_model=d_model, num_levels=len(self.fpn_levels))
        
        self.loss_weights = nn.Parameter(torch.ones(len(self.fpn_levels) + 1))
        
        self.edge_head = nn.Sequential(
            nn.Conv2d(d_model, d_model // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_model // 4, out_channels, kernel_size=1)
        )

    def forward(self, fpn_features: dict, rois: torch.Tensor):
        device = rois.device
        if rois.numel() == 0:
            sz = self.level_to_size[self.fpn_levels[0]] if len(self.fpn_levels) > 0 else (64, 64)
            result = {"main": torch.empty((0, 1, *sz), device=device, dtype=torch.float32)}
            for i in range(len(self.fpn_levels)):
                result[f"level_{i}"] = torch.empty((0, 1, *sz), device=device, dtype=torch.float32)
            return result

        encoder_features = []
        
        sizes = []
        for lvl_idx, lvl in enumerate(self.fpn_levels):
            if lvl not in fpn_features:
                continue
            feat = fpn_features[lvl]
            roi_feat = self.roi_align[lvl](feat, rois)
            proj = self.fpn_enhance[lvl](roi_feat)
            encoded = self.encoder_blocks[lvl_idx](proj)
            encoder_features.append(encoded)
            sizes.append(self.level_to_size[lvl])

        if not encoder_features:
            sz = self.level_to_size[self.fpn_levels[0]] if len(self.fpn_levels) > 0 else (64, 64)
            result = {"main": torch.empty((0, 1, *sz), device=device, dtype=torch.float32)}
            for i in range(len(self.fpn_levels)):
                result[f"level_{i}"] = torch.empty((0, 1, *sz), device=device, dtype=torch.float32)
            return result

        level_tokens = []
        for lvl_idx, encoded_feat in enumerate(encoder_features):
            lvl = self.fpn_levels[lvl_idx]
            patches = self.patch_embed[lvl](encoded_feat)
            tokens = patches.flatten(2).transpose(1, 2)
            tokens = tokens + self.level_embed[lvl_idx].view(1, 1, -1)
            level_tokens.append(tokens)

        x = torch.cat(level_tokens, dim=1)
        x = x + self.pos_embed[:, :x.size(1), :]

        if self.use_checkpoint and self.training:
            for layer in self.transformer.layers:
                x = checkpoint.checkpoint(layer, x)
            if self.transformer.norm is not None:
                x = self.transformer.norm(x)
        else:
            x = self.transformer(x)

        n = rois.size(0)
        t = self.tokens_per_level
        num_lvls = len(encoder_features)
        
        transformer_features = []
        
        for i in range(num_lvls):
            seg_i = x[:, i*t:(i+1)*t, :]
            seg_i = seg_i.transpose(1, 2).view(n, self.d_model, 8, 8)
            H_i, W_i = sizes[i]
            seg_i_up = F.interpolate(seg_i, size=(H_i, W_i), mode='bilinear', align_corners=True)
            transformer_features.append(seg_i_up)

        decoder_features = []
        decoder_predictions = {}
        target_size = self.level_to_size[self.fpn_levels[0]]

        for i in reversed(range(num_lvls)):
            if i == num_lvls-1:
                decoder_feat = transformer_features[i]
            else:
                skip_connection = encoder_features[i]
                prev_decoder = decoder_features[-1]
                prev_decoder = F.interpolate(
                  prev_decoder, 
                  size=skip_connection.shape[-2:], 
                  mode='bilinear', 
                  align_corners=True
                )

                decoder_feat = self.decoder_blocks[i]['skip_fusion'](skip_connection, prev_decoder)

            decoder_features.append(decoder_feat)
            upsampled_feat = F.interpolate(
                decoder_feat, 
                size=target_size, 
                mode='bilinear', 
                align_corners=True
            )
            level_pred = self.decoder_level_heads[i](upsampled_feat)
            decoder_predictions[f"level_{i}"] = level_pred

        upsampled_decoder = [
            F.interpolate(f, size=target_size, mode='bilinear', align_corners=True) 
            for f in decoder_features
        ]

        enhanced_decoder = [self.detail_enhance(f) for f in upsampled_decoder]
        fused, pixel_weights = self.pixel_fusion(enhanced_decoder)
        fused = self.detail_enhance(fused)
        
        main_pred = self.main_head(fused)
        edge_pred = self.edge_head(fused)
        
        result = {
            "main": main_pred,
            "edge": edge_pred
            }
        result.update(decoder_predictions)
        
        return result


class FasterRCNNUNet(nn.Module):
    def __init__(self, num_classes=1, box_detections_per_img=3, use_amp: bool = False, use_checkpoint: bool = False):
        super().__init__()
        self.verbose = False
        self.forward_calls = 0
        self.forward_train_steps = 0
        self.forward_eval_steps = 0
        self.use_amp = use_amp
        self.total_epochs = 100
        self.current_epoch = 0
        self._cached_fpn_features = None
        backbone = resnet_fpn_backbone(
            backbone_name='resnet50',
            weights=ResNet50_Weights.IMAGENET1K_V1,
            trainable_layers=4
        )

        self.detector = torchvision.models.detection.FasterRCNN(
            backbone=backbone,
            num_classes=2,
            box_detections_per_img=box_detections_per_img,
            box_score_thresh=0.35,
            box_nms_thresh=0.5,
            transform=None
        )

        self._set_transform()

        self.seg_head = MultiScaleUNetSegHead(
            in_channels=256, 
            out_channels=num_classes, 
            d_model=256, 
            nhead=8, 
            num_layers=4,
            fpn_levels=("0", "1", "2", "3"),  
            use_checkpoint=use_checkpoint
        )

        def _cache_fpn_hook(module, inputs, output):
            self._cached_fpn_features = output
        self._fpn_hook_handle = self.detector.backbone.register_forward_hook(_cache_fpn_hook)

    def _set_transform(self):
        img_mean = [0.0, 0.0, 0.0] 
        img_std = [1.0, 1.0, 1.0]
        min_size = 512
        max_size = 512
        self.detector.transform = GeneralizedRCNNTransform(
            min_size=min_size,
            max_size=max_size,
            image_mean=img_mean,
            image_std=img_std
        )

    def _validate_targets(self, targets, device):
        valid_targets = []
        for t in targets:
            boxes = t['boxes'].to(device, dtype=torch.float32)
            if boxes.ndim != 2 or boxes.shape[1] != 4:
                boxes = torch.zeros((0, 4), device=device, dtype=torch.float32)
            
            labels = t['labels'].to(device, dtype=torch.int64)
            if labels.ndim != 1 or labels.shape[0] != boxes.shape[0]:
                labels = torch.zeros((boxes.shape[0],), device=device, dtype=torch.int64)
            
            masks = t['masks'].to(device, dtype=torch.float32)
            
            valid_targets.append({
                'boxes': boxes,
                'labels': labels,
                'masks': masks
            })
        return valid_targets

    def compute_seg_loss(self, pred_dict, proposals, targets):
        
        device = proposals[0].device if proposals else targets[0]['boxes'].device
        if len(proposals) == 0 or pred_dict["main"].numel() == 0:
            empty_detail = {
                "level_0": 0.0, "level_1": 0.0, "level_2": 0.0, "level_3": 0.0,
                "main": 0.0, "consistency": 0.0, "edge":0.0, "total": 0.0
            }
            return torch.tensor(0.0, device=device, dtype=torch.float32), empty_detail
    
        roi_size = (64, 64)
        seg_targets = []
    
        for img_idx, boxes in enumerate(proposals):
            masks = targets[img_idx]["masks"].float()
            target_bin = (masks > 0).float()
    
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.tolist())
                x1 = max(0, min(x1, 511))
                y1 = max(0, min(y1, 511))
                x2 = max(x1+1, min(x2, 512))
                y2 = max(y1+1, min(y2, 512))
    
                crop_gt = target_bin[:, y1:y2, x1:x2]
                gt_roi = F.interpolate(
                    crop_gt.unsqueeze(0),
                    size=roi_size,
                    mode="bilinear",
                    align_corners=True
                ).squeeze(0)
                gt_roi = (gt_roi > 0.5).float()
                seg_targets.append(gt_roi)
    
        if not seg_targets:
            empty_detail = {
                "level_0": 0.0, "level_1": 0.0, "level_2": 0.0, "level_3": 0.0,
                "main": 0.0, "consistency": 0.0, "edge":0.0, "total": 0.0
            }
            return torch.tensor(0.0, device=device, dtype=torch.float32), empty_detail
    
        seg_targets = torch.stack(seg_targets, 0).to(device)
        full_preds = pred_dict
    
        level_weights = self.adjust_level_weights_during_training()
    
        seg_loss, seg_loss_detail = compute_fused_multi_scale_loss(
            full_preds,
            seg_targets,
            loss_weights=self.seg_head.loss_weights,
            level_weights=level_weights,
            consistency_weight=0.05
        )
    
        return seg_loss, seg_loss_detail

    def set_training_progress(self, current_epoch: int, total_epochs: int):
        self.current_epoch = current_epoch
        self.total_epochs = total_epochs
    
    def adjust_level_weights_during_training(self):
        if self.total_epochs == 0:
            return None
        
        progress = self.current_epoch / self.total_epochs
        
        if progress < 0.3:
            level_weights = [1.0, 0.8, 0.6, 0.4, 0.2]
        elif progress < 0.6:
            level_weights = [0.8, 0.9, 1.0, 1.1, 1.2]
        else:
            level_weights = [0.7, 0.8, 0.9, 1.0, 1.1]
        
        return level_weights
    
    def get_consistency_weight(self):
        if self.total_epochs == 0:
            return 0.1
        
        progress = self.current_epoch / self.total_epochs
        consistency_weight = 0.1 + 0.3 * progress
        return min(consistency_weight, 0.1)    

    def apply_roi_jitter(self, rois, img_size, jitter_ratio=0.05):
        if jitter_ratio <= 0 or rois.shape[0] == 0:
            return rois
    
        device = rois.device
        w = rois[:, 2] - rois[:, 0]
        h = rois[:, 3] - rois[:, 1]
    
        sigma = jitter_ratio
        dw = torch.randn_like(w) * sigma * w
        dh = torch.randn_like(h) * sigma * h
        rw = torch.randn_like(w) * sigma * w
        rh = torch.randn_like(h) * sigma * h
    
        new_rois = rois.clone()
        new_rois[:, 0] += dw - rw / 2
        new_rois[:, 1] += dh - rh / 2
        new_rois[:, 2] += dw + rw / 2
        new_rois[:, 3] += dh + rh / 2
    
        new_rois[:, [0, 2]] = new_rois[:, [0, 2]].clamp(0, img_size[1])
        new_rois[:, [1, 3]] = new_rois[:, [1, 3]].clamp(0, img_size[0])
    
        return new_rois

    def forward(self, images, targets=None):
        if isinstance(images, list):
            images = torch.stack(images, dim=0)
        
        assert images.shape[1] == 3, f"Input channel error: expected 3, got {images.shape[1]}"
        
        device = images.device

        if self.training or (targets is not None):
            assert targets is not None, "Targets must be provided for loss calculation"
            
            targets = self._validate_targets(targets, device)
            
            
            det_losses = self.detector(images, targets)
            
            if not isinstance(det_losses, dict):
                det_losses = {
                    'loss_classifier': torch.tensor(0.0, device=device, dtype=torch.float32),
                    'loss_box_reg': torch.tensor(0.0, device=device, dtype=torch.float32),
                    'loss_objectness': torch.tensor(0.0, device=device, dtype=torch.float32),
                    'loss_rpn_box_reg': torch.tensor(0.0, device=device, dtype=torch.float32)
                }
            self.det_loss_detail = {}
            for loss_name, loss_val in det_losses.items():
                if isinstance(loss_val, torch.Tensor) and loss_val.ndim == 0:
                    self.det_loss_detail[loss_name] = loss_val.item() if torch.isfinite(loss_val) else 0.0
                else:
                    self.det_loss_detail[loss_name] = 0.0
            cleaned_det_losses = {}
            for k, v in det_losses.items():
                if isinstance(v, torch.Tensor) and v.ndim == 0:
                    cleaned_det_losses[k] = v.to(device)
                else:
                    cleaned_det_losses[k] = torch.tensor(0.0, device=device, dtype=torch.float32)
            
            proposals = [t["boxes"] for t in targets]
            fpn_features = self._cached_fpn_features if self._cached_fpn_features is not None else {}

            rois_list = []
            for img_idx, boxes in enumerate(proposals):
                if boxes.numel() == 0:
                    continue
                batch_idx = torch.full((boxes.shape[0], 1), img_idx, device=device)
                rois_list.append(torch.cat([batch_idx, boxes], 1))
            
            pred_dict = {}
            seg_loss_detail = {}
            if rois_list:
                all_rois = torch.cat(rois_list, 0)    
                if self.training:
                    img_h, img_w = images.shape[-2:]
                    roi_indices = all_rois[:, :1]
                    roi_boxes = all_rois[:, 1:]
                    roi_boxes = self.apply_roi_jitter(roi_boxes, (img_h, img_w), jitter_ratio=0.05)
                    all_rois = torch.cat([roi_indices, roi_boxes], dim=1)
                device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
                with autocast(device_type, enabled=self.use_amp):
                    pred_dict = self.seg_head(fpn_features, all_rois)
                seg_loss, seg_loss_detail = self.compute_seg_loss(pred_dict, proposals, targets)
            else:
                seg_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
                seg_loss_detail = {
                    "level_0": 0.0, "level_1": 0.0, "level_2": 0.0, "level_3": 0.0,
                    "main": 0.0, "consistency": 0.0, "total": 0.0
                }
            
            seg_loss = seg_loss * 1.0
            
            if torch.isfinite(seg_loss):
                seg_loss_detail["total"] = seg_loss.item()
            else:
                seg_loss_detail["total"] = 0.0
            
            total_losses = {**cleaned_det_losses, 'seg_loss': seg_loss}
            total_losses['seg_loss_detail'] = seg_loss_detail
            total_losses['det_loss_detail'] = self.det_loss_detail
            return total_losses
        else:
            det_results = self.detector(images)

            filtered_results = []
            proposals = []
            for res in det_results:
                boxes = res.get("boxes", torch.empty((0, 4), device=images.device))
                scores = res.get("scores", torch.empty((0,), device=images.device))
                labels = res.get("labels", torch.empty((0,), dtype=torch.long, device=images.device))
                keep = (labels == 1)
                boxes = boxes[keep]
                scores = scores[keep]
                k = min(1, boxes.shape[0])
                if k > 0:
                    _, idxs = scores.topk(k)
                    boxes = boxes[idxs]
                    scores = scores[idxs]
                    labels = torch.ones((k,), dtype=torch.long, device=labels.device)
                else:
                    boxes = torch.empty((0, 4), device=images.device)
                    labels = torch.empty((0,), dtype=torch.long, device=images.device)
                    scores = torch.empty((0,), device=images.device)
                filtered_results.append({"boxes": boxes, "labels": labels, "scores": scores})
                proposals.append(boxes)
            
            with torch.no_grad():
                processed_images, _ = self.detector.transform(images, None)
            fpn_features = self._cached_fpn_features
            if fpn_features is None:
                fpn_features = self.detector.backbone(processed_images.tensors)

            rois_list = []
            for img_idx, boxes in enumerate(proposals):
                if boxes.numel() == 0:
                    continue
                batch_idx = torch.full((boxes.shape[0], 1), img_idx, device=device)
                rois_list.append(torch.cat([batch_idx, boxes], 1))
           
            pred_dict = {}
            target_size = self.seg_head.level_to_size[self.seg_head.fpn_levels[0]] if len(self.seg_head.fpn_levels) > 0 else (64, 64)
            
            num_levels = len(self.seg_head.fpn_levels)
            for i in range(num_levels):
                pred_dict[f"level_{i}"] = torch.empty((0, 1, *target_size), device=device, dtype=torch.float32)
            
            pred_dict["main"] = torch.empty((0, 1, *target_size), device=device, dtype=torch.float32)
            
            if rois_list:
                all_rois = torch.cat(rois_list, 0)
                device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
                with autocast(device_type, enabled=self.use_amp):
                    pred_dict = self.seg_head(fpn_features, all_rois)
            
            return {'detections': filtered_results, 'segmentations': pred_dict["main"]}

    def infer_full_mask(self, images, threshold=0.5):
        self.eval()
        with torch.no_grad():
            results = self.forward(images, targets=None)
            seg_logits = results['segmentations']
            proposals = [res['boxes'] for res in results['detections']]
            
            full_pred_masks = map_roi_seg_to_full_image(
                seg_logits, 
                proposals, 
                full_img_size=images.shape[2:],
                threshold=threshold
            )
            return full_pred_masks


def dice_coefficient(pred, target, smooth=1e-6):
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    return (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)


def compute_fused_multi_scale_loss(pred_dict, seg_targets, loss_weights=None,
                                   level_weights=None, consistency_weight=0.05):
    device = seg_targets.device

    def dice_loss(pred, target):
        dice_coef = dice_coefficient(torch.sigmoid(pred), target)
        return torch.clamp(1 - dice_coef, 0.0, 1.0)
    
    def simple_iou_loss(pred, target):
        pred = torch.sigmoid(pred)
        inter = (pred * target).sum()
        union = pred.sum() + target.sum() - inter
        iou = (inter + 1e-6) / (union + 1e-6)
        return torch.clamp(1 - iou, 0.0, 1.0)
    
    def focal_loss(pred, target, alpha: float = 0.30, gamma: float = 2.0):
        prob = torch.sigmoid(pred)
        ce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.where(target == 1, prob, 1 - prob)
        loss = (alpha * (1 - pt).pow(gamma) * ce).mean()
        return torch.clamp(loss, 0.0, 5.0)

    def boundary_loss(pred, target):
        prob = torch.sigmoid(pred)
        tgt_edge = target - F.max_pool2d(target, kernel_size=3, stride=1, padding=1)
        tgt_edge = (tgt_edge > 0).float()
        prb_edge = prob - F.max_pool2d(prob, kernel_size=3, stride=1, padding=1)
        prb_edge = (prb_edge > 0.5).float()
        edge_dice = dice_coefficient(prb_edge, tgt_edge)
        return torch.clamp(1 - edge_dice, 0.0, 1.0)

    def get_gt_boundary(x):
        laplacian_kernel = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], 
                                        dtype=torch.float32, device=x.device).reshape(1, 1, 3, 3)
        boundary = F.conv2d(x, laplacian_kernel, padding=1)
        return (boundary > 0.1).float()

    def edge_specific_loss(pred, target):
        return F.binary_cross_entropy_with_logits(pred, target, pos_weight=torch.tensor([10.0]).to(device))

    all_predictions = []
    prediction_names = []
    for key in sorted(pred_dict.keys()):
        if key.startswith("level_"):
            all_predictions.append(pred_dict[key])
            prediction_names.append(key)
    all_predictions.append(pred_dict["main"])
    prediction_names.append("main")
    
    all_losses = []
    num_predictions = len(all_predictions)
    for i, pred in enumerate(all_predictions):
        l_dice = dice_loss(pred, seg_targets)
        l_boundary = boundary_loss(pred, seg_targets)
        l_focal = focal_loss(pred, seg_targets)
        
        if i < num_predictions - 1:
            if i == 0:
                level_loss = 0.4*l_dice + 0.2*l_boundary + 0.4*l_focal
            elif i == num_predictions - 2:
                level_loss = 0.5*l_dice + 0.3*l_boundary + 0.2*l_focal
            else:
                level_loss = 0.5*l_dice + 0.2*l_boundary + 0.3*l_focal
        else:
            level_loss = 0.5*l_dice + 0.3*l_boundary + 0.2*l_focal
        
        all_losses.append(level_loss)

    edge_loss_val = 0.0
    if "edge" in pred_dict:
        edge_gt = get_gt_boundary(seg_targets)
        l_edge = edge_specific_loss(pred_dict["edge"], edge_gt)
        edge_loss_val = l_edge

    loss_detail = {}
    for name, loss in zip(prediction_names, all_losses):
        loss_detail[name] = loss.item() if torch.isfinite(loss) else 0.0
    if isinstance(edge_loss_val, torch.Tensor):
        loss_detail["edge_head"] = edge_loss_val.item()

    total_loss = 0.0
    if loss_weights is not None:
        weights = F.softmax(loss_weights, dim=0)
        for i, loss in enumerate(all_losses):
            total_loss += weights[i] * loss
    elif level_weights is not None:
        if len(level_weights) != len(all_losses):
            level_weights = [1.0] * len(all_losses)
        for i, (loss, weight) in enumerate(zip(all_losses, level_weights)):
            total_loss += weight * loss
    else:
        default_weights = [0.7, 0.8, 0.9, 1.0, 1.1]
        for i, loss in enumerate(all_losses):
            weight = default_weights[i] if i < len(default_weights) else 1.0
            total_loss += weight * loss
    
    if isinstance(edge_loss_val, torch.Tensor):
        total_loss += edge_loss_val * 0.2

    consistency_loss_val = 0.0
    if consistency_weight > 0 and len(all_predictions) > 1:
        consistency_loss = 0.0
        num_pairs = 0
        for i in range(len(all_predictions) - 1):
            pred_i = torch.sigmoid(all_predictions[i])
            pred_j = torch.sigmoid(all_predictions[i+1])
            mse_loss = F.mse_loss(pred_i, pred_j)
            consistency_loss += mse_loss
            num_pairs += 1
        if num_pairs > 0:
            consistency_loss /= num_pairs
            total_loss += consistency_weight * consistency_loss
            consistency_loss_val = consistency_loss.item() if torch.isfinite(consistency_loss) else 0.0
    
    loss_detail["consistency"] = consistency_loss_val
    loss_detail["edge"] = edge_loss_val * 0.2 if isinstance(edge_loss_val, torch.Tensor) else 0.0
    loss_detail["total"] = total_loss.item() if torch.isfinite(total_loss) else 0.0

    total_loss = torch.clamp(total_loss, 0.0, 50.0)
    
    return total_loss, loss_detail


def iou_score(pred, target, smooth=1e-6):
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    return (intersection + smooth) / (union + smooth)


def map_roi_logits_to_full_image(seg_logits, proposals, full_img_size=(512, 512)):
    device = seg_logits.device if seg_logits is not None else torch.device('cpu')
    batch_size = len(proposals)
    full_pred_logits = torch.zeros((batch_size, 1, *full_img_size), device=device)
    if seg_logits.numel() == 0:
        return full_pred_logits
    roi_idx = 0
    for img_idx in range(batch_size):
        boxes = proposals[img_idx]
        if boxes.numel() == 0:
            continue
        accum = torch.full((1, 1, *full_img_size), fill_value=-1e9, device=device)
        for box in boxes:
            if roi_idx >= seg_logits.shape[0]:
                break
            x1, y1, x2, y2 = map(int, box.tolist())
            x1 = max(0, min(x1, full_img_size[1]-1))
            y1 = max(0, min(y1, full_img_size[0]-1))
            x2 = max(x1+1, min(x2, full_img_size[1]))
            y2 = max(y1+1, min(y2, full_img_size[0]))
            roi_h, roi_w = y2 - y1, x2 - x1
            if roi_h <= 0 or roi_w <= 0:
                roi_idx += 1
                continue
            roi_logit = seg_logits[roi_idx:roi_idx+1]
            resized_roi_logit = F.interpolate(
                roi_logit,
                size=(roi_h, roi_w),
                mode='bilinear',
                align_corners=True
            )
            canvas = torch.full_like(accum, fill_value=-1e9)
            canvas[:, :, y1:y2, x1:x2] = resized_roi_logit
            accum = torch.maximum(accum, canvas)
            roi_idx += 1
        full_pred_logits[img_idx:img_idx+1] = accum
    return full_pred_logits


def map_roi_seg_to_full_image(seg_logits, proposals, full_img_size=(512, 512), threshold=0.5):
    device = seg_logits.device if seg_logits is not None else torch.device('cpu')
    batch_size = len(proposals)
    full_pred_masks = torch.zeros((batch_size, 1, *full_img_size), device=device)
    
    if seg_logits.numel() == 0:
        return full_pred_masks
    
    seg_probs = torch.sigmoid(seg_logits)
    binary_segs = (seg_probs > threshold).float()
    
    roi_idx = 0
    for img_idx in range(batch_size):
        boxes = proposals[img_idx]
        if boxes.numel() == 0:
            continue
        
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.tolist())
            x1 = max(0, min(x1, full_img_size[1]-1))
            y1 = max(0, min(y1, full_img_size[0]-1))
            x2 = max(x1+1, min(x2, full_img_size[1]))
            y2 = max(y1+1, min(y2, full_img_size[0]))
            roi_h, roi_w = y2 - y1, x2 - x1
            if roi_h <= 0 or roi_w <= 0:
                roi_idx += 1
                continue
            
            if roi_idx >= binary_segs.shape[0]:
                break
            
            roi_seg = binary_segs[roi_idx:roi_idx+1]
            resized_roi_seg = F.interpolate(
                roi_seg, 
                size=(roi_h, roi_w), 
                mode='bilinear',
                align_corners=True
            )
            
            full_pred_masks[img_idx, :, y1:y2, x1:x2] = resized_roi_seg
            roi_idx += 1
    
    return full_pred_masks