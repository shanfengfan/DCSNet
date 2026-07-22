import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


def expand_mask_channel(mask, **kwargs):
    """确保掩码为单通道（和图像维度对齐）"""
    if len(mask.shape) == 2:
        return np.expand_dims(mask, axis=-1)
    else:
        return mask[..., 0:1]


class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_names, transform=None, target_size=(512, 512)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_names = image_names
        self.transform = transform
        self.target_size = target_size

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        #image = Image.open(img_path).convert('L')
        image = Image.open(img_path).convert('RGB')
        image = np.array(image, dtype=np.uint8)

        # 加载单通道掩码
        mask = Image.open(mask_path).convert('L')
        mask = np.array(mask, dtype=np.uint8)

        # 数据增强
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # 掩码二值化
        if isinstance(mask, torch.Tensor):
            mask_np = mask.squeeze().cpu().numpy().astype(np.uint8)
        else:
            mask_np = mask.squeeze().astype(np.uint8)

        mask_np = (mask_np > 127).astype(np.uint8)
        h, w = mask_np.shape

        # 连通域分析生成实例框（给检测头用）
        num_components, comp_map = cv2.connectedComponents(mask_np, connectivity=8)
        instance_masks = []
        boxes_list = []
        expand_pixel = 5

        for comp_id in range(1, num_components):
            comp_mask = (comp_map == comp_id).astype(np.uint8)
            if comp_mask.sum() < 4:
                continue

            ys, xs = np.where(comp_mask > 0)
            x1, y1 = xs.min(), ys.min()
            x2, y2 = xs.max(), ys.max()

            # 轻微外扩框，让ROI更稳定
            x1 = max(0, x1 - expand_pixel)
            y1 = max(0, y1 - expand_pixel)
            x2 = min(w - 1, x2 + expand_pixel)
            y2 = min(h - 1, y2 + expand_pixel)

            if x2 - x1 < 3 or y2 - y1 < 3:
                continue

            boxes_list.append([x1, y1, x2, y2])
            instance_masks.append(torch.from_numpy(comp_mask).unsqueeze(0))

        # 构建模型需要的 target
        if instance_masks:
            seg_mask = torch.stack(instance_masks).sum(dim=0).clamp(0, 1).float()
            boxes = torch.tensor(boxes_list, dtype=torch.float32)
            labels = torch.ones((len(boxes),), dtype=torch.int64)
        else:
            seg_mask = torch.zeros((1, h, w), dtype=torch.float32)
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': seg_mask,
            'image_name': img_name
        }

        return image.float(), target


def create_train_val_test_datasets(
    data_root, target_size=(512, 512)
):
    # 手动划分的文件夹结构
    train_img_dir = os.path.join(data_root, "train", "images")
    train_mask_dir = os.path.join(data_root, "train", "masks")
    val_img_dir = os.path.join(data_root, "val", "images")
    val_mask_dir = os.path.join(data_root, "val", "masks")
    test_img_dir = os.path.join(data_root, "test", "images")
    test_mask_dir = os.path.join(data_root, "test", "masks")

    def get_image_names(img_dir):
        names = []
        for f in os.listdir(img_dir):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                names.append(f)
        return sorted(names)

    train_names = get_image_names(train_img_dir)
    val_names = get_image_names(val_img_dir)
    test_names = get_image_names(test_img_dir)

    print(f"\n✅ 手动划分数据集加载完成：")
    print(f"- 训练集：{len(train_names)} 张")
    print(f"- 验证集：{len(val_names)} 张")
    print(f"- 测试集：{len(test_names)} 张\n")

    # 训练增强（医学图像专用）
    train_transform = A.Compose([
        A.Resize(height=target_size[0], width=target_size[1]),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.3),
        A.ShiftScaleRotate(p=0.4, shift_limit=0.05, scale_limit=0.1, rotate_limit=15, border_mode=cv2.BORDER_CONSTANT),
        A.GridDistortion(p=0.2, distort_limit=0.1, border_mode=cv2.BORDER_CONSTANT),
        A.RandomBrightnessContrast(p=0.4, brightness_limit=0.15, contrast_limit=0.15),
        A.RandomGamma(p=0.2, gamma_limit=(80, 120)),
        A.GaussNoise(p=0.2),
        A.GaussianBlur(p=0.2, blur_limit=(3, 5)),
        A.Lambda(mask=expand_mask_channel),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2()
    ])

    # 验证/测试无增强
    val_test_transform = A.Compose([
        A.Resize(height=target_size[0], width=target_size[1]),
        A.Lambda(mask=expand_mask_channel),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2()
    ])

    train_dataset = SegmentationDataset(
        image_dir=train_img_dir, mask_dir=train_mask_dir, image_names=train_names,
        transform=train_transform, target_size=target_size
    )
    val_dataset = SegmentationDataset(
        image_dir=val_img_dir, mask_dir=val_mask_dir, image_names=val_names,
        transform=val_test_transform, target_size=target_size
    )
    test_dataset = SegmentationDataset(
        image_dir=test_img_dir, mask_dir=test_mask_dir, image_names=test_names,
        transform=val_test_transform, target_size=target_size
    )

    return train_dataset, val_dataset, test_dataset