# pytorch_project/augmentation/transforms.py

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2 # Albumentations 通常需要 OpenCV

# --- 配置 ---
# 从原始脚本 image_augmentor 函数和通用实践中获取灵感
# 您可以根据需要调整这些参数
IMG_SIZE = 256 # 应该与您在 1_crop_resize.py 中使用的 TARGET_SIZE 一致
ROTATION_LIMIT = 25
SHEAR_LIMIT = 10
SCALE_LIMIT = 0.1 # 对应 zoom_random(percentage_area=0.9)，需要调整以匹配效果
GRID_DISTORTION_LIMIT = 0.2 # 作为 random_distortion 的替代方案
# ImageNet 均值和标准差，用于归一化
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# --- 训练集增强 ---
def get_train_transforms():
    """定义用于训练集的在线数据增强流程"""
    return A.Compose([
        # 几何变换 (参考原始脚本)
        A.Resize(height=IMG_SIZE, width=IMG_SIZE, interpolation=cv2.INTER_AREA), # 确保尺寸正确
        A.Rotate(limit=ROTATION_LIMIT, p=0.7, border_mode=cv2.BORDER_CONSTANT),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        # A.RandomScale(scale_limit=SCALE_LIMIT, p=0.7), # 类似 zoom_random
        # A.Affine(shear=(-SHEAR_LIMIT, SHEAR_LIMIT), p=0.5), # 类似 shear
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=SCALE_LIMIT, rotate_limit=0, p=0.7, border_mode=cv2.BORDER_CONSTANT), # 包含缩放(Zoom)和位移
        A.OneOf([ # 选择其中一种扭曲/倾斜类增强
            A.GridDistortion(num_steps=5, distort_limit=GRID_DISTORTION_LIMIT, p=1.0),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0),
            # A.Perspective(scale=(0.05, 0.1), p=1.0) # 类似 skew_tilt，可选
        ], p=0.5),

        # 颜色/像素级增强 (可选，但常用)
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.3),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.5),
            A.MotionBlur(blur_limit=(3, 7), p=0.5),
        ], p=0.2),
        # A.Cutout(num_holes=8, max_h_size=IMG_SIZE//8, max_w_size=IMG_SIZE//8, fill_value=0, p=0.3), # 可选，类似 Cutout

        # 归一化和转换为 Tensor
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(), # 将 NumPy (H,W,C) 转换为 PyTorch Tensor (C,H,W)
    ])

# --- 验证/测试集增强 ---
def get_val_transforms():
    """定义用于验证/测试集的在线数据增强流程 (通常只有 Resize 和 Normalize)"""
    return A.Compose([
        A.Resize(height=IMG_SIZE, width=IMG_SIZE, interpolation=cv2.INTER_AREA),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])

# --- (可选) 如果您的 Dataset 返回 PIL Image，可以使用 torchvision ---
# import torchvision.transforms as T
# def get_train_transforms_torchvision():
#     return T.Compose([
#         T.Resize((IMG_SIZE, IMG_SIZE)),
#         T.RandomRotation(degrees=ROTATION_LIMIT),
#         T.RandomHorizontalFlip(p=0.5),
#         T.RandomVerticalFlip(p=0.5),
#         T.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(1.0-SCALE_LIMIT, 1.0+SCALE_LIMIT), shear=SHEAR_LIMIT),
#         T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
#         T.ToTensor(),
#         T.Normalize(mean=MEAN, std=STD)
#     ])
# def get_val_transforms_torchvision():
#      return T.Compose([
#         T.Resize((IMG_SIZE, IMG_SIZE)),
#         T.ToTensor(),
#         T.Normalize(mean=MEAN, std=STD)
#     ])