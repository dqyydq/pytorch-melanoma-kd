# pytorch_project/metrics/fid.py

import os
import argparse # Keep argparse for the if __name__ == '__main__' block

# --- !! 修改导入和添加新导入 !! ---
import torch
import torch.nn as nn
import numpy as np
from torchvision import models
from torchvision import transforms # Use torchvision transforms here
from torch.utils.data import Dataset, DataLoader # Use standard DataLoader
from PIL import Image # Use PIL for image loading
from pathlib import Path # Use Path for path operations
from scipy import linalg
# from core.data_loader import get_eval_loader # <<<--- 删除或注释掉这行
import torch.nn.functional as F # <<< 添加这行
try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not installed
    def tqdm(x, **kwargs): return x
# --- !! 导入结束 !! ---


class InceptionV3(nn.Module):
    """Pretrained InceptionV3 network returning feature maps"""
    # --- !! 添加必要的修改以适应新的 PyTorch 版本 !! ---
    # Block changes: remove inception aux module, use default init weights
    # See https://pytorch.org/hub/pytorch_vision_inception_v3/

    def __init__(self):
        super().__init__()
        # Load pretrained Inception V3 model
        inception = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT) # Use updated weights argument

        # Block 1: Initial layers up to max pool
        self.block1 = nn.Sequential(
            inception.Conv2d_1a_3x3, inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2))
        # Block 2: Further layers up to the next max pool
        self.block2 = nn.Sequential(
            inception.Conv2d_3b_1x1, inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2))
        # Block 3: Mixed layers up to the final convolutional layers before pooling
        self.block3 = nn.Sequential(
            inception.Mixed_5b, inception.Mixed_5c,
            inception.Mixed_5d, inception.Mixed_6a,
            inception.Mixed_6b, inception.Mixed_6c,
            inception.Mixed_6d, inception.Mixed_6e)
        # Block 4: Final mixed layers and adaptive average pooling
        self.block4 = nn.Sequential(
            inception.Mixed_7a, inception.Mixed_7b,
            inception.Mixed_7c,
            nn.AdaptiveAvgPool2d(output_size=(1, 1)))

    def forward(self, x):
        """Extract features of the given image x from pool3 layer."""
        # N x 3 x 299 x 299
        x = self.block1(x)
        # N x 64 x 147 x 147
        x = self.block2(x)
        # N x 192 x 71 x 71
        x = self.block3(x)
        # N x 768 x 35 x 35 (depends on version, check InceptionV3 structure)
        # The original FID implementation often uses features before the final avg pool
        # For consistency with common FID implementations, let's stop after block3 (Mixed_6e)
        # If block4 features are needed, uncomment the relevant lines
        # x = self.block4(x)
        # N x 2048 x 1 x 1
        # return x.view(x.size(0), -1) # N x 2048

        # Return features from the end of block3 (pool3 layer features)
        # Need to apply pooling here if using block3 output directly
        x = F.adaptive_avg_pool2d(x, output_size=(1, 1)) # Apply pooling
        return x.view(x.size(0), -1) # N x 768 (or dim after Mixed_6e)


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer for the first image set.
    -- mu2   : Numpy array containing the activations of a layer for the second image set.
    -- sigma1: Numpy array containing the covariance matrix of the first image set.
    -- sigma2: Numpy array containing the covariance matrix of the second image set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            #raise ValueError('Imaginary component {}'.format(m))
            print('Warning: Imaginary component {}'.format(m)) # Use warning instead of error
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


@torch.no_grad()
def calculate_activation_statistics(loader, model, batch_size, dims, device):
    """Calculation of the statistics used by the FID.
    Params:
    -- loader       : DataLoader providing images
    -- model        : Instance of inception model
    -- batch_size   : Batch size of images for inception model
    -- dims         : Dimensionality of features returned by inception
    -- device       : Device to run calculations
    Returns:
    -- mu    : The mean over features of the activations of the pool_3 layer of
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
    """
    model.eval()
    act_list = []

    # Use tqdm for progress indication
    for batch in tqdm(loader, desc="Calculating Activations"):
        batch = batch.to(device)
        with torch.no_grad():
            act = model(batch) # Get activations
        act_list.append(act.cpu().numpy()) # Store activations as numpy array

    act = np.concatenate(act_list, axis=0)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


# +++ 添加简单的图像加载 Dataset +++
class FidImageDataset(Dataset):
    """Simple Dataset for loading images for FID calculation."""
    def __init__(self, root, transform=None):
        # Find common image file extensions recursively
        self.files = list(Path(root).rglob('*.[pP][nN][gG]')) + \
                     list(Path(root).rglob('*.[jJ][pP][gG]')) + \
                     list(Path(root).rglob('*.[jJ][pP][eE][gG]'))
        if not self.files:
             print(f"Warning: No image files found in {root}")
        self.files.sort() # Sort for consistency (optional)
        self.transform = transform
        if self.transform is None:
            # Default transform if none provided (matches InceptionV3 needs)
            self.transform = transforms.Compose([
                transforms.Resize((299, 299)),
                transforms.ToTensor(), # To [0, 1] range
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        try:
            img = Image.open(path).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            return img
        except Exception as e:
            print(f"Warning: Error loading image {path} for FID: {e}")
            # Return a dummy tensor on error
            return torch.zeros((3, 299, 299))


# +++ 修改 calculate_fid_given_paths 函数 +++
def calculate_fid_given_paths(paths, img_size, batch_size, device='cuda', dims=2048, num_workers=0): # Set num_workers default to 0 for simplicity/compatibility
    """Calculates the FID of two paths by loading images manually"""
    print('Calculating FID given paths %s and %s...' % (paths[0], paths[1]))

    if not os.path.exists(paths[0]):
        raise RuntimeError('Invalid path: %s' % paths[0])
    if not os.path.exists(paths[1]):
        raise RuntimeError('Invalid path: %s' % paths[1])

    # Use the updated InceptionV3 model
    print("Loading InceptionV3 model...")
    inception_model = InceptionV3()
    inception_weights_path = "D:/python_code/pytorch_melanama_kd/my_models/inception_v3_google-0cc3c7bd.pth" # <--- 您的新路径

# 3. 加载本地权重
    if Path(inception_weights_path).exists():
        try:
            state_dict = torch.load(inception_weights_path, map_location=device)
            # InceptionV3 默认加载的 state_dict 可能包含辅助分类器或其他不匹配的键
            # 需要进行一些处理，或者确保加载时严格匹配 (strict=False)
            inception_model.load_state_dict(state_dict, strict=False) # strict=False 允许部分加载
            print(f"InceptionV3 weights loaded from local path: {inception_weights_path}")
        except Exception as e:
            print(f"Error loading InceptionV3 weights from {inception_weights_path}: {e}")
            # 可以选择退出或使用未初始化的模型 (FID结果将无效)
    else:
        print(f"Error: InceptionV3 weights not found at local path: {inception_weights_path}")
        # 退出或处理错误

    inception_model.eval().to(device)

    # FID traditionally uses specific output dims (e.g., 2048 from pool3, or 768 from later layers)
    # Let's adapt dims based on our InceptionV3 forward method output (N x 768 if using block3)
    # You might need to adjust 'dims' based on the actual output size of your InceptionV3 forward pass
    # dims = 768 # Or 2048 if using block4 output
    # inception_model.eval().to(device)
    print("InceptionV3 model loaded.")

    # --- !! 使用我们定义的简单 Dataset 和 DataLoader !! ---
    # Define transforms needed for InceptionV3
    transform_fid = transforms.Compose([
            transforms.Resize((299, 299)), # InceptionV3 input size
            transforms.ToTensor(), # Range [0, 1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print(f"Creating DataLoader for Real images: {paths[0]}")
    dataset1 = FidImageDataset(root=paths[0], transform=transform_fid)
    loader1 = DataLoader(dataset1, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=True)

    print(f"Creating DataLoader for Fake images: {paths[1]}")
    dataset2 = FidImageDataset(root=paths[1], transform=transform_fid)
    loader2 = DataLoader(dataset2, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=True)
    # --- !! 数据加载结束 !! ---

    print(f"Calculating activation statistics for Real images ({len(dataset1)} images)...")
    mu1, sigma1 = calculate_activation_statistics(loader1, inception_model, batch_size, dims, device)
    print(f"Calculating activation statistics for Fake images ({len(dataset2)} images)...")
    mu2, sigma2 = calculate_activation_statistics(loader2, inception_model, batch_size, dims, device)

    print(f"Calculating Frechet Distance...")
    fid_value = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)

    return fid_value


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Keep command-line arguments for potential direct use
    parser.add_argument('--paths', type=str, nargs=2, required=True, help='Paths to the directory of real images and the directory of fake images respectively.')
    parser.add_argument('--img_size', type=int, default=299, help='Image size (usually 299 for InceptionV3)') # Default to 299
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size to use for InceptionV3 inference.') # Default lower batch size
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda or cpu). Autodetects if None.')
    parser.add_argument('--dims', type=int, default=768, help='Dimensionality of Inception features to use. Default 768 corresponds to output of Mixed_6e.') # Default to 768
    parser.add_argument('--num_workers', type=int, default=0, help='Number of dataloader workers.')

    args = parser.parse_args()

    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    fid_value = calculate_fid_given_paths(paths=args.paths,
                                          img_size=args.img_size, # Pass img_size, though FidImageDataset uses 299
                                          batch_size=args.batch_size,
                                          device=device,
                                          dims=args.dims,
                                          num_workers=args.num_workers)
    print('Calculated FID score:', fid_value)

# Example command line usage:
# python metrics/fid.py --paths /path/to/real_images /path/to/fake_images --batch_size 16