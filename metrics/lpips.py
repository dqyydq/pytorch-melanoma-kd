# """
# StarGAN v2
# Copyright (c) 2020-present NAVER Corp.

# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
# """

# import torch
# import torch.nn as nn
# from torchvision import models


# def normalize(x, eps=1e-10):
#     return x * torch.rsqrt(torch.sum(x**2, dim=1, keepdim=True) + eps)


# class AlexNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.layers = models.alexnet(pretrained=True).features
#         self.channels = []
#         for layer in self.layers:
#             if isinstance(layer, nn.Conv2d):
#                 self.channels.append(layer.out_channels)

#     def forward(self, x):
#         fmaps = []
#         for layer in self.layers:
#             x = layer(x)
#             if isinstance(layer, nn.ReLU):
#                 fmaps.append(x)
#         return fmaps


# class Conv1x1(nn.Module):
#     def __init__(self, in_channels, out_channels=1):
#         super().__init__()
#         self.main = nn.Sequential(
#             nn.Dropout(0.5),
#             nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False))

#     def forward(self, x):
#         return self.main(x)


# class LPIPS(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.alexnet = AlexNet()
#         self.lpips_weights = nn.ModuleList()
#         for channels in self.alexnet.channels:
#             self.lpips_weights.append(Conv1x1(channels, 1))
#         self._load_lpips_weights()
#         # imagenet normalization for range [-1, 1]
#         self.mu = torch.tensor([-0.03, -0.088, -0.188]).view(1, 3, 1, 1).cuda()
#         self.sigma = torch.tensor([0.458, 0.448, 0.450]).view(1, 3, 1, 1).cuda()

#     def _load_lpips_weights(self):
#         own_state_dict = self.state_dict()
#         if torch.cuda.is_available():
#             state_dict = torch.load('metrics/lpips_weights.ckpt')
#         else:
#             state_dict = torch.load('metrics/lpips_weights.ckpt',
#                                     map_location=torch.device('cpu'))
#         for name, param in state_dict.items():
#             if name in own_state_dict:
#                 own_state_dict[name].copy_(param)

#     def forward(self, x, y):
#         x = (x - self.mu) / self.sigma
#         y = (y - self.mu) / self.sigma
#         x_fmaps = self.alexnet(x)
#         y_fmaps = self.alexnet(y)
#         lpips_value = 0
#         for x_fmap, y_fmap, conv1x1 in zip(x_fmaps, y_fmaps, self.lpips_weights):
#             x_fmap = normalize(x_fmap)
#             y_fmap = normalize(y_fmap)
#             lpips_value += torch.mean(conv1x1((x_fmap - y_fmap)**2))
#         return lpips_value


# @torch.no_grad()
# def calculate_lpips_given_images(group_of_images):
#     # group_of_images = [torch.randn(N, C, H, W) for _ in range(10)]
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     lpips = LPIPS().eval().to(device)
#     lpips_values = []
#     num_rand_outputs = len(group_of_images)

#     # calculate the average of pairwise distances among all random outputs
#     for i in range(num_rand_outputs-1):
#         for j in range(i+1, num_rand_outputs):
#             lpips_values.append(lpips(group_of_images[i], group_of_images[j]))
#     lpips_value = torch.mean(torch.stack(lpips_values, dim=0))
#     return lpips_value.item()



# pytorch_project/metrics/lpips.py
"""
LPIPS metric calculation, adapted from StarGAN v2 official implementation
to load weights from specified local paths.
Copyright (c) 2020-present NAVER Corp. (Original License applies)
"""

import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path # Import Path for robust path handling
import numpy as np # Import numpy for mean calculation if needed later

# --- !! 修改为您移动后的权重文件实际路径 !! ---
# 例如: "D:/my_models/alexnet-owt-7be5be79.pth" 或相对于项目根目录的路径
DEFAULT_ALEXNET_WEIGHTS_PATH = "D:/python_code/pytorch_melanama_kd/my_models/alexnet-owt-7be5be79.pth"
# 例如: "./metrics/lpips_weights.ckpt" (如果它在 metrics 目录下) 或 "D:/my_models/lpips_weights.ckpt"
DEFAULT_LPIPS_CKPT_PATH = "./metrics/lpips_weights.ckpt"
# --- !! 路径修改结束 !! ---


def normalize(x, eps=1e-10):
    """Normalizes activations."""
    # Computes norm along channel dimension
    return x * torch.rsqrt(torch.sum(x**2, dim=1, keepdim=True) + eps)


class AlexNet(nn.Module):
    """AlexNet feature extractor, loading weights locally."""
    def __init__(self, weights_path=DEFAULT_ALEXNET_WEIGHTS_PATH): # Use default path
        super().__init__()
        print(f"Initializing AlexNet Feature Extractor...")
        # 1. Create AlexNet structure without pretrained weights
        alexnet_full_model = models.alexnet(weights=None)

        # 2. Load weights from the specified local path
        weights_path = Path(weights_path)
        if weights_path.is_file():
            try:
                state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
                alexnet_full_model.load_state_dict(state_dict)
                print(f"  AlexNet weights loaded successfully from: {weights_path}")
            except Exception as e:
                print(f"  Warning: Error loading AlexNet weights from {weights_path}: {e}.")
                print("           Using randomly initialized AlexNet features (LPIPS results will be invalid).")
        else:
            print(f"  Warning: AlexNet weights file not found at {weights_path}.")
            print("           Using randomly initialized AlexNet features (LPIPS results will be invalid).")

        # 3. Keep only the feature extraction layers
        self.layers = alexnet_full_model.features

        # 4. Store output channels for each ReLU layer (used by LPIPS)
        self.channels = []
        temp_model = nn.Sequential(*self.layers) # Temporarily wrap layers
        temp_x = torch.zeros(1, 3, 224, 224) # AlexNet standard input size (approx)
        current_channels = 3
        for layer in self.layers:
             # Correctly get out_channels for Conv2d layers
             if isinstance(layer, nn.Conv2d):
                  current_channels = layer.out_channels
             # Store channels *after* ReLU activation which is where LPIPS samples features
             if isinstance(layer, nn.ReLU):
                  self.channels.append(current_channels)


    def forward(self, x):
        """Extract feature maps after ReLU activations."""
        fmaps = []
        for layer in self.layers:
            x = layer(x)
            # Collect feature map after ReLU layers
            if isinstance(layer, nn.ReLU):
                fmaps.append(x)
        return fmaps


class Conv1x1(nn.Module):
    """1x1 Convolution layer used in LPIPS."""
    def __init__(self, in_channels, out_channels=1):
        super().__init__()
        # Original implementation had Dropout(0.5), keep it for consistency
        self.main = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False))

    def forward(self, x):
        return self.main(x)


class LPIPS(nn.Module):
    """LPIPS Model loading weights locally."""
    # --- !! Pass local paths during initialization !! ---
    def __init__(self, alexnet_weights_path=DEFAULT_ALEXNET_WEIGHTS_PATH,
                 lpips_ckpt_path=DEFAULT_LPIPS_CKPT_PATH):
        super().__init__()
        print("Initializing LPIPS Model...")
        # Initialize AlexNet feature extractor using the specified local path
        self.alexnet = AlexNet(weights_path=alexnet_weights_path)

        # Initialize LPIPS layers (1x1 convs)
        self.lpips_weights = nn.ModuleList()
        if hasattr(self.alexnet, 'channels') and self.alexnet.channels: # Check if channels were populated
            for channels in self.alexnet.channels:
                self.lpips_weights.append(Conv1x1(channels, 1))
        else:
             print("Warning: Could not determine AlexNet feature channels. LPIPS layers not initialized.")


        # Load the pretrained LPIPS layer weights from the specified local path
        self._load_lpips_weights(lpips_ckpt_path)

        # Register normalization constants as buffers
        self.register_buffer('mu', torch.tensor([-0.03, -0.088, -0.188]).view(1, 3, 1, 1))
        self.register_buffer('sigma', torch.tensor([0.458, 0.448, 0.450]).view(1, 3, 1, 1))

    def _load_lpips_weights(self, ckpt_path=DEFAULT_LPIPS_CKPT_PATH):
        """Loads LPIPS 1x1 conv weights from local checkpoint."""
        ckpt_path = Path(ckpt_path)
        if ckpt_path.is_file():
            try:
                # Load state dict (can load directly to CPU)
                state_dict = torch.load(ckpt_path, map_location=torch.device('cpu'))
                # Load into the ModuleList
                self.lpips_weights.load_state_dict(state_dict, strict=False) # Use strict=False for flexibility
                print(f"  LPIPS layer weights loaded successfully from: {ckpt_path}")
            except Exception as e:
                 print(f"  Warning: Error loading LPIPS layer weights from {ckpt_path}: {e}.")
                 print("           Using randomly initialized LPIPS layers (results will be invalid).")
        else:
            print(f"  Warning: LPIPS layer weights file not found at {ckpt_path}.")
            print("           Using randomly initialized LPIPS layers (results will be invalid).")


    def forward(self, x, y):
        """Calculates LPIPS distance between images x and y."""
        # Normalize images using ImageNet stats adjusted for [-1, 1] range
        # The constants mu/sigma seem designed for input in [0, 1], then normalized.
        # If input x, y are already in [-1, 1], normalization might need adjustment.
        # Assuming x, y are in [-1, 1] based on StarGAN generator output:
        # No, the original code applies this norm regardless, assuming input is [-1, 1]
        # Let's keep the original normalization logic.
        x_norm = (x - self.mu) / self.sigma
        y_norm = (y - self.mu) / self.sigma

        # Get feature maps from AlexNet
        x_fmaps = self.alexnet(x_norm)
        y_fmaps = self.alexnet(y_norm)

        # Calculate weighted L2 distance in feature space
        lpips_value = 0
        if len(x_fmaps) == len(y_fmaps) == len(self.lpips_weights): # Check for safety
            for x_fmap, y_fmap, conv1x1 in zip(x_fmaps, y_fmaps, self.lpips_weights):
                # Normalize activations before comparison
                x_fmap_norm = normalize(x_fmap)
                y_fmap_norm = normalize(y_fmap)
                # Calculate squared difference, apply 1x1 conv (weight), average
                lpips_value += torch.mean(conv1x1((x_fmap_norm - y_fmap_norm)**2))
        else:
             print("Warning: Mismatch in number of feature maps or LPIPS weights. Returning 0.")

        return lpips_value


@torch.no_grad()
def calculate_lpips_given_images(group_of_images, lpips_model=None, device=None):
    """
    Calculates the average LPIPS distance among a group of images.
    Assumes images in the group correspond to the same input but different styles.

    Args:
        group_of_images (list): A list of image tensors (N, C, H, W), each in range [-1, 1].
        lpips_model (torch.nn.Module, optional): Pre-loaded LPIPS model instance.
                                                If None, a new instance is created.
        device (torch.device, optional): Device to run calculations on. Autodetected if None.

    Returns:
        float: The average LPIPS distance (diversity).
    """
    if len(group_of_images) < 2:
        print("Warning: Need at least 2 images in the group to calculate LPIPS diversity.")
        return 0.0

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load LPIPS model if not provided
    if lpips_model is None:
        try:
            # Use the modified LPIPS class that loads weights locally
            # Ensure default paths at the top are correct or pass them here
            lpips_model = LPIPS().eval().to(device)
            print("LPIPS model loaded internally for calculation.")
        except Exception as e:
             print(f"Error loading LPIPS model internally: {e}")
             return -1.0 # Indicate error

    lpips_values = []
    num_rand_outputs = len(group_of_images)

    # Calculate the average of pairwise distances among all random outputs
    # Move images to device inside the loop if they aren't already
    for i in range(num_rand_outputs - 1):
        img_i = group_of_images[i].to(device)
        for j in range(i + 1, num_rand_outputs):
            img_j = group_of_images[j].to(device)
            try:
                # Calculate LPIPS distance using the loaded model instance
                dist = lpips_model(img_i, img_j)
                # Stack results from the batch if batch size > 1
                lpips_values.append(dist) # Keep batch results separate initially
            except Exception as e_calc:
                 print(f"Error calculating LPIPS between style {i} and {j}: {e_calc}")


    if not lpips_values:
        print("Warning: No LPIPS values were calculated.")
        return 0.0

    # Stack all pairwise distances and compute the mean
    # Each element in lpips_values might be a tensor of size [BatchSize]
    try:
        all_lpips_tensors = torch.stack(lpips_values, dim=0) # Shape [NumPairs, BatchSize] or similar
        lpips_value = torch.mean(all_lpips_tensors) # Calculate mean over all pairs and batches
        return lpips_value.item()
    except Exception as e_mean:
         print(f"Error averaging LPIPS values: {e_mean}")
         # Fallback: Calculate mean from individual items if stacking fails
         all_items = [v.mean().item() for v in lpips_values if torch.is_tensor(v)]
         return np.mean(all_items) if all_items else 0.0