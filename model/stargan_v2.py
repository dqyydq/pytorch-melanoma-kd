# pytorch_project/models/stargan_v2.py
"""
StarGAN v2 Model Architectures adapted from the official implementation.
Copyright (c) 2020-present NAVER Corp.
"""

import copy
import math

# from munch import Munch # Munch is not needed here if build_model is removed
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Note: FAN (Facial Alignment Network) related imports and usage are removed
# from core.wing import FAN # Removed


class ResBlk(nn.Module):
    """Residual block."""
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample=False):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = downsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        if self.normalize:
            # Use InstanceNorm for style transfer tasks
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            # Use average pooling for downsampling
            x = F.avg_pool2d(x, 2)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        # Combine shortcut and residual, apply scaling for unit variance
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)


class AdaIN(nn.Module):
    """Adaptive Instance Normalization."""
    def __init__(self, style_dim, num_features):
        super().__init__()
        # Instance norm without learnable affine parameters
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        # Linear layer to predict scale (gamma) and bias (beta) from style code
        self.fc = nn.Linear(style_dim, num_features*2)

    def forward(self, x, s):
        # Predict gamma and beta from style vector s
        h = self.fc(s)
        # Reshape to match spatial dimensions for broadcasting
        h = h.view(h.size(0), h.size(1), 1, 1)
        # Split into gamma and beta
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        # Apply AdaIN: norm(x) * (1 + gamma) + beta
        return (1 + gamma) * self.norm(x) + beta


class AdainResBlk(nn.Module):
    """Residual block with AdaIN."""
    def __init__(self, dim_in, dim_out, style_dim=64, w_hpf=0,
                 actv=nn.LeakyReLU(0.2), upsample=False):
        super().__init__()
        self.w_hpf = w_hpf # High-pass filter weight (set to 0 for our case)
        self.actv = actv
        self.upsample = upsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim)

    def _build_weights(self, dim_in, dim_out, style_dim=64):
        self.conv1 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, 1, 1)
        # Use AdaIN for normalization, controlled by style vector s
        self.norm1 = AdaIN(style_dim, dim_in)
        self.norm2 = AdaIN(style_dim, dim_out)
        if self.learned_sc:
            # 1x1 convolution for shortcut if dimensions change
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.upsample:
            # Use nearest neighbor interpolation for upsampling
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        x = self.norm1(x, s) # Apply AdaIN using style s
        x = self.actv(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv1(x)
        x = self.norm2(x, s) # Apply AdaIN using style s
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        # If not using high-pass filter, combine shortcut and residual
        if self.w_hpf == 0:
            out = (out + self._shortcut(x)) / math.sqrt(2)
        # If w_hpf > 0, the HighPass module would be applied externally in the Generator
        return out


class HighPass(nn.Module):
    """High-pass filter using a fixed Laplacian kernel."""
    def __init__(self, w_hpf, device):
        super(HighPass, self).__init__()
        self.filter = torch.tensor([[-1, -1, -1],
                                    [-1, 8., -1],
                                    [-1, -1, -1]]).to(device) / w_hpf
        # Make filter non-trainable
        self.filter = nn.Parameter(self.filter, requires_grad=False)


    def forward(self, x):
        # Expand filter to match input channels and apply grouped convolution
        filter = self.filter.unsqueeze(0).unsqueeze(1).repeat(x.size(1), 1, 1, 1)
        return F.conv2d(x, filter, padding=1, groups=x.size(1))


class Generator(nn.Module):
    """StarGAN v2 Generator Network."""
    def __init__(self, img_size=256, style_dim=64, max_conv_dim=512, w_hpf=0): # Set w_hpf default to 0
        super().__init__()
        # Calculate initial dimension based on image size
        dim_in = 2**14 // img_size
        self.img_size = img_size
        # Initial convolution layer mapping RGB to internal dimension
        self.from_rgb = nn.Conv2d(3, dim_in, 3, 1, 1)
        self.encode = nn.ModuleList() # Encoder blocks
        self.decode = nn.ModuleList() # Decoder blocks
        # Final layers to map back to RGB, includes Tanh activation
        self.to_rgb = nn.Sequential(
            nn.InstanceNorm2d(dim_in, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim_in, 3, 1, 1, 0),
            nn.Tanh() # Ensure output is in [-1, 1] range
            )

        # down/up-sampling blocks
        # Number of down/up-sampling blocks depends on image size
        repeat_num = int(np.log2(img_size)) - 4
        # Removed conditional increment for w_hpf as we assume w_hpf=0
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            # Add encoder block (ResBlk with downsampling)
            self.encode.append(
                ResBlk(dim_in, dim_out, normalize=True, downsample=True))
            # Add corresponding decoder block (AdainResBlk with upsampling)
            # Insert at the beginning to reverse the order
            self.decode.insert(
                0, AdainResBlk(dim_out, dim_in, style_dim,
                               w_hpf=w_hpf, upsample=True))
            dim_in = dim_out # Update dimension for next block

        # bottleneck blocks (no dimension change or down/up-sampling)
        for _ in range(2):
            self.encode.append(
                ResBlk(dim_out, dim_out, normalize=True))
            self.decode.insert(
                0, AdainResBlk(dim_out, dim_out, style_dim, w_hpf=w_hpf))

        # Removed HighPass filter initialization as w_hpf=0
        # if w_hpf > 0:
        #     device = torch.device(
        #         'cuda' if torch.cuda.is_available() else 'cpu')
        #     self.hpf = HighPass(w_hpf, device)

    # Simplified forward pass, removing masks and cache logic
    def forward(self, x, s): # Removed masks argument
        x = self.from_rgb(x)
        # Encoder pass
        for block in self.encode:
            x = block(x)
        # Decoder pass, applying style code s in AdainResBlk
        for block in self.decode:
            x = block(x, s)
        # Map back to RGB and apply Tanh
        return self.to_rgb(x)


class MappingNetwork(nn.Module):
    """Maps latent code z and domain label y to style code s."""
    def __init__(self, latent_dim=16, style_dim=64, num_domains=2):
        super().__init__()
        layers = []
        # Shared MLP layers
        layers += [nn.Linear(latent_dim, 512)]
        layers += [nn.ReLU()]
        for _ in range(3):
            layers += [nn.Linear(512, 512)]
            layers += [nn.ReLU()]
        self.shared = nn.Sequential(*layers)

        # Unshared (domain-specific) MLP layers
        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared += [nn.Sequential(nn.Linear(512, 512),
                                            nn.ReLU(),
                                            nn.Linear(512, 512),
                                            nn.ReLU(),
                                            nn.Linear(512, 512),
                                            nn.ReLU(),
                                            nn.Linear(512, style_dim))]

    def forward(self, z, y):
        # Pass latent code through shared layers
        h = self.shared(z)
        # Pass through domain-specific layers and stack results
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        out = torch.stack(out, dim=1)  # (batch, num_domains, style_dim)
        # Select the style code corresponding to the target domain label y
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        s = out[idx, y]  # Select using advanced indexing -> (batch, style_dim)
        return s


class StyleEncoder(nn.Module):
    """Extracts style code s from image x given domain label y."""
    def __init__(self, img_size=256, style_dim=64, num_domains=2, max_conv_dim=512):
        super().__init__()
        # Calculate initial dimension
        dim_in = 2**14 // img_size
        blocks = []
        blocks += [nn.Conv2d(3, dim_in, 3, 1, 1)]

        # Downsampling blocks using ResBlk
        repeat_num = int(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out

        # Final convolutional layers
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)] # 4x4 conv, stride 1, no padding -> reduces spatial dim
        blocks += [nn.LeakyReLU(0.2)]
        # Shared convolutional backbone
        self.shared = nn.Sequential(*blocks)

        # Unshared (domain-specific) linear layers to produce style code
        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared += [nn.Linear(dim_out, style_dim)]

    def forward(self, x, y):
        # Pass image through shared backbone
        h = self.shared(x)
        # Flatten the spatial dimensions
        h = h.view(h.size(0), -1)
        # Pass through domain-specific linear layers
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        out = torch.stack(out, dim=1)  # (batch, num_domains, style_dim)
        # Select the style code for the given domain label y
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        s = out[idx, y]  # (batch, style_dim)
        return s


class Discriminator(nn.Module):
    """StarGAN v2 Discriminator Network."""
    def __init__(self, img_size=256, num_domains=2, max_conv_dim=512):
        super().__init__()
        # Calculate initial dimension
        dim_in = 2**14 // img_size
        blocks = []
        blocks += [nn.Conv2d(3, dim_in, 3, 1, 1)]

        # Downsampling blocks using ResBlk
        repeat_num = int(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out

        # Final layers to produce per-domain outputs
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)] # 4x4 conv, stride 1, no padding
        blocks += [nn.LeakyReLU(0.2)]
        # Final 1x1 convolution producing one output channel per domain
        blocks += [nn.Conv2d(dim_out, num_domains, 1, 1, 0)]
        self.main = nn.Sequential(*blocks)

    def forward(self, x, y):
        # Pass image through the network
        out = self.main(x)
        # Flatten spatial dimensions
        out = out.view(out.size(0), -1)  # (batch, num_domains)
        # Select the output corresponding to the domain label y
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        out = out[idx, y]  # Select using advanced indexing -> (batch)
        return out

# Note: The build_model function is removed as models will be instantiated directly
# in the training script. FAN network related logic is also removed.