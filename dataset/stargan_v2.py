# pytorch_project/datasets/stargan_v2.py

import os
import random
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np
from PIL import Image
import cv2 # Needed for Albumentations

import torch
from torch.utils import data
from torch.utils.data.sampler import WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++ 添加缺失的函数定义 +++
from pathlib import Path # 确保 pathlib 已导入 (如果之前没有单独导入)

def create_dir(dir_path):
    """如果目录不存在，则创建它。"""
    Path(dir_path).mkdir(parents=True, exist_ok=True) # parents=True允许创建多级目录
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# --- Configuration ---
# Define the 8 known classes and their integer mapping
KNOWN_CLASSES = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']
CLASS_TO_INT_MAPPING = { name: i for i, name in enumerate(KNOWN_CLASSES) }
INT_TO_CLASS_MAPPING = { i: name for i, name in enumerate(KNOWN_CLASSES) }
NUM_DOMAINS = len(KNOWN_CLASSES) # Should be 8
TARGET_LABELS = list(INT_TO_CLASS_MAPPING.keys()) # [0, 1, 2, 3, 4, 5, 6, 7]

# --- Transforms (Consistent with StarGAN v2 Defaults) ---
def get_stargan_transform(img_size=256):
    """
    Defines transforms for StarGAN v2 training:
    Resize, Random Horizontal Flip, ToTensor, Normalize to [-1, 1].
    """
    return A.Compose([
        A.Resize(height=img_size, width=img_size, interpolation=cv2.INTER_AREA),
        A.HorizontalFlip(p=0.5), # Standard augmentation for GANs
        # StarGAN v2 normalization is typically to [-1, 1] for tanh output
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2(), # Converts NumPy HWC to PyTorch CHW Tensor
    ])

# --- Base Dataset Class ---
class ISICBaseDataset(data.Dataset):
    """
    Base class to read ISIC K-Fold CSVs, filter labels,
    and provide access to image paths and labels.
    """
    def __init__(self, csv_path, img_root, transform=None, target_labels=None, label_col='integer_label'):
        """
        Args:
            csv_path (str or Path): Path to the K-Fold CSV file (e.g., train_fold_0.csv).
            img_root (str or Path): Path to the root directory containing the actual image files
                                     (should point to the directory holding the images referenced
                                     by image_path in the CSV, e.g., inpainted_pytorch/train/).
            transform (callable, optional): Albumentations transform to be applied on a sample.
            target_labels (list, optional): List of integer labels to include.
                                             If None, all labels are included.
            label_col (str): Name of the column in the CSV containing integer labels.
        """
        self.csv_path = Path(csv_path)
        self.img_root = Path(img_root) # Store img_root for resolving paths later if needed
        self.transform = transform
        self.target_labels = target_labels if target_labels is not None else []
        self.label_col = label_col

        try:
            df = pd.read_csv(self.csv_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file not found at: {self.csv_path}")

        # --- Filtering based on target_labels ---
        if self.target_labels:
            print(f"Filtering dataset for labels: {self.target_labels}")
            df = df[df[self.label_col].isin(self.target_labels)].reset_index(drop=True)
            if df.empty:
                print(f"Warning: No samples found for target labels {self.target_labels} in {self.csv_path}")

        # Store image paths and labels efficiently
        # Assuming image_path in CSV is relative to img_root or absolute
        # If relative, uncomment the join logic
        self.image_paths = df['image_path'].tolist()
        # self.image_paths = [str(self.img_root / p) for p in df['image_path']] # If paths in CSV are relative
        self.labels = df[self.label_col].tolist()
        self.targets = self.labels # Needed for sampler

        # --- Precompute indices by label (useful for ReferenceDataset) ---
        self.indices_by_label = defaultdict(list)
        for idx, label in enumerate(self.labels):
            self.indices_by_label[label].append(idx)

        print(f"Loaded {len(self.image_paths)} samples from {self.csv_path}.")

    def __len__(self):
        return len(self.image_paths)

    def _load_image(self, index):
        """Loads image at the given index."""
        img_path = self.image_paths[index]
        try:
            # Load using PIL, convert to RGB, then to NumPy for Albumentations
            img = Image.open(img_path).convert('RGB')
            img_np = np.array(img)
            return img_np
        except FileNotFoundError:
            print(f"Warning: Image file not found at: {img_path}. Skipping sample at index {index}.")
            # Return a placeholder or handle appropriately
            # Returning None might require handling in the DataLoader's collate_fn
            return None
        except Exception as e:
            print(f"Warning: Error loading image {img_path}: {e}. Skipping sample at index {index}.")
            return None


# --- Source Dataset ---
class ISICSourceDataset(ISICBaseDataset):
    """
    Dataset for StarGAN v2 'source' input. Returns (image, label).
    Filters for the 8 known classes by default.
    """
    def __init__(self, csv_path, img_root, transform=None, label_col='integer_label'):
        super().__init__(csv_path, img_root, transform=transform, target_labels=TARGET_LABELS, label_col=label_col)

    def __getitem__(self, index):
        label = self.labels[index]
        img_np = self._load_image(index)

        if img_np is None:
             # Handle case where image loading failed (e.g., return a dummy sample or raise error)
             # This basic implementation might cause issues if not handled in collate_fn
             # For simplicity, we might return the first valid sample if one exists
             if len(self) > 0:
                 return self.__getitem__(0) # Recursive call, be careful with empty datasets
             else:
                 raise IndexError("Dataset is empty or cannot load any images.")


        # Apply Albumentations transforms
        if self.transform is not None:
            transformed = self.transform(image=img_np)
            img_tensor = transformed['image']
        else:
            # Basic conversion if no augmentation pipeline is provided
            img_tensor = torch.from_numpy(img_np.transpose((2, 0, 1))).float() / 255.0 # Normalize to [0,1] here? StarGAN expects [-1,1]

        # Ensure output is normalized to [-1, 1] if transform wasn't applied or didn't do it
        if self.transform is None or not isinstance(self.transform.transforms[-2], A.Normalize) or self.transform.transforms[-2].mean != [0.5, 0.5, 0.5]:
             # Basic normalize to [-1, 1] if not done by transform
             img_tensor = (img_tensor * 2.0) - 1.0


        return img_tensor, torch.tensor(label).long() # Return image tensor and long tensor label


# --- Reference Dataset ---
class ISICReferenceDataset(ISICBaseDataset):
    """
    Dataset for StarGAN v2 'reference' input. Returns (image1, image2, label),
    where image1 and image2 are different images from the same domain (label).
    Filters for the 8 known classes by default.
    """
    def __init__(self, csv_path, img_root, transform=None, label_col='integer_label'):
         super().__init__(csv_path, img_root, transform=transform, target_labels=TARGET_LABELS, label_col=label_col)
         if not self.indices_by_label:
              print("Warning: No samples found after filtering, ReferenceDataset will be empty.")

    def __getitem__(self, index):
        label = self.labels[index]
        img1_np = self._load_image(index)

        if img1_np is None:
             # Handle case where image loading failed
             if len(self) > 0:
                  # Try fetching another sample recursively
                  return self.__getitem__((index + 1) % len(self)) # Move to next index
             else:
                  raise IndexError("Dataset is empty or cannot load any images.")

        # --- Find a different image (img2) with the same label ---
        possible_indices = self.indices_by_label.get(label, [])
        if len(possible_indices) > 1:
            # More than one image in this class, find a different one
            index2 = index
            while index2 == index: # Ensure index2 is different from index
                index2 = random.choice(possible_indices)
        elif len(possible_indices) == 1:
            # Only one image in this class, return the same image as img2
             index2 = index
             # print(f"Warning: Only one sample found for label {label}. Using same image for reference.")
        else:
            # Should not happen if init logic is correct and dataset not empty, but handle defensively
            print(f"Error: No indices found for label {label} during getitem for index {index}.")
            # Fallback: maybe use the same image or an image from another class? Using same for now.
            index2 = index

        img2_np = self._load_image(index2)

        if img2_np is None:
            # Handle case where second image loading failed
             print(f"Warning: Could not load reference image 2 (index {index2}) for index {index}. Using img1 instead.")
             img2_np = img1_np.copy() # Use a copy of img1 as fallback

        # Apply transforms to both images independently
        img1_tensor, img2_tensor = None, None
        if self.transform is not None:
            transformed1 = self.transform(image=img1_np)
            img1_tensor = transformed1['image']
            transformed2 = self.transform(image=img2_np)
            img2_tensor = transformed2['image']
        else:
            # Basic conversion if no transform
             img1_tensor = torch.from_numpy(img1_np.transpose((2, 0, 1))).float() / 255.0
             img2_tensor = torch.from_numpy(img2_np.transpose((2, 0, 1))).float() / 255.0
             # Normalize to [-1, 1]
             img1_tensor = (img1_tensor * 2.0) - 1.0
             img2_tensor = (img2_tensor * 2.0) - 1.0


        # Ensure output is normalized to [-1, 1] if transform wasn't applied or didn't do it
        if self.transform is None or not isinstance(self.transform.transforms[-2], A.Normalize) or self.transform.transforms[-2].mean != [0.5, 0.5, 0.5]:
             # Basic normalize to [-1, 1] if not done by transform
             if img1_tensor.max() > 1.0: # Basic check if already normalized
                 img1_tensor = (img1_tensor * 2.0) - 1.0
             if img2_tensor.max() > 1.0:
                 img2_tensor = (img2_tensor * 2.0) - 1.0

        return img1_tensor, img2_tensor, torch.tensor(label).long() # Return two images and label


# --- Helper Function for Balanced Sampler ---
# (Adapted from original StarGAN v2 core/data_loader.py)
def _make_balanced_sampler(labels):
    """Creates a WeightedRandomSampler to balance sampling across classes."""
    if not labels: # Handle empty labels list
        return None
    class_counts = np.bincount(labels)
    # Prevent division by zero if a class has zero samples (though filtering should prevent this)
    class_weights = 1. / np.maximum(class_counts, 1)
    weights = class_weights[labels]
    return WeightedRandomSampler(weights, len(weights), replacement=True)


# --- Function to get DataLoaders ---
def get_stargan_train_loaders(csv_path, img_root, img_size=256, batch_size=8, num_workers=4, label_col='integer_label'):
    """
    Creates the source and reference DataLoaders needed for StarGAN v2 training,
    using balanced sampling.

    Args:
        csv_path (str or Path): Path to the K-Fold training CSV file.
        img_root (str or Path): Path to the root directory containing images.
        img_size (int): Target image size.
        batch_size (int): Training batch size.
        num_workers (int): Number of workers for DataLoader.
        label_col (str): Name of the integer label column in the CSV.

    Returns:
        tuple: (loader_src, loader_ref) DataLoaders for source and reference data.
               Returns (None, None) if datasets are empty.
    """
    print(f"Preparing StarGAN v2 DataLoaders for: {csv_path}")
    transform = get_stargan_transform(img_size)

    # Create source dataset and sampler
    dataset_src = ISICSourceDataset(csv_path, img_root, transform=transform, label_col=label_col)
    if len(dataset_src) == 0: return None, None # Return None if dataset is empty
    sampler_src = _make_balanced_sampler(dataset_src.targets)

    # Create reference dataset and sampler
    dataset_ref = ISICReferenceDataset(csv_path, img_root, transform=transform, label_col=label_col)
    if len(dataset_ref) == 0: return None, None # Return None if dataset is empty
    sampler_ref = _make_balanced_sampler(dataset_ref.targets)

    # Create DataLoaders
    loader_src = data.DataLoader(dataset=dataset_src,
                                 batch_size=batch_size,
                                 sampler=sampler_src,
                                 num_workers=num_workers,
                                 pin_memory=True,
                                 drop_last=True,
                                 shuffle=False) # Sampler handles shuffling

    loader_ref = data.DataLoader(dataset=dataset_ref,
                                 batch_size=batch_size,
                                 sampler=sampler_ref,
                                 num_workers=num_workers,
                                 pin_memory=True,
                                 drop_last=True,
                                 shuffle=False) # Sampler handles shuffling

    print("Source and Reference DataLoaders created.")
    return loader_src, loader_ref

# --- Example Usage (for testing this script directly) ---
if __name__ == '__main__':
    # --- !!! Create dummy data for testing !!! ---
    print("--- Running Example Usage ---")
    dummy_img_root = Path("./dummy_stargan_data/images")
    dummy_csv_dir = Path("./dummy_stargan_data/folds")
    dummy_csv_path = dummy_csv_dir / "train_fold_0.csv"

    # Use the create_dir function defined at the top
    create_dir(dummy_img_root)
    create_dir(dummy_csv_dir)

    # Create dummy CSV
    num_samples_per_class = {0: 5, 1: 20, 2: 8, 3: 4, 4: 10, 5: 3, 6: 2, 7: 6} # Imbalanced classes (8 classes)
    image_paths = []
    labels = []
    img_count = 0
    print("Creating dummy image files and CSV...")
    for label, count in num_samples_per_class.items():
         # Only create data for labels 0-7
         if label in TARGET_LABELS:
            for i in range(count):
                fname = f"image_{img_count:04d}_class{label}.jpg"
                fpath = dummy_img_root / fname
                # Create a small dummy image file
                try:
                    dummy_img = Image.new('RGB', (64, 64), color = (random.randint(0,255), random.randint(0,255), random.randint(0,255)))
                    dummy_img.save(fpath)
                    image_paths.append(str(fpath)) # Store absolute path
                    labels.append(label)
                    img_count += 1
                except Exception as e:
                    print(f"Error creating dummy image {fpath}: {e}")

    if not image_paths:
        print("Failed to create dummy data.")
    else:
        df_dummy = pd.DataFrame({'image_path': image_paths, 'integer_label': labels})
        df_dummy.to_csv(dummy_csv_path, index=False)
        print(f"Dummy CSV created at: {dummy_csv_path}")
        print(f"Dummy images created in: {dummy_img_root}")

        # Test the loaders
        print("\nTesting get_stargan_train_loaders...")
        # Pass img_root as '.' if image_path in CSV is absolute, or the correct root if relative
        loader_src, loader_ref = get_stargan_train_loaders(
            csv_path=dummy_csv_path,
            img_root='.', # Assuming absolute paths in dummy CSV
            img_size=64, # Use smaller size for dummy images
            batch_size=4,
            num_workers=0 # Use 0 for main process debugging
        )

        if loader_src and loader_ref:
            print("\nFetching one batch from source loader...")
            try:
                src_batch_img, src_batch_label = next(iter(loader_src))
                print("Source Batch - Images shape:", src_batch_img.shape) # Expect [4, 3, 64, 64]
                print("Source Batch - Labels:", src_batch_label) # Expect tensor of 4 labels
                print("Source Batch - Image range:", src_batch_img.min().item(), src_batch_img.max().item()) # Expect approx -1 to 1
            except Exception as e:
                print(f"Error fetching source batch: {e}")

            print("\nFetching one batch from reference loader...")
            try:
                ref_batch_img1, ref_batch_img2, ref_batch_label = next(iter(loader_ref))
                print("Reference Batch - Images1 shape:", ref_batch_img1.shape) # Expect [4, 3, 64, 64]
                print("Reference Batch - Images2 shape:", ref_batch_img2.shape) # Expect [4, 3, 64, 64]
                print("Reference Batch - Labels:", ref_batch_label) # Expect tensor of 4 labels
                print("Reference Batch - Image1 range:", ref_batch_img1.min().item(), ref_batch_img1.max().item()) # Expect approx -1 to 1
            except Exception as e:
                print(f"Error fetching reference batch: {e}")

        else:
            print("Failed to create DataLoaders, likely due to empty datasets.")

        # Clean up dummy data (optional)
        # import shutil
        # shutil.rmtree("./dummy_stargan_data")
        # print("\nCleaned up dummy data.")