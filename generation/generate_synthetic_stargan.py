# pytorch_project/generation/generate_synthetic_stargan.py

import os
import random
from pathlib import Path
import yaml
from munch import Munch
import copy
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from PIL import Image
import cv2 # For potential inpainting
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
# --- Import project modules ---
try:
    # Ensure correct relative paths if running from generation dir or project root
    from model.stargan_v2 import Generator, MappingNetwork # We mainly need G and F
    # Assuming helpers.py is in utils directory
    from utils.helpers import load_config, load_checkpoint
    # Import denormalize if needed for saving/viewing, and save_image might be useful
    from utils.stargan_utils import denormalize, save_image
    # Import inpainting function if needed
    from preprocessing.inpaint import removeHair_inpainting # Assuming you put it here
except ImportError:
    print("Import failed using absolute path, trying relative...")
    import sys
    project_root = str(Path(__file__).resolve().parent.parent)
    if project_root not in sys.path:
        sys.path.append(project_root)
        print(f"Added {project_root} to sys.path")

    from model.stargan_v2 import Generator, MappingNetwork
    from utils.helpers import load_config, load_checkpoint
    from utils.stargan_utils import denormalize #, save_image # Optional for direct saving
    # Make sure this path is correct based on where you put the inpaint function
    try:
        from preprocessing.inpaint import removeHair_inpainting
    except ImportError:
         print("Warning: Could not import removeHair_inpainting. Inpainting step will be skipped.")
         removeHair_inpainting = None # Define as None if not found


# --- Simple Dataset for Loading Source Images ---
class SourceImageDataset(Dataset):
    def __init__(self, image_paths, img_size, normalize=True):
        self.image_paths = image_paths
        self.img_size = img_size
        self.normalize = normalize # Normalize to [-1, 1] for StarGAN input
        # Basic transform: Resize, ToTensor, Normalize
        self.transform = A.Compose([
            A.Resize(height=img_size, width=img_size, interpolation=cv2.INTER_AREA),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) if normalize else A.NoOp(),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            img_np = np.array(img)
            transformed = self.transform(image=img_np)
            return transformed['image']
        except Exception as e:
            print(f"Warning: Error loading source image {img_path}: {e}")
            # Return a dummy tensor of the correct shape if loading fails
            return torch.zeros((3, self.img_size, self.img_size))

# --- Main Generation Function ---
def generate_synthetic_images(config_path="configs/stargan_v2_config.yaml", generation_config_path="configs/generation_config.yaml"):
    """Generates synthetic images using a trained StarGAN v2 model."""

    # --- 1. Load Configurations ---
    try:
        config = load_config(config_path)
        args = Munch(config['model_params'])
        train_args = Munch(config['train_params']) # Needed for checkpoint path
        data_args = Munch(config['data_params']) # Needed for num_domains etc.

        gen_config = load_config(generation_config_path)
        gen_args = Munch(gen_config['generation_params'])
        print("Configurations loaded.")
    except Exception as e:
        print(f"Error loading config files: {e}")
        return

    # --- 2. Setup Environment ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # No need for full seeding here unless you want deterministic random styles

    # --- 3. Build and Load Models (EMA versions) ---
    print("Building models...")
    # We mainly need Generator_ema and MappingNetwork_ema
    nets_ema = Munch()
    nets_ema.generator = Generator(args.img_size, args.style_dim, w_hpf=args.w_hpf, max_conv_dim=args.max_conv_dim)
    nets_ema.mapping_network = MappingNetwork(args.latent_dim, args.style_dim, args.num_domains)
    # Style Encoder might be needed if using reference-based generation, but latent is more common for augmentation
    # nets_ema.style_encoder = StyleEncoder(args.img_size, args.style_dim, args.num_domains, max_conv_dim=args.max_conv_dim)

    print("Loading checkpoint...")
    checkpoint_dir = Path(train_args.checkpoint_dir)
    # Use the resume_iter specified in generation config, or default to total_iters from training
    resume_iter = gen_args.get('resume_iter', train_args.total_iters)
    checkpoint_path = checkpoint_dir / f"{resume_iter:06d}.ckpt"

    if not checkpoint_path.exists():
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        return

    try:
        checkpoint = load_checkpoint(checkpoint_path, device) # From helpers.py
        # Load EMA states
        for name, net_ema in nets_ema.items():
            if name in checkpoint.get('nets_ema', {}):
                 net_ema.load_state_dict(checkpoint['nets_ema'][name])
                 print(f"Loaded state for nets_ema.{name}")
            else:
                 # Fallback: try loading from 'nets' if 'nets_ema' is missing
                 if name in checkpoint.get('nets', {}):
                     print(f"Warning: EMA state for {name} not found, loading from training state.")
                     net_ema.load_state_dict(checkpoint['nets'][name])
                 else:
                     print(f"Warning: State for nets_ema.{name} not found in checkpoint. Using initialized weights.")

        # Move to device and set to eval mode
        for net_ema in nets_ema.values():
             net_ema.to(device)
             net_ema.eval()
        print("EMA models loaded successfully.")
    except Exception as e:
        print(f"Error loading checkpoint states: {e}")
        return

    # --- 4. Prepare Source Images ---
    print("Preparing source images...")
    # Option 1: Use images from a specific fold CSV
    source_fold_num = gen_args.get('source_fold_num', 0)
    source_csv_path = Path(data_args.fold_csv_dir) / f"train_fold_{source_fold_num}.csv"
    img_root = Path(data_args.img_root_dir)
    source_label = gen_args.source_domain_label # e.g., 1 for 'NV'
    label_col = data_args.label_column

    try:
        df_source = pd.read_csv(source_csv_path)
        # Filter for the source domain label
        df_source = df_source[df_source[label_col] == source_label]
        if df_source.empty:
            print(f"Error: No images found for source label {source_label} in {source_csv_path}")
            return
        source_image_paths = df_source['image_path'].tolist()
        # Make paths absolute if they are relative in the CSV
        # source_image_paths = [str(img_root / p) if not Path(p).is_absolute() else p for p in source_image_paths]
        print(f"Found {len(source_image_paths)} source images for label {source_label}.")
    except FileNotFoundError:
        print(f"Error: Source CSV not found at {source_csv_path}")
        return
    except KeyError:
         print(f"Error: Column '{label_col}' or 'image_path' not found in {source_csv_path}")
         return

    # Limit the number of source images if specified
    num_source_imgs = gen_args.get('num_source_images', len(source_image_paths))
    if num_source_imgs < len(source_image_paths):
        source_image_paths = random.sample(source_image_paths, num_source_imgs)
        print(f"Using a random subset of {num_source_imgs} source images.")

    source_dataset = SourceImageDataset(source_image_paths, args.img_size)
    source_loader = DataLoader(source_dataset, batch_size=gen_args.batch_size, shuffle=False, num_workers=train_args.num_workers)

    # --- 5. Generation Loop ---
    output_dir = Path(gen_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generating synthetic images to: {output_dir}")

    generated_count = defaultdict(int)
    total_generated_for_all_targets = 0

    # Determine target labels (all domains except the source domain)
    target_labels_to_generate = gen_args.get('target_domain_labels',
                                             [l for l in range(args.num_domains) if l != source_label])
    print(f"Target labels for generation: {target_labels_to_generate}")

    num_batches = len(source_loader)
    total_imgs_to_generate = len(target_labels_to_generate) * len(source_dataset) * gen_args.num_styles_per_source
    pbar_desc = "Generating Images"
    pbar_gen = tqdm(total=total_imgs_to_generate, desc=pbar_desc)

    # Generate N different styles for each source image per target domain
    for style_idx in range(gen_args.num_styles_per_source):
        print(f"\nGenerating style #{style_idx+1}/{gen_args.num_styles_per_source}...")
        iter_source_loader = iter(source_loader) # Recreate iterator for each style pass

        for batch_idx in range(num_batches):
            try:
                x_src_batch = next(iter_source_loader)
                x_src_batch = x_src_batch.to(device)
                current_batch_size = x_src_batch.size(0)

                # Generate latent code for this style pass (same z for all targets in this batch)
                z_trg_batch = torch.randn(current_batch_size, args.latent_dim).to(device)

                # Generate for each target label
                for target_label in target_labels_to_generate:
                    # Create target label tensor for the batch
                    y_trg_batch = torch.tensor([target_label] * current_batch_size).to(device)

                    with torch.no_grad():
                        # Get style code from mapping network
                        s_trg_batch = nets_ema.mapping_network(z_trg_batch, y_trg_batch)
                        # Generate fake image
                        x_fake_batch = nets_ema.generator(x_src_batch, s_trg_batch)

                    # Post-process and save each image in the batch
                    for i in range(current_batch_size):
                        x_fake = x_fake_batch[i]
                        # Denormalize from [-1, 1] to [0, 1], then to [0, 255] uint8
                        img_np = (denormalize(x_fake).cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)

                        # Apply inpainting if configured and function available
                        if gen_args.apply_inpainting and removeHair_inpainting:
                            try:
                                img_np = removeHair_inpainting(img_np) # Assumes RGB input/output
                            except Exception as e_inp:
                                print(f"Warning: Inpainting failed for an image: {e_inp}")

                        # Convert NumPy array (HWC, RGB) to PIL Image
                        img_pil = Image.fromarray(img_np)

                        # Construct filename
                        count = generated_count[target_label]
                        # Format: synth_{target_label_name}_{count}.jpg
                        target_label_name = data_args.get('int_to_class_mapping', {}).get(target_label, f"label{target_label}") # Get name if possible
                        filename = output_dir / f"synth_{target_label_name}_{count:05d}.jpg"

                        # Save the image
                        try:
                            img_pil.save(filename)
                            generated_count[target_label] += 1
                            total_generated_for_all_targets += 1
                            pbar_gen.update(1)
                            pbar_gen.set_description(f"{pbar_desc} (Saved {filename.name})")
                        except Exception as e_save:
                            print(f"Warning: Failed to save image {filename}: {e_save}")

            except StopIteration:
                break # End of source loader for this style pass
            except Exception as e_batch:
                 print(f"Error processing batch {batch_idx}: {e_batch}")
                 # Continue to next batch or handle more robustly

    pbar_gen.close()
    print("\nSynthetic image generation finished.")
    print("Generated counts per label:")
    for label, count in generated_count.items():
        print(f"  Label {label}: {count} images")
    print(f"Total generated: {total_generated_for_all_targets} images.")


# --- Entry Point ---
if __name__ == "__main__":
    # --- !!! TODO: Create a generation_config.yaml file !!! ---
    # Example YAML structure (save as configs/generation_config.yaml):
    # generation_params:
    #   resume_iter: 100000         # Iteration number of the checkpoint to load (e.g., final iteration)
    #   source_domain_label: 1     # Integer label of the source domain (e.g., 1 for 'NV')
    #   # target_domain_labels: [0, 2, 3, 4, 5, 6, 7] # Optional: List of target labels (defaults to all except source)
    #   num_source_images: -1      # Number of source images to use (-1 for all available in the source fold)
    #   source_fold_num: 0         # Which fold CSV to use for source images
    #   num_styles_per_source: 1   # How many different styles (latent codes z) to generate per source image
    #                              # Set > 1 for more diversity, but more images
    #   batch_size: 16             # Batch size for generation (adjust based on GPU memory)
    #   output_dir: './pytorch_project/data/ISIC2019/synthetic_stargan_pytorch/train' # Output directory
    #   apply_inpainting: True     # Whether to apply hair removal to generated images

    # Paths relative to the script location or project root
    script_dir = Path(__file__).parent
    project_root_dir = script_dir.parent
    default_stargan_config_path = project_root_dir / "configs" / "stargan_v2_config.yaml"
    default_generation_config_path = project_root_dir / "configs" / "generation_config.yaml"

    # Check if config files exist
    if not default_stargan_config_path.exists():
        print(f"Error: StarGAN v2 training config not found at {default_stargan_config_path}")
    elif not default_generation_config_path.exists():
        print(f"Error: Generation config not found at {default_generation_config_path}")
        print("Please create a YAML config file (see example structure in script comments).")
    else:
        generate_synthetic_images(
            config_path=default_stargan_config_path,
            generation_config_path=default_generation_config_path
        )