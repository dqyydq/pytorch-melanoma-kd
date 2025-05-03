# pytorch_project/generation/generate_synthetic_stargan.py

import os
import random
from pathlib import Path
import yaml
from munch import Munch
import copy

import torch
import torch.nn as nn
# from torch.utils.data import Dataset # Not needed for loading single source image
import pandas as pd
import numpy as np
from PIL import Image
import cv2 # For resizing
from tqdm import tqdm

# --- Import project modules ---
# Using absolute imports assuming 'pytorch_project' is the root or accessible
try:
    from model.stargan_v2 import Generator, MappingNetwork
    from utils.helpers import load_config, load_checkpoint
    from utils.stargan_utils import denormalize
    # Import the transform function used during training for source image preprocessing
    from dataset.stargan_v2 import get_stargan_transform
    # Import inpainting function if needed
    # from pytorch_project.preprocessing.inpainting_utils import removeHair_inpainting
except ImportError:
    # Fallback for running directly from generation dir or if structure differs
    print("Import failed using absolute path, trying relative...")
    import sys
    project_root = str(Path(__file__).resolve().parent.parent)
    if project_root not in sys.path:
        sys.path.append(project_root)
        print(f"Added {project_root} to sys.path")

    from model.stargan_v2 import Generator, MappingNetwork
    from utils.helpers import load_config, load_checkpoint
    from utils.stargan_utils import denormalize
    from dataset.stargan_v2 import get_stargan_transform
    try:
        # from preprocessing.inpainting_utils import removeHair_inpainting
        removeHair_inpainting = None
        print("Note: Inpainting function not imported.")
    except ImportError:
         removeHair_inpainting = None
         print("Warning: Could not import removeHair_inpainting. Inpainting step will be skipped.")


# --- Configuration for Generation ---
# --- !! MODIFY THESE VALUES !! ---
CONFIG_PATH = "./pytorch_project/configs/stargan_v2_config.yaml" # Path to your main StarGAN config
CHECKPOINT_ITER = 90000 # Iteration of the checkpoint to load (e.g., final one) <<< VERIFY
SOURCE_DOMAIN_LABEL = 1 # Integer label of the source domain (e.g., 1 for 'NV') <<< VERIFY
NUM_SAMPLES_PER_CLASS = 10 # How many images to generate for each target class <<< SET TO 10
OUTPUT_DIR = "./expr/stargan_v2_isic/generated_samples_per_class" # Directory to save images <<< CHOOSE DIR
APPLY_INPAINTING = True # Whether to apply inpainting to generated images <<< SET True/False
# --- End of user configuration ---

# Mapping from integer label to class name (for saving files clearly)
# Ensure this matches your training setup (8 classes)
INT_TO_CLASS_MAPPING = {
    0: 'MEL', 1: 'NV', 2: 'BCC', 3: 'AK',
    4: 'BKL', 5: 'DF', 6: 'VASC', 7: 'SCC'
}
TARGET_LABELS_TO_GENERATE = list(INT_TO_CLASS_MAPPING.keys()) # Generate for all 8 classes


def generate_per_class_samples(config_path, checkpoint_iter, source_label, num_samples_per_class, output_dir, apply_inpainting):
    """Loads a StarGAN v2 model and generates N samples for each target class."""

    # --- 1. Load Main Config ---
    try:
        config = load_config(config_path)
        args = Munch(config['model_params'])
        train_args = Munch(config['train_params'])
        data_args = Munch(config['data_params'])
        # Ensure parameters used below match config
        IMG_SIZE = args.img_size
        LATENT_DIM = args.latent_dim
        NUM_DOMAINS = args.num_domains
        if NUM_DOMAINS != len(INT_TO_CLASS_MAPPING):
            print(f"Warning: num_domains in config ({NUM_DOMAINS}) does not match INT_TO_CLASS_MAPPING ({len(INT_TO_CLASS_MAPPING)}). Using mapping.")
            NUM_DOMAINS = len(INT_TO_CLASS_MAPPING)

        checkpoint_dir_cfg = Path(train_args.checkpoint_dir)
        img_root_cfg = Path(data_args.img_root_dir)
        fold_csv_dir_cfg = Path(data_args.fold_csv_dir)
        label_col_cfg = data_args.label_column
        print("Configuration loaded.")
    except FileNotFoundError:
        print(f"Error: Config file not found at {config_path}")
        return
    except Exception as e:
        print(f"Error loading or parsing config file {config_path}: {e}")
        return

    # --- 2. Setup Environment ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 3. Build and Load Models (EMA versions) ---
    print("Building models...")
    nets_ema = Munch()
    nets_ema.generator = Generator(IMG_SIZE, args.style_dim, w_hpf=args.w_hpf, max_conv_dim=args.max_conv_dim)
    nets_ema.mapping_network = MappingNetwork(LATENT_DIM, args.style_dim, NUM_DOMAINS)

    print("Loading checkpoint...")
    checkpoint_path = checkpoint_dir_cfg / f"{checkpoint_iter:06d}.ckpt"
    print(f"Attempting to load checkpoint from: {checkpoint_path}")

    try:
        checkpoint = load_checkpoint(checkpoint_path, device)
        if not isinstance(checkpoint, dict):
             print(f"Error: Loaded checkpoint is not a dictionary (type: {type(checkpoint)}).")
             return

        print("Loading EMA model state dictionaries...")
        for name, net_ema in nets_ema.items():
             loaded_from = None; state_dict_to_load = None
             if name in checkpoint.get('nets_ema', {}):
                 state_dict_to_load = checkpoint['nets_ema'][name]; loaded_from = 'nets_ema'
             elif name in checkpoint.get('nets', {}):
                  state_dict_to_load = checkpoint['nets'][name]; loaded_from = 'nets'; print(f"Warn: EMA state for {name} not found, loading from training state.")
             else: print(f"Warn: State for {name} not found in checkpoint.")

             if state_dict_to_load:
                  try: net_ema.load_state_dict(state_dict_to_load); print(f"  Loaded state for nets_ema.{name} from '{loaded_from}'.")
                  except Exception as e_load: print(f"  Error loading state dict for {name}: {e_load}.")

        for net_ema in nets_ema.values(): net_ema.to(device); net_ema.eval()
        print("EMA models loading process finished.")

    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        return
    except Exception as e:
        print(f"Error during checkpoint loading: {e}")
        return

    # --- 4. Prepare ONE Source Image ---
    print(f"Preparing one source image from label {source_label}...")
    source_fold_num = 0
    source_csv_path = fold_csv_dir_cfg / f"train_fold_{source_fold_num}.csv"
    img_root = img_root_cfg
    label_col = label_col_cfg

    if not source_csv_path.exists(): print(f"Error: Source CSV not found: {source_csv_path}"); return
    if not img_root.exists(): print(f"Error: Image root dir not found: {img_root}"); return

    x_src = None
    try:
        df_source = pd.read_csv(source_csv_path)
        df_source = df_source[df_source[label_col] == source_label]
        if df_source.empty: print(f"Error: No images for source label {source_label} in {source_csv_path}"); return

        # Get the first valid image path
        source_img_path_str = None
        for p_str in df_source['image_path'].tolist():
            p = Path(p_str)
            if not p.is_absolute(): p = img_root / p # Handle relative paths
            if p.exists(): source_img_path_str = str(p); break # Found one

        if not source_img_path_str: print(f"Error: No existing source image found for label {source_label}."); return

        # Load and preprocess the single source image
        print(f"Loading source image: {source_img_path_str}")
        img = Image.open(source_img_path_str).convert('RGB')
        img_np = np.array(img)
        transform = get_stargan_transform(IMG_SIZE) # Get the correct transform
        transformed = transform(image=img_np)
        x_src = transformed['image'].unsqueeze(0).to(device) # Add batch dim and move to device
        print(f"Source image prepared, shape: {x_src.shape}")

    except Exception as e:
        print(f"Error preparing source image: {e}")
        return

    # --- 5. Generation Loop ---
    output_path_base = Path(output_dir)
    output_path_base.mkdir(parents=True, exist_ok=True)
    print(f"Generating {num_samples_per_class} images for each of {len(TARGET_LABELS_TO_GENERATE)} target labels to: {output_path_base}")

    total_generated = 0
    # Wrap outer loop with tqdm for class progress
    for target_label in tqdm(TARGET_LABELS_TO_GENERATE, desc="Generating Classes"):
        target_label_name = INT_TO_CLASS_MAPPING.get(target_label, f"label{target_label}")
        y_trg = torch.tensor([target_label]).to(device) # Target label tensor (batch size 1)

        # Generate N samples for this target class
        for i in range(num_samples_per_class):
            with torch.no_grad():
                # Generate a different random latent code for each sample
                z_trg = torch.randn(1, LATENT_DIM).to(device) # Batch size 1
                # Calculate style code
                s_trg = nets_ema.mapping_network(z_trg, y_trg)
                # Generate image
                x_fake = nets_ema.generator(x_src, s_trg) # x_src is already on device

            # Post-process the generated image
            img_np = (denormalize(x_fake.squeeze(0)).cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8) # Remove batch dim

            # Apply inpainting if configured
            if apply_inpainting and removeHair_inpainting:
                try:
                    img_np = removeHair_inpainting(img_np)
                except Exception as e_inp:
                    print(f"Warning: Inpainting failed for generated image {target_label_name}_{i:04d}: {e_inp}")

            # Convert to PIL and save
            img_pil = Image.fromarray(img_np)
            filename = output_path_base / f"synth_{target_label_name}_{i:04d}.jpg" # Consistent naming
            try:
                img_pil.save(filename)
                total_generated += 1
            except Exception as e_save:
                print(f"Warning: Failed to save image {filename}: {e_save}")

    print(f"\nFinished generating demo samples. Total images saved: {total_generated}")


# --- Entry Point ---
if __name__ == "__main__":
    # --- Configuration ---
    stargan_config_file = "D:/python_code/pytorch_melanama_kd/configs/stargan_v2_config.yaml"
    output_directory = "D:/python_code/pytorch_melanama_kd/training/expr/stargan_v2_isic/generated_samples_per_class" # Output directory

    # --- !! MODIFY THESE AS NEEDED !! ---
    checkpoint_iteration =  90000  # <<< !!! Iteration number of the checkpoint to load !!!
    source_image_label = 1      # <<< Integer label of the source domain (e.g., 1 for NV)
    samples_per_target_class = 10 # <<< Generate 10 images per target class
    apply_hair_removal = True   # <<< Set to False to skip inpainting

    # --- Run Generation ---
    if not Path(stargan_config_file).exists():
        print(f"Error: StarGAN v2 config file not found at {stargan_config_file}")
    else:
        generate_per_class_samples(
            config_path=stargan_config_file,
            checkpoint_iter=checkpoint_iteration,
            source_label=source_image_label,
            num_samples_per_class=samples_per_target_class,
            output_dir=output_directory,
            apply_inpainting=apply_hair_removal
        )