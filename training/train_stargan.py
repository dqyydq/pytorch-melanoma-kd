# # pytorch_project/training/train_stargan.py

# import os
# import time
# import datetime
# from pathlib import Path
# import yaml # For loading config
# from munch import Munch # Similar to argparse.Namespace, for easy config access
# import random
# import numpy as np
# import copy

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# from tqdm import tqdm # Progress bar

# # --- Import project modules ---
# # Using absolute imports assuming 'pytorch_project' is the root or accessible
# try:
#     from dataset.stargan_v2 import get_stargan_train_loaders
#     from model.stargan_v2 import Generator, MappingNetwork, StyleEncoder, Discriminator
#     from utils.stargan_utils import (
#         he_init, print_network, save_image, denormalize,
#         moving_average, adv_loss, r1_reg
#     )
#     # Assuming helpers.py is in utils directory
#     from utils.helpers import load_config, save_checkpoint, load_checkpoint
# except ImportError:
#     # Fallback for running directly from training dir or if structure differs
#     print("Import failed using absolute path, trying relative...")
#     import sys
#     project_root = str(Path(__file__).resolve().parent.parent)
#     if project_root not in sys.path:
#         sys.path.append(project_root)
#         print(f"Added {project_root} to sys.path")

#     from dataset.stargan_v2 import get_stargan_train_loaders
#     from model.stargan_v2 import Generator, MappingNetwork, StyleEncoder, Discriminator
#     from utils.stargan_utils import (
#         he_init, print_network, save_image, denormalize,
#         moving_average, adv_loss, r1_reg
#     )
#     from utils.helpers import load_config, save_checkpoint, load_checkpoint

# # --- Loss Computation Helper Functions ---

# def compute_g_loss(nets, args, x_real, y_org, y_trg, z_trgs=None, x_refs=None):
#     """Computes generator loss components."""
#     assert (z_trgs is None) != (x_refs is None)
#     if z_trgs is not None:
#         z_trg, z_trg2 = z_trgs
#     if x_refs is not None:
#         x_ref, x_ref2 = x_refs

#     # Adversarial loss
#     if z_trgs is not None:
#         s_trg = nets.mapping_network(z_trg, y_trg)
#     else: # x_refs is not None
#         s_trg = nets.style_encoder(x_ref, y_trg)

#     x_fake = nets.generator(x_real, s_trg)
#     out = nets.discriminator(x_fake, y_trg)
#     loss_adv = adv_loss(out, 1)

#     # Style reconstruction loss
#     s_pred = nets.style_encoder(x_fake, y_trg)
#     loss_sty = torch.mean(torch.abs(s_pred - s_trg))

#     # Diversity sensitive loss
#     if z_trgs is not None:
#         s_trg2 = nets.mapping_network(z_trg2, y_trg)
#     else: # x_refs is not None
#         s_trg2 = nets.style_encoder(x_ref2, y_trg)
#     x_fake2 = nets.generator(x_real, s_trg2)
#     x_fake2 = x_fake2.detach()
#     loss_ds = torch.mean(torch.abs(x_fake - x_fake2))

#     # Cycle-consistency/Reconstruction loss
#     s_org = nets.style_encoder(x_real, y_org)
#     x_rec = nets.generator(x_fake, s_org)
#     loss_cyc = torch.mean(torch.abs(x_rec - x_real))

#     # Total generator loss
#     loss = (loss_adv +
#             args.lambda_sty * loss_sty -
#             args.lambda_ds * loss_ds + # Note the minus sign for diversity
#             args.lambda_cyc * loss_cyc)

#     return loss, Munch(adv=loss_adv.item(),
#                        sty=loss_sty.item(),
#                        ds=loss_ds.item(),
#                        cyc=loss_cyc.item())


# def compute_d_loss(nets, args, x_real, y_org, y_trg, z_trg=None, x_ref=None):
#     """Computes discriminator loss components."""
#     assert (z_trg is None) != (x_ref is None)
#     # Loss with real images
#     x_real.requires_grad_()
#     out = nets.discriminator(x_real, y_org)
#     loss_real = adv_loss(out, 1)
#     loss_reg = r1_reg(out, x_real)

#     # Loss with fake images
#     with torch.no_grad():
#         if z_trg is not None:
#             s_trg = nets.mapping_network(z_trg, y_trg)
#         else:  # x_ref is not None
#             s_trg = nets.style_encoder(x_ref, y_trg)
#         x_fake = nets.generator(x_real, s_trg)

#     out = nets.discriminator(x_fake, y_trg)
#     loss_fake = adv_loss(out, 0)

#     # Total discriminator loss
#     loss = loss_real + loss_fake + args.lambda_reg * loss_reg
#     return loss, Munch(real=loss_real.item(),
#                        fake=loss_fake.item(),
#                        reg=loss_reg.item())


# # --- Main Training Function ---
# def train_stargan_v2(config_path="configs/stargan_v2_config.yaml"):
#     """Main function to train the StarGAN v2 model."""

#     # --- 1. Load Configuration ---
#     try:
#         config = load_config(config_path)
#         # Use Munch to allow attribute-style access (like args.img_size)
#         args = Munch(config['model_params'])
#         train_args = Munch(config['train_params'])
#         data_args = Munch(config['data_params'])
#         print("Configuration loaded successfully.")
#         # print("Model Args:", args)
#         # print("Train Args:", train_args)
#         # print("Data Args:", data_args)
#     except Exception as e:
#         print(f"Error loading or parsing config file {config_path}: {e}")
#         return

#     # --- 2. Setup Environment ---
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")
#     # Set random seed for reproducibility
#     torch.manual_seed(train_args.seed)
#     random.seed(train_args.seed)
#     np.random.seed(train_args.seed)

#     # --- 3. Build Models and EMA Models ---
#     print("Building models...")
#     nets = Munch()
#     try:
#         nets.generator = Generator(args.img_size, args.style_dim, w_hpf=args.w_hpf, max_conv_dim=args.max_conv_dim)
#         nets.mapping_network = MappingNetwork(args.latent_dim, args.style_dim, args.num_domains)
#         nets.style_encoder = StyleEncoder(args.img_size, args.style_dim, args.num_domains, max_conv_dim=args.max_conv_dim)
#         nets.discriminator = Discriminator(args.img_size, args.num_domains, max_conv_dim=args.max_conv_dim)
#     except Exception as e:
#         print(f"Error building models: {e}")
#         return

#     nets_ema = Munch()
#     nets_ema.generator = copy.deepcopy(nets.generator)
#     nets_ema.mapping_network = copy.deepcopy(nets.mapping_network)
#     nets_ema.style_encoder = copy.deepcopy(nets.style_encoder)

#     # Move models to device and print structure
#     for name, net in nets.items():
#         print_network(net, name)
#         net.to(device)
#     for name, net_ema in nets_ema.items():
#         net_ema.to(device)
#         # EMA networks are used for evaluation/inference, set to eval mode
#         net_ema.eval()

#     # --- 4. Setup Optimizers ---
#     print("Setting up optimizers...")
#     optims = Munch()
#     # Use try-except for robustness in case parameters are missing
#     try:
#         optims.generator = torch.optim.Adam(
#             nets.generator.parameters(),
#             lr=train_args.lr, betas=(train_args.beta1, train_args.beta2),
#             weight_decay=train_args.weight_decay
#         )
#         optims.mapping_network = torch.optim.Adam(
#             nets.mapping_network.parameters(),
#             lr=train_args.f_lr, betas=(train_args.beta1, train_args.beta2),
#             weight_decay=train_args.weight_decay
#         )
#         optims.style_encoder = torch.optim.Adam(
#             nets.style_encoder.parameters(),
#             lr=train_args.lr, betas=(train_args.beta1, train_args.beta2),
#             weight_decay=train_args.weight_decay
#         )
#         optims.discriminator = torch.optim.Adam(
#             nets.discriminator.parameters(),
#             lr=train_args.lr, betas=(train_args.beta1, train_args.beta2),
#             weight_decay=train_args.weight_decay
#         )
#     except AttributeError as e:
#         print(f"Error setting up optimizers. Missing parameter in config? {e}")
#         return

#     # --- 5. Prepare Checkpoint Directories and Load (if resuming) ---
#     checkpoint_dir = Path(train_args.checkpoint_dir)
#     sample_dir = Path(train_args.sample_dir)
#     # Use exist_ok=True to avoid errors if directories already exist
#     checkpoint_dir.mkdir(parents=True, exist_ok=True)
#     sample_dir.mkdir(parents=True, exist_ok=True)

#     start_iter = 0
#     if train_args.resume_iter > 0:
#         print(f"Attempting to resume training from iteration {train_args.resume_iter}...")
#         checkpoint_path = checkpoint_dir / f"{train_args.resume_iter:06d}.ckpt"

#         if checkpoint_path.exists():
#             try:
#                 # Load the entire checkpoint dictionary
#                 # Assumes load_checkpoint is implemented in helpers.py
#                 checkpoint = load_checkpoint(checkpoint_path, device)

#                 # Load model states
#                 print("Loading model state dictionaries...")
#                 for name, net in nets.items():
#                     if name in checkpoint.get('nets', {}):
#                         net.load_state_dict(checkpoint['nets'][name])
#                         print(f"  Loaded state for nets.{name}")
#                     else:
#                         print(f"  Warning: State for nets.{name} not found in checkpoint.")

#                 # Load EMA model states
#                 print("Loading EMA model state dictionaries...")
#                 for name, net_ema in nets_ema.items():
#                     ema_key_in_ckpt = name # Assume keys match
#                     if ema_key_in_ckpt in checkpoint.get('nets_ema', {}):
#                         net_ema.load_state_dict(checkpoint['nets_ema'][ema_key_in_ckpt])
#                         print(f"  Loaded state for nets_ema.{name}")
#                     else:
#                         print(f"  Warning: State for nets_ema.{name} (key: {ema_key_in_ckpt}) not found.")

#                 # Load optimizer states
#                 print("Loading optimizer state dictionaries...")
#                 for name, optim in optims.items():
#                     if name in checkpoint.get('optims', {}):
#                         try:
#                            optim.load_state_dict(checkpoint['optims'][name])
#                            print(f"  Loaded state for optims.{name}")
#                         except ValueError as ve:
#                            print(f"  Warning: Could not load state for optims.{name}. Size mismatch? {ve}")
#                         except Exception as oe:
#                            print(f"  Warning: Error loading state for optims.{name}. {oe}")
#                     else:
#                          print(f"  Warning: State for optims.{name} not found in checkpoint.")

#                 # Load the starting iteration
#                 start_iter = checkpoint.get('iter', train_args.resume_iter)
#                 print(f"Successfully resumed from iteration {start_iter}.")

#             except Exception as e:
#                 print(f"Error loading checkpoint file {checkpoint_path}: {e}")
#                 print("Starting training from scratch.")
#                 start_iter = 0
#                 # Initialize weights if loading failed
#                 print("Initializing model weights...")
#                 for name, network in nets.items():
#                     network.apply(he_init)
#                 print("Weights initialized.")
#         else:
#             print(f"Warning: Checkpoint file not found at {checkpoint_path}. Starting training from scratch.")
#             start_iter = 0
#              # Initialize weights if starting from scratch
#             print("Initializing model weights...")
#             for name, network in nets.items():
#                 network.apply(he_init)
#             print("Weights initialized.")

#     else: # If not resuming
#         print("Initializing model weights...")
#         for name, network in nets.items():
#              network.apply(he_init)
#         print("Weights initialized.")


#     # --- 6. Prepare DataLoaders ---
#     # Assuming training on one specific fold defined by current_fold
#     current_fold = data_args.get('train_fold_num', 0) # Default to fold 0 if not specified
#     train_csv_path = Path(data_args.fold_csv_dir) / f"train_fold_{current_fold}.csv"
#     img_root = Path(data_args.img_root_dir)

#     print(f"Preparing dataloaders for fold {current_fold}...")
#     try:
#         loader_src, loader_ref = get_stargan_train_loaders(
#             csv_path=train_csv_path,
#             img_root=img_root,
#             img_size=args.img_size,
#             batch_size=train_args.batch_size,
#             num_workers=train_args.num_workers,
#             label_col=data_args.label_column
#         )
#         if loader_src is None or loader_ref is None:
#             print(f"Error: Could not create DataLoaders for {train_csv_path}. Check dataset setup.")
#             return
#     except FileNotFoundError:
#          print(f"Error: Training CSV file not found: {train_csv_path}")
#          return
#     except Exception as e:
#         print(f"Error creating dataloaders: {e}")
#         return

#     # Use infinite iterators to simplify the loop
#     iter_src = iter(loader_src)
#     iter_ref = iter(loader_ref)

#     # --- 7. Training Loop ---
#     print(f'Start training from iteration {start_iter}...')
#     start_time = time.time()
#     initial_lambda_ds = args.lambda_ds # Store initial value for decay calculation
#     current_lambda_ds = args.lambda_ds # Track current decayed value

#     # Use tqdm for progress bar over iterations
#     pbar = tqdm(range(start_iter, train_args.total_iters), initial=start_iter, total=train_args.total_iters, desc="Training StarGAN v2")

#     for i in pbar:
#         # --- Fetch data ---
#         try:
#             x_real, y_org = next(iter_src)
#         except (StopIteration, AttributeError): # Handle StopIteration and potential AttributeError if iter isn't initialized
#             iter_src = iter(loader_src)
#             x_real, y_org = next(iter_src)

#         try:
#             x_ref, x_ref2, y_trg = next(iter_ref)
#         except (StopIteration, AttributeError):
#             iter_ref = iter(loader_ref)
#             x_ref, x_ref2, y_trg = next(iter_ref)

#         # Move data to device
#         x_real, y_org = x_real.to(device), y_org.to(device)
#         x_ref, x_ref2, y_trg = x_ref.to(device), x_ref2.to(device), y_trg.to(device)

#         # Generate random latent codes
#         z_trg = torch.randn(x_real.size(0), args.latent_dim).to(device) # Ensure batch size matches x_real
#         z_trg2 = torch.randn(x_real.size(0), args.latent_dim).to(device)

#         # --- Train Discriminator ---
#         optims.discriminator.zero_grad()
#         # Latent-guided
#         d_loss_latent, d_losses_latent = compute_d_loss(
#             nets, args, x_real, y_org, y_trg, z_trg=z_trg)
#         # Reference-guided
#         d_loss_ref, d_losses_ref = compute_d_loss(
#             nets, args, x_real, y_org, y_trg, x_ref=x_ref)
#         # Combine and backward
#         d_loss = d_loss_latent + d_loss_ref
#         d_loss.backward()
#         optims.discriminator.step()

#         # --- Train Generator, MappingNetwork, StyleEncoder ---
#         optims.generator.zero_grad()
#         optims.mapping_network.zero_grad()
#         optims.style_encoder.zero_grad()
#         # Latent-guided
#         g_loss_latent, g_losses_latent = compute_g_loss(
#             nets, args, x_real, y_org, y_trg, z_trgs=[z_trg, z_trg2])
#         # Reference-guided
#         g_loss_ref, g_losses_ref = compute_g_loss(
#             nets, args, x_real, y_org, y_trg, x_refs=[x_ref, x_ref2])
#         # Combine and backward
#         g_loss = g_loss_latent + g_loss_ref
#         g_loss.backward()
#         optims.generator.step()
#         optims.mapping_network.step()
#         optims.style_encoder.step()

#         # --- Update EMA networks ---
#         moving_average(nets.generator, nets_ema.generator, beta=0.999)
#         moving_average(nets.mapping_network, nets_ema.mapping_network, beta=0.999)
#         moving_average(nets.style_encoder, nets_ema.style_encoder, beta=0.999)

#         # --- Decay lambda_ds ---
#         if current_lambda_ds > 0 and train_args.ds_iter > 0 : # Add check for ds_iter > 0
#              current_lambda_ds -= (initial_lambda_ds / train_args.ds_iter)
#              args.lambda_ds = max(0, current_lambda_ds) # Update config value used by compute_g_loss

#         # --- Logging ---
#         if (i + 1) % train_args.print_every == 0:
#             elapsed = time.time() - start_time
#             elapsed_str = str(datetime.timedelta(seconds=elapsed))[:-7]
#             log_dict = {
#                 "Iter": f"{i+1}/{train_args.total_iters}",
#                 "D/lat_real": d_losses_latent.real, "D/lat_fake": d_losses_latent.fake, "D/lat_reg": d_losses_latent.reg,
#                 "D/ref_real": d_losses_ref.real, "D/ref_fake": d_losses_ref.fake, "D/ref_reg": d_losses_ref.reg,
#                 "G/lat_adv": g_losses_latent.adv, "G/lat_sty": g_losses_latent.sty, "G/lat_ds": g_losses_latent.ds, "G/lat_cyc": g_losses_latent.cyc,
#                 "G/ref_adv": g_losses_ref.adv, "G/ref_sty": g_losses_ref.sty, "G/ref_ds": g_losses_ref.ds, "G/ref_cyc": g_losses_ref.cyc,
#                 "lambda_ds": args.lambda_ds
#             }
#             pbar.set_postfix(log_dict) # Update tqdm progress bar postfix
#             # Optional: More detailed console print
#             # print(f"Elapsed: [{elapsed_str}] " + " ".join([f"{k}: [{v:.4f}]" for k,v in log_dict.items()]))


#         # --- Sample Images ---
#         if (i + 1) % train_args.sample_every == 0:
#             # Use nets_ema for generation
#             nets_ema.generator.eval()
#             nets_ema.mapping_network.eval()
#             nets_ema.style_encoder.eval()
#             with torch.no_grad():
#                  N_sample = min(x_real.size(0), 4) # Use first few images from current batch
#                  x_src_sample = x_real[:N_sample]
#                  # Use a fixed latent code for consistent sampling across iterations (optional)
#                  # fixed_z = torch.randn(1, args.latent_dim).to(device)
#                  # z_trg_sample = fixed_z.repeat(N_sample, 1)
#                  # Or use random code each time:
#                  z_trg_sample = torch.randn(N_sample, args.latent_dim).to(device)

#                  x_concat_sample = [denormalize(x_src_sample)]
#                  # Generate for all target domains
#                  for target_label in range(args.num_domains):
#                      y_trg_sample = torch.tensor([target_label] * N_sample).to(device)
#                      s_trg_sample = nets_ema.mapping_network(z_trg_sample, y_trg_sample)
#                      x_fake_sample = nets_ema.generator(x_src_sample, s_trg_sample)
#                      x_concat_sample.append(denormalize(x_fake_sample))

#                  save_image(torch.cat(x_concat_sample, dim=0), N_sample, sample_dir / f"{i+1:06d}_sample_latent.jpg")
#             print(f"\nSample images saved to {sample_dir} for iteration {i+1}")
#             # Set models back to train mode if needed (optimizers handle this usually)
#             # nets.generator.train(); nets.mapping_network.train(); ...


#         # --- Save Checkpoint ---
#         if (i + 1) % train_args.save_every == 0:
#              print(f"\nSaving checkpoint for iteration {i+1}...")
#              state = {
#                  'iter': i + 1,
#                  'nets': {name: net.state_dict() for name, net in nets.items()},
#                  'nets_ema': {name: net_ema.state_dict() for name, net_ema in nets_ema.items()},
#                  'optims': {name: optim.state_dict() for name, optim in optims.items()},
#                  'current_lambda_ds': args.lambda_ds # Save current decayed lambda
#              }
#              filename = checkpoint_dir / f"{i+1:06d}.ckpt"
#              try:
#                  save_checkpoint(state, filename) # Use helper function
#              except Exception as e:
#                  print(f"Error saving checkpoint to {filename}: {e}")


#     print('Training finished.')
#     pbar.close()


# # --- Entry Point ---
# if __name__ == "__main__":
#     # Determine the config file path relative to this script's location
#     script_dir = Path(__file__).parent
#     project_root_dir = script_dir.parent # Assumes training/ is one level below project root
#     default_config_path = project_root_dir / "configs" / "stargan_v2_config.yaml"

#     # --- !!! You MUST create stargan_v2_config.yaml based on the example below !!! ---
#     # Example YAML structure (save as configs/stargan_v2_config.yaml):
#     # model_params:
#     #   img_size: 256
#     #   num_domains: 8           # Based on ISIC 2019 excluding UNK
#     #   latent_dim: 16
#     #   hidden_dim: 512
#     #   style_dim: 64
#     #   w_hpf: 0                 # Disable high-pass filter for skin lesions
#     #   max_conv_dim: 512
#     #   lambda_reg: 1            # R1 regularization weight
#     #   lambda_cyc: 1            # Cycle-consistency loss weight
#     #   lambda_sty: 1            # Style reconstruction loss weight
#     #   lambda_ds: 1             # Diversity sensitive loss weight (start value)
#     # train_params:
#     #   total_iters: 100000      # Total training iterations
#     #   resume_iter: 0           # Iteration to resume from (0 for new training)
#     #   batch_size: 4            # Adjust based on GPU memory (16GB might handle 4 or 8)
#     #   val_batch_size: 8        # Batch size for validation (not used in this script yet)
#     #   lr: 0.0001               # Learning rate for G, E, D
#     #   f_lr: 0.000001           # Learning rate for Mapping Network (F)
#     #   beta1: 0.0
#     #   beta2: 0.99
#     #   weight_decay: 0.0001
#     #   ds_iter: 100000          # Iterations over which to decay lambda_ds to 0
#     #   seed: 42
#     #   num_workers: 4           # Number of CPU workers for data loading
#     #   print_every: 100         # Log frequency (iterations)
#     #   sample_every: 1000       # Sample image generation frequency (iterations)
#     #   save_every: 10000        # Checkpoint saving frequency (iterations)
#     #   checkpoint_dir: './expr/stargan_v2_isic/checkpoints' # Where to save checkpoints
#     #   sample_dir: './expr/stargan_v2_isic/samples'         # Where to save sample images
#     # data_params:
#     #   train_fold_num: 0        # Which fold CSV to use for training (e.g., train_fold_0.csv)
#     #   fold_csv_dir: './pytorch_project/data/ISIC2019/folds_pytorch' # Path to directory containing fold CSVs
#     #   img_root_dir: './pytorch_project/data/ISIC2019/inpainted_pytorch/train' # Path to directory containing processed images
#     #   label_column: 'integer_label' # Name of the integer label column in the CSVs

#     config_file = default_config_path # Use the calculated default path
#     if not config_file.exists():
#         print(f"Error: Config file not found at {config_file}")
#         print("Please create a YAML config file (see example structure in script comments).")
#     else:
#         train_stargan_v2(config_path=config_file)





# pytorch_project/training/train_stargan.py

import os
import time
import datetime
from pathlib import Path
import yaml # For loading config
from munch import Munch # Similar to argparse.Namespace, for easy config access
import random
import numpy as np
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm # Progress bar
import torchvision # Needed for make_grid

# --- Import project modules ---
# Using absolute imports assuming 'pytorch_project' is the root or accessible
try:
    # Ensure correct relative paths if running from training dir
    from dataset.stargan_v2 import get_stargan_train_loaders
    from model.stargan_v2 import Generator, MappingNetwork, StyleEncoder, Discriminator
    from utils.stargan_utils import (
        he_init, print_network, save_image, denormalize,
        moving_average, adv_loss, r1_reg
    )
    # Assuming helpers.py is in utils directory
    from utils.helpers import load_config, save_checkpoint, load_checkpoint
    # Import TensorBoard writer
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    # Fallback for running directly from training dir or if structure differs
    print("Import failed using absolute path, trying relative...")
    import sys
    project_root = str(Path(__file__).resolve().parent.parent)
    if project_root not in sys.path:
        sys.path.append(project_root)
        print(f"Added {project_root} to sys.path")

    from dataset.stargan_v2 import get_stargan_train_loaders
    from model.stargan_v2 import Generator, MappingNetwork, StyleEncoder, Discriminator
    from utils.stargan_utils import (
        he_init, print_network, save_image, denormalize,
        moving_average, adv_loss, r1_reg
    )
    from utils.helpers import load_config, save_checkpoint, load_checkpoint
    # Import TensorBoard writer
    from torch.utils.tensorboard import SummaryWriter


# --- Loss Computation Helper Functions ---

def compute_g_loss(nets, args, x_real, y_org, y_trg, z_trgs=None, x_refs=None):
    """Computes generator loss components."""
    assert (z_trgs is None) != (x_refs is None)
    if z_trgs is not None:
        z_trg, z_trg2 = z_trgs
    if x_refs is not None:
        x_ref, x_ref2 = x_refs

    # Adversarial loss
    if z_trgs is not None:
        s_trg = nets.mapping_network(z_trg, y_trg)
    else: # x_refs is not None
        s_trg = nets.style_encoder(x_ref, y_trg)

    x_fake = nets.generator(x_real, s_trg)
    out = nets.discriminator(x_fake, y_trg)
    loss_adv = adv_loss(out, 1)

    # Style reconstruction loss
    s_pred = nets.style_encoder(x_fake, y_trg)
    loss_sty = torch.mean(torch.abs(s_pred - s_trg))

    # Diversity sensitive loss
    if z_trgs is not None:
        s_trg2 = nets.mapping_network(z_trg2, y_trg)
    else: # x_refs is not None
        s_trg2 = nets.style_encoder(x_ref2, y_trg)
    x_fake2 = nets.generator(x_real, s_trg2)
    x_fake2 = x_fake2.detach()
    loss_ds = torch.mean(torch.abs(x_fake - x_fake2))

    # Cycle-consistency/Reconstruction loss
    s_org = nets.style_encoder(x_real, y_org)
    x_rec = nets.generator(x_fake, s_org)
    loss_cyc = torch.mean(torch.abs(x_rec - x_real))

    # Total generator loss
    loss = (loss_adv +
            args.lambda_sty * loss_sty -
            args.lambda_ds * loss_ds + # Note the minus sign for diversity
            args.lambda_cyc * loss_cyc)

    return loss, Munch(adv=loss_adv.item(),
                       sty=loss_sty.item(),
                       ds=loss_ds.item(),
                       cyc=loss_cyc.item())


def compute_d_loss(nets, args, x_real, y_org, y_trg, z_trg=None, x_ref=None):
    """Computes discriminator loss components."""
    assert (z_trg is None) != (x_ref is None)
    # Loss with real images
    x_real.requires_grad_()
    out = nets.discriminator(x_real, y_org)
    loss_real = adv_loss(out, 1)
    loss_reg = r1_reg(out, x_real)

    # Loss with fake images
    with torch.no_grad():
        if z_trg is not None:
            s_trg = nets.mapping_network(z_trg, y_trg)
        else:  # x_ref is not None
            s_trg = nets.style_encoder(x_ref, y_trg)
        x_fake = nets.generator(x_real, s_trg)

    out = nets.discriminator(x_fake, y_trg)
    loss_fake = adv_loss(out, 0)

    # Total discriminator loss
    loss = loss_real + loss_fake + args.lambda_reg * loss_reg
    return loss, Munch(real=loss_real.item(),
                       fake=loss_fake.item(),
                       reg=loss_reg.item())


# --- Main Training Function ---
def train_stargan_v2(config_path="configs/stargan_v2_config.yaml"):
    """Main function to train the StarGAN v2 model."""

    # --- 1. Load Configuration ---
    try:
        config = load_config(config_path)
        args = Munch(config['model_params'])
        train_args = Munch(config['train_params'])
        data_args = Munch(config['data_params'])
        print("Configuration loaded successfully.")
    except Exception as e:
        print(f"Error loading or parsing config file {config_path}: {e}")
        return

    # --- 2. Setup Environment ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.manual_seed(train_args.seed)
    random.seed(train_args.seed)
    np.random.seed(train_args.seed)

    # --- 3. Setup TensorBoard ---
    print("Setting up TensorBoard...")
    log_dir = Path(train_args.get('log_dir', './expr/stargan_v2_isic/logs'))
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"TensorBoard logs will be saved to: {writer.log_dir}") # writer.log_dir has the timestamped path

    # --- 4. Build Models and EMA Models ---
    print("Building models...")
    nets = Munch()
    try:
        nets.generator = Generator(args.img_size, args.style_dim, w_hpf=args.w_hpf, max_conv_dim=args.max_conv_dim)
        nets.mapping_network = MappingNetwork(args.latent_dim, args.style_dim, args.num_domains)
        nets.style_encoder = StyleEncoder(args.img_size, args.style_dim, args.num_domains, max_conv_dim=args.max_conv_dim)
        nets.discriminator = Discriminator(args.img_size, args.num_domains, max_conv_dim=args.max_conv_dim)
    except Exception as e:
        print(f"Error building models: {e}")
        writer.close() # Close writer if error occurs
        return

    nets_ema = Munch()
    nets_ema.generator = copy.deepcopy(nets.generator)
    nets_ema.mapping_network = copy.deepcopy(nets.mapping_network)
    nets_ema.style_encoder = copy.deepcopy(nets.style_encoder)

    # Move models to device and print structure
    for name, net in nets.items():
        print_network(net, name)
        net.to(device)
    for name, net_ema in nets_ema.items():
        net_ema.to(device)
        net_ema.eval()

    # --- 5. Setup Optimizers ---
    print("Setting up optimizers...")
    optims = Munch()
    try:
        optims.generator = torch.optim.Adam(
            nets.generator.parameters(),
            lr=train_args.lr, betas=(train_args.beta1, train_args.beta2),
            weight_decay=train_args.weight_decay
        )
        optims.mapping_network = torch.optim.Adam(
            nets.mapping_network.parameters(),
            lr=train_args.f_lr, betas=(train_args.beta1, train_args.beta2),
            weight_decay=train_args.weight_decay
        )
        optims.style_encoder = torch.optim.Adam(
            nets.style_encoder.parameters(),
            lr=train_args.lr, betas=(train_args.beta1, train_args.beta2),
            weight_decay=train_args.weight_decay
        )
        optims.discriminator = torch.optim.Adam(
            nets.discriminator.parameters(),
            lr=train_args.lr, betas=(train_args.beta1, train_args.beta2),
            weight_decay=train_args.weight_decay
        )
    except AttributeError as e:
        print(f"Error setting up optimizers. Missing parameter in config? {e}")
        writer.close()
        return

    # --- 6. Prepare Checkpoint Directories and Load (if resuming) ---
    checkpoint_dir = Path(train_args.checkpoint_dir)
    sample_dir = Path(train_args.sample_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    start_iter = 0
    if train_args.resume_iter > 0:
        print(f"Attempting to resume training from iteration {train_args.resume_iter}...")
        checkpoint_path = checkpoint_dir / f"{train_args.resume_iter:06d}.ckpt"

        if checkpoint_path.exists():
            try:
                checkpoint = load_checkpoint(checkpoint_path, device) # Assumes load_checkpoint in helpers.py

                print("Loading model state dictionaries...")
                for name, net in nets.items():
                    if name in checkpoint.get('nets', {}):
                        net.load_state_dict(checkpoint['nets'][name])
                        print(f"  Loaded state for nets.{name}")
                    else: print(f"  Warning: State for nets.{name} not found.")

                print("Loading EMA model state dictionaries...")
                for name, net_ema in nets_ema.items():
                    ema_key_in_ckpt = name
                    if ema_key_in_ckpt in checkpoint.get('nets_ema', {}):
                        net_ema.load_state_dict(checkpoint['nets_ema'][ema_key_in_ckpt])
                        print(f"  Loaded state for nets_ema.{name}")
                    else: print(f"  Warning: State for nets_ema.{name} (key: {ema_key_in_ckpt}) not found.")

                print("Loading optimizer state dictionaries...")
                for name, optim in optims.items():
                    if name in checkpoint.get('optims', {}):
                        try:
                           optim.load_state_dict(checkpoint['optims'][name])
                           print(f"  Loaded state for optims.{name}")
                        except ValueError as ve: print(f"  Warning: Could not load optims.{name} state. Size mismatch? {ve}")
                        except Exception as oe: print(f"  Warning: Error loading optims.{name} state. {oe}")
                    else: print(f"  Warning: State for optims.{name} not found.")

                start_iter = checkpoint.get('iter', train_args.resume_iter)
                # Load lambda_ds if saved, otherwise recalculate based on start_iter
                initial_lambda_ds = args.lambda_ds # Store original lambda from config
                current_lambda_ds = checkpoint.get('current_lambda_ds', initial_lambda_ds * max(0, 1 - start_iter / train_args.ds_iter))
                args.lambda_ds = current_lambda_ds # Update args for loss calculation
                print(f"Successfully resumed from iteration {start_iter} with lambda_ds={current_lambda_ds:.4f}.")

            except Exception as e:
                print(f"Error loading checkpoint file {checkpoint_path}: {e}")
                print("Starting training from scratch.")
                start_iter = 0
                initial_lambda_ds = args.lambda_ds
                current_lambda_ds = args.lambda_ds
                print("Initializing model weights...")
                for name, network in nets.items(): network.apply(he_init)
                print("Weights initialized.")
        else:
            print(f"Warning: Checkpoint file not found at {checkpoint_path}. Starting training from scratch.")
            start_iter = 0
            initial_lambda_ds = args.lambda_ds
            current_lambda_ds = args.lambda_ds
            print("Initializing model weights...")
            for name, network in nets.items(): network.apply(he_init)
            print("Weights initialized.")
    else: # If not resuming
        initial_lambda_ds = args.lambda_ds
        current_lambda_ds = args.lambda_ds
        print("Initializing model weights...")
        for name, network in nets.items(): network.apply(he_init)
        print("Weights initialized.")


    # --- 7. Prepare DataLoaders ---
    current_fold = data_args.get('train_fold_num', 0)
    train_csv_path = Path(data_args.fold_csv_dir) / f"train_fold_{current_fold}.csv"
    img_root = Path(data_args.img_root_dir)

    print(f"Preparing dataloaders for fold {current_fold}...")
    try:
        loader_src, loader_ref = get_stargan_train_loaders(
            csv_path=train_csv_path, img_root=img_root, img_size=args.img_size,
            batch_size=train_args.batch_size, num_workers=train_args.num_workers,
            label_col=data_args.label_column
        )
        if loader_src is None or loader_ref is None:
            print(f"Error: Could not create DataLoaders for {train_csv_path}.")
            writer.close(); return
    except FileNotFoundError:
         print(f"Error: Training CSV file not found: {train_csv_path}")
         writer.close(); return
    except Exception as e:
        print(f"Error creating dataloaders: {e}")
        writer.close(); return

    iter_src = iter(loader_src)
    iter_ref = iter(loader_ref)

    # --- 8. Training Loop ---
    print(f'Start training from iteration {start_iter}...')
    start_time = time.time()
    pbar = tqdm(range(start_iter, train_args.total_iters), initial=start_iter, total=train_args.total_iters, desc="Training StarGAN v2")

    for i in pbar:
        # --- Fetch data ---
        try: x_real, y_org = next(iter_src)
        except (StopIteration, AttributeError): iter_src = iter(loader_src); x_real, y_org = next(iter_src)
        try: x_ref, x_ref2, y_trg = next(iter_ref)
        except (StopIteration, AttributeError): iter_ref = iter(loader_ref); x_ref, x_ref2, y_trg = next(iter_ref)

        x_real, y_org = x_real.to(device), y_org.to(device)
        x_ref, x_ref2, y_trg = x_ref.to(device), x_ref2.to(device), y_trg.to(device)
        z_trg = torch.randn(x_real.size(0), args.latent_dim).to(device)
        z_trg2 = torch.randn(x_real.size(0), args.latent_dim).to(device)

        # --- Train Discriminator ---
        optims.discriminator.zero_grad()
        d_loss_latent, d_losses_latent = compute_d_loss(nets, args, x_real, y_org, y_trg, z_trg=z_trg)
        d_loss_ref, d_losses_ref = compute_d_loss(nets, args, x_real, y_org, y_trg, x_ref=x_ref)
        d_loss = d_loss_latent + d_loss_ref
        d_loss.backward()
        optims.discriminator.step()

        # --- Train Generator, MappingNetwork, StyleEncoder ---
        optims.generator.zero_grad()
        optims.mapping_network.zero_grad()
        optims.style_encoder.zero_grad()
        g_loss_latent, g_losses_latent = compute_g_loss(nets, args, x_real, y_org, y_trg, z_trgs=[z_trg, z_trg2])
        g_loss_ref, g_losses_ref = compute_g_loss(nets, args, x_real, y_org, y_trg, x_refs=[x_ref, x_ref2])
        g_loss = g_loss_latent + g_loss_ref
        g_loss.backward()
        optims.generator.step()
        optims.mapping_network.step()
        optims.style_encoder.step()

        # --- Update EMA networks ---
        moving_average(nets.generator, nets_ema.generator, beta=0.999)
        moving_average(nets.mapping_network, nets_ema.mapping_network, beta=0.999)
        moving_average(nets.style_encoder, nets_ema.style_encoder, beta=0.999)

        # --- Decay lambda_ds ---
        if current_lambda_ds > 0 and train_args.ds_iter > 0 :
             current_lambda_ds -= (initial_lambda_ds / train_args.ds_iter)
             args.lambda_ds = max(0, current_lambda_ds)

        # --- Logging ---
        if (i + 1) % train_args.print_every == 0:
            elapsed = time.time() - start_time
            elapsed_str = str(datetime.timedelta(seconds=elapsed))[:-7]
            log_dict = {
                "Iter": f"{i+1}/{train_args.total_iters}",
                "D/lat_real": d_losses_latent.real, "D/lat_fake": d_losses_latent.fake, "D/lat_reg": d_losses_latent.reg,
                "D/ref_real": d_losses_ref.real, "D/ref_fake": d_losses_ref.fake, "D/ref_reg": d_losses_ref.reg,
                "G/lat_adv": g_losses_latent.adv, "G/lat_sty": g_losses_latent.sty, "G/lat_ds": g_losses_latent.ds, "G/lat_cyc": g_losses_latent.cyc,
                "G/ref_adv": g_losses_ref.adv, "G/ref_sty": g_losses_ref.sty, #"G/ref_ds": g_losses_ref.ds, "G/ref_cyc": g_losses_ref.cyc, # ref loss only has adv, sty
                "lambda_ds": args.lambda_ds
            }
            pbar.set_postfix(log_dict)

            # Write losses to TensorBoard
            writer.add_scalar('Loss/D/latent_real', d_losses_latent.real, i + 1)
            writer.add_scalar('Loss/D/latent_fake', d_losses_latent.fake, i + 1)
            writer.add_scalar('Loss/D/latent_reg', d_losses_latent.reg, i + 1)
            writer.add_scalar('Loss/D/ref_real', d_losses_ref.real, i + 1)
            writer.add_scalar('Loss/D/ref_fake', d_losses_ref.fake, i + 1)
            writer.add_scalar('Loss/D/ref_reg', d_losses_ref.reg, i + 1)
            writer.add_scalar('Loss/D/total', d_loss.item(), i + 1)
            writer.add_scalar('Loss/G/latent_adv', g_losses_latent.adv, i + 1)
            writer.add_scalar('Loss/G/latent_sty', g_losses_latent.sty, i + 1)
            writer.add_scalar('Loss/G/latent_ds', g_losses_latent.ds, i + 1)
            writer.add_scalar('Loss/G/latent_cyc', g_losses_latent.cyc, i + 1)
            writer.add_scalar('Loss/G/ref_adv', g_losses_ref.adv, i + 1)
            writer.add_scalar('Loss/G/ref_sty', g_losses_ref.sty, i + 1)
            # Note: Original compute_g_loss for ref doesn't return ds/cyc items directly in Munch
            writer.add_scalar('Loss/G/total', g_loss.item(), i + 1)
            writer.add_scalar('Params/lambda_ds', args.lambda_ds, i + 1)
            # Optionally add learning rates
            # writer.add_scalar('Params/LR_G', optims.generator.param_groups[0]['lr'], i + 1)


        # --- Sample Images ---
        if (i + 1) % train_args.sample_every == 0:
            nets_ema.generator.eval()
            nets_ema.mapping_network.eval()
            nets_ema.style_encoder.eval()
            with torch.no_grad():
                N_sample = min(x_real.size(0), 4)
                x_src_sample = x_real[:N_sample]
                z_trg_sample = torch.randn(N_sample, args.latent_dim).to(device)
                sample_filename = sample_dir / f"{i+1:06d}_sample_latent.jpg"

                x_concat_sample = [denormalize(x_src_sample)]
                for target_label in range(args.num_domains):
                    y_trg_sample = torch.tensor([target_label] * N_sample).to(device)
                    s_trg_sample = nets_ema.mapping_network(z_trg_sample, y_trg_sample)
                    x_fake_sample = nets_ema.generator(x_src_sample, s_trg_sample)
                    x_concat_sample.append(denormalize(x_fake_sample))

                x_concat_grid_tensor = torch.cat(x_concat_sample, dim=0)
                save_image(x_concat_grid_tensor, N_sample, sample_filename) # Save grid to file

                # Add grid to TensorBoard
                grid = torchvision.utils.make_grid(x_concat_grid_tensor.cpu(), nrow=N_sample, padding=0)
                writer.add_image(f'Samples/latent_guided', grid, global_step=i+1)

            print(f"\nSample images saved to {sample_filename} and TensorBoard for iteration {i+1}")
            # Set models back to train mode (though often handled by optimizer steps)
            # for net in nets.values(): net.train()


        # --- Save Checkpoint ---
        if (i + 1) % train_args.save_every == 0:
             print(f"\nSaving checkpoint for iteration {i+1}...")
             state = {
                 'iter': i + 1,
                 'nets': {name: net.state_dict() for name, net in nets.items()},
                 'nets_ema': {name: net_ema.state_dict() for name, net_ema in nets_ema.items()},
                 'optims': {name: optim.state_dict() for name, optim in optims.items()},
                 'current_lambda_ds': args.lambda_ds
             }
             filename = checkpoint_dir / f"{i+1:06d}.ckpt"
             try:
                 save_checkpoint(state, filename) # Use helper function
             except Exception as e:
                 print(f"Error saving checkpoint to {filename}: {e}")


    print('Training finished.')
    pbar.close()
    writer.close() # Close TensorBoard writer at the end


# --- Entry Point ---
if __name__ == "__main__":
    # Determine the config file path relative to this script's location
    script_dir = Path(__file__).parent
    project_root_dir = script_dir.parent # Assumes training/ is one level below project root
    default_config_path = project_root_dir / "configs" / "stargan_v2_config.yaml"

    # --- !!! You MUST create stargan_v2_config.yaml based on the example below !!! ---
    # Example YAML structure (save as configs/stargan_v2_config.yaml):
    # model_params:
    #   img_size: 256
    #   num_domains: 8           # Based on ISIC 2019 excluding UNK
    #   latent_dim: 16
    #   hidden_dim: 512
    #   style_dim: 64
    #   w_hpf: 0                 # Disable high-pass filter for skin lesions
    #   max_conv_dim: 512
    #   lambda_reg: 1            # R1 regularization weight
    #   lambda_cyc: 1            # Cycle-consistency loss weight
    #   lambda_sty: 1            # Style reconstruction loss weight
    #   lambda_ds: 1             # Diversity sensitive loss weight (start value)
    # train_params:
    #   total_iters: 100000      # Total training iterations
    #   resume_iter: 0           # Iteration to resume from (0 for new training)
    #   batch_size: 4            # Adjust based on GPU memory (16GB might handle 4 or 8)
    #   val_batch_size: 8        # Batch size for validation (not used in this script yet)
    #   lr: 0.0001               # Learning rate for G, E, D
    #   f_lr: 0.000001           # Learning rate for Mapping Network (F)
    #   beta1: 0.0
    #   beta2: 0.99
    #   weight_decay: 0.0001
    #   ds_iter: 100000          # Iterations over which to decay lambda_ds to 0
    #   seed: 42
    #   num_workers: 4           # Number of CPU workers for data loading
    #   print_every: 100         # Log frequency (iterations)
    #   sample_every: 1000       # Sample image generation frequency (iterations)
    #   save_every: 10000        # Checkpoint saving frequency (iterations)
    #   checkpoint_dir: './expr/stargan_v2_isic/checkpoints' # Where to save checkpoints
    #   sample_dir: './expr/stargan_v2_isic/samples'         # Where to save sample images
    #   log_dir: './expr/stargan_v2_isic/logs'               # Where to save TensorBoard logs
    # data_params:
    #   train_fold_num: 0        # Which fold CSV to use for training (e.g., train_fold_0.csv)
    #   fold_csv_dir: './pytorch_project/data/ISIC2019/folds_pytorch' # Path to directory containing fold CSVs
    #   img_root_dir: './pytorch_project/data/ISIC2019/inpainted_pytorch/train' # Path to directory containing processed images
    #   label_column: 'integer_label' # Name of the integer label column in the CSVs

    config_file = default_config_path # Use the calculated default path
    if not config_file.exists():
        print(f"Error: Config file not found at {config_file}")
        print("Please create a YAML config file (see example structure in script comments).")
    else:
        train_stargan_v2(config_path=config_file)