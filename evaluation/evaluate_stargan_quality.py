# pytorch_project/evaluation/evaluate_stargan_quality.py
# python evaluation/evaluate_stargan_quality.py ^
#     --config_path ./configs/stargan_v2_config.yaml ^
#     --resume_iter 80000 ^
#     --real_img_dir ./data/ISIC2019/real_images_for_eval ^
#     --eval_output_dir ./training/expr/stargan_v2_isic/evaluation_results_80k ^
#     --source_domain_fid NV ^
#     --num_fakes_for_fid 1000 ^
#     --inception_dims 2048 ^
#     --batch_size 16 ^
#     --fid_batch_size 32 ^
#     --lpips_batch_size 16 ^
#     --num_samples_per_input 5 ^
#     --calculate_lpips ^
#     --cleanup_fake ^
#     --num_workers 0



# import os
# import shutil
# import random
# from pathlib import Path
# import yaml
# from munch import Munch
# import json
# from collections import OrderedDict, defaultdict
# import argparse # 使用 argparse 来传递参数

# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# import pandas as pd
# import numpy as np
# from PIL import Image
# import cv2
# from tqdm import tqdm
# import torchvision.transforms as T
# import torchvision
# import sys # Import sys for path manipulation

# # --- 添加项目根目录到 sys.path ---
# PROJECT_ROOT = Path(__file__).resolve().parent.parent
# if str(PROJECT_ROOT) not in sys.path:
#     sys.path.append(str(PROJECT_ROOT))
#     print(f"Added {PROJECT_ROOT} to sys.path")

# # --- 导入必要的模块 ---
# try:
#     from model.stargan_v2 import Generator, MappingNetwork, StyleEncoder
#     from utils.helpers import load_config, load_checkpoint
#     from utils.stargan_utils import denormalize # save_image not needed if using torchvision directly
#     from metrics.fid import calculate_fid_given_paths
#     from metrics.lpips import calculate_lpips_given_images
# except ImportError as e:
#     print(f"Error importing project modules: {e}")
#     sys.exit(1)
# except FileNotFoundError as e_fnf:
#      print(f"Error: A required file might be missing: {e_fnf}")
#      sys.exit(1)


# # --- 简单的评估数据集 ---
# class EvalDataset(Dataset):
#     """Simple Dataset for loading images for evaluation."""
#     def __init__(self, root, img_size=299, transform=None, file_limit=None): # Default img_size to 299 for FID/LPIPS
#         print(f"Scanning directory: {root}")
#         root_path = Path(root)
#         if not root_path.is_dir():
#              raise FileNotFoundError(f"Directory not found: {root}")

#         self.files = list(root_path.rglob('*.[pP][nN][gG]')) + \
#                      list(root_path.rglob('*.[jJ][pP][gG]')) + \
#                      list(root_path.rglob('*.[jJ][pP][eE][gG]'))
#         if not self.files:
#             raise FileNotFoundError(f"No image files found in {root}")

#         self.files.sort()
#         if file_limit is not None and file_limit > 0:
#              if len(self.files) > file_limit:
#                   print(f"Limiting dataset from {len(self.files)} to {file_limit} files.")
#                   self.files = self.files[:file_limit]
#              else:
#                   print(f"Found {len(self.files)} files, less than limit {file_limit}. Using all.")

#         self.transform = transform
#         if self.transform is None:
#             # Default transform for InceptionV3/LPIPS
#             self.transform = T.Compose([
#                 T.Resize((img_size, img_size)), # Use passed img_size
#                 T.ToTensor(), # Range [0, 1]
#                 T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet norm
#             ])
#         self._length = len(self.files)
#         print(f"Initialized EvalDataset with {self._length} images from {root}")

#     def __len__(self):
#         return self._length

#     def __getitem__(self, i):
#         path = self.files[i]
#         try:
#             img = Image.open(path).convert('RGB')
#             if self.transform is not None:
#                 img = self.transform(img)
#             return img
#         except Exception as e:
#             print(f"Warning: Error loading image {path} for evaluation: {e}")
#             # Determine output shape based on transform
#             c, h, w = 3, 299, 299 # Default
#             if isinstance(self.transform, T.Compose):
#                  for t in reversed(self.transform.transforms):
#                       if isinstance(t, T.Resize):
#                            size = t.size
#                            if isinstance(size, int): h=w=size
#                            elif isinstance(size, (list, tuple)) and len(size) == 2: h, w = size
#                            break
#             return torch.zeros((c, h, w))


# # --- 主评估函数 ---
# def evaluate_stargan(eval_args): # Renamed from main to avoid potential conflicts
#     """Performs StarGAN v2 evaluation (FID and LPIPS Diversity)."""

#     # --- 1. 加载主配置文件 ---
#     try:
#         config = load_config(eval_args.config_path)
#         model_cfg = Munch(config['model_params'])
#         train_cfg = Munch(config['train_params'])
#     except FileNotFoundError: print(f"Error: Main config not found: {eval_args.config_path}"); return
#     except Exception as e: print(f"Error loading main config: {e}"); return

#     # --- 2. 设置设备 ---
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"Using device: {device}")

#     # --- 3. 构建并加载 EMA 模型 ---
#     print("Building EMA models...")
#     nets_ema = Munch()
#     try:
#         nets_ema.generator = Generator(model_cfg.img_size, model_cfg.style_dim, w_hpf=model_cfg.w_hpf, max_conv_dim=model_cfg.max_conv_dim)
#         nets_ema.mapping_network = MappingNetwork(model_cfg.latent_dim, model_cfg.style_dim, model_cfg.num_domains)
#         # StyleEncoder is needed for reference mode LPIPS, load it if potentially needed
#         if eval_args.mode == 'reference' or eval_args.calculate_lpips: # Load if reference or lpips needed
#              nets_ema.style_encoder = StyleEncoder(model_cfg.img_size, model_cfg.style_dim, model_cfg.num_domains, max_conv_dim=model_cfg.max_conv_dim)

#     except AttributeError as e: print(f"Error building models. Missing param in config? {e}"); return

#     print("Loading checkpoint...")
#     checkpoint_dir = Path(train_cfg.checkpoint_dir)
#     checkpoint_path = checkpoint_dir / f"{eval_args.resume_iter:06d}.ckpt"

#     try:
#         checkpoint = load_checkpoint(checkpoint_path, device)
#         if not isinstance(checkpoint, dict): print("Error: Loaded checkpoint not dict."); return

#         print("Loading EMA model states...")
#         for name, net_ema in nets_ema.items():
#              loaded_from = None; state_dict_to_load = None
#              if name in checkpoint.get('nets_ema', {}): state_dict_to_load = checkpoint['nets_ema'][name]; loaded_from = 'nets_ema'
#              elif name in checkpoint.get('nets', {}): state_dict_to_load = checkpoint['nets'][name]; loaded_from = 'nets'; print(f"Warn: EMA {name} not found, loading train state.")
#              else: print(f"Warn: State for {name} not found.")
#              if state_dict_to_load:
#                   try: net_ema.load_state_dict(state_dict_to_load); print(f"  Loaded state for nets_ema.{name} from '{loaded_from}'.")
#                   except Exception as e_load: print(f"  Error loading state for {name}: {e_load}.")

#         for net_ema in nets_ema.values(): net_ema.to(device); net_ema.eval()
#         print("EMA models loaded.")

#     except FileNotFoundError as e: print(f"Error: {e}"); return
#     except Exception as e: print(f"Error during checkpoint loading: {e}"); return

#     # --- 4. 定义参数和目录 ---
#     real_img_dir = Path(eval_args.real_img_dir)
#     eval_output_dir = Path(eval_args.eval_output_dir)
#     tmp_fake_dir = eval_output_dir / f"fake_images_iter{eval_args.resume_iter}"
#     mode = eval_args.mode
#     batch_size = eval_args.batch_size
#     num_domains = model_cfg.num_domains
#     latent_dim = model_cfg.latent_dim
#     num_fakes_per_task = eval_args.num_fakes_for_fid
#     lpips_num_samples = eval_args.num_samples_per_input
#     img_size_eval = 299 # Force 299 for FID/LPIPS eval transforms
#     img_size_model = model_cfg.img_size # Model input size

#     if not real_img_dir.is_dir(): print(f"Error: Real image dir not found: {real_img_dir}"); return
#     try: domains = sorted([d.name for d in real_img_dir.iterdir() if d.is_dir()])
#     except Exception as e: print(f"Error listing domains in {real_img_dir}: {e}"); return
#     if len(domains) != num_domains: print(f"Error: Found {len(domains)} dirs in {real_img_dir}, expected {num_domains}. Found: {domains}"); return
#     print(f"Found domains: {domains}")
#     domain_to_idx = {name: i for i, name in enumerate(domains)}
#     idx_to_domain = {i: name for i, name in enumerate(domains)}

#     eval_output_dir.mkdir(parents=True, exist_ok=True)
#     if tmp_fake_dir.exists(): print(f"Removing existing temp dir: {tmp_fake_dir}"); shutil.rmtree(tmp_fake_dir)
#     tmp_fake_dir.mkdir(parents=True)

#     # Transform for source images fed into generator
#     source_transform = T.Compose([
#         T.Resize((img_size_model, img_size_model)), # Resize to model input size
#         T.ToTensor(),
#         T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Normalize to [-1, 1]
#     ])

#     # --- 5. 生成合成图像 (用于 FID) ---
#     print(f"\nGenerating {num_fakes_per_task} fake images per task for FID...")
#     source_domain_name = eval_args.source_domain_fid
#     if source_domain_name not in domains: print(f"Error: Source domain '{source_domain_name}' not found."); return
#     source_img_dir = real_img_dir / source_domain_name
#     if not source_img_dir.is_dir(): print(f"Error: Source image dir not found: {source_img_dir}"); return

#     print(f"Using source domain: {source_domain_name}")
#     try:
#         # Use EvalDataset but with generator's input transform
#         source_dataset = EvalDataset(root=source_img_dir, img_size=img_size_model, transform=source_transform, file_limit=num_fakes_per_task)
#         source_loader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True, num_workers=eval_args.num_workers, pin_memory=True)
#     except FileNotFoundError: print(f"Error: No source images found in {source_img_dir}"); return
#     except Exception as e: print(f"Error creating source dataloader: {e}"); return

#     generated_counts = defaultdict(int)
#     target_domains = [d for d in domains if d != source_domain_name]

#     for target_domain_name in target_domains:
#         target_label = domain_to_idx[target_domain_name]
#         task_fake_dir = tmp_fake_dir / f"{source_domain_name}2{target_domain_name}"
#         task_fake_dir.mkdir(parents=True, exist_ok=True)
#         print(f"  Generating for task: {source_domain_name} -> {target_domain_name}")
#         pbar_task = tqdm(total=num_fakes_per_task, desc=f" Task {source_domain_name}2{target_domain_name}")
#         source_iter = iter(source_loader)

#         while generated_counts[target_domain_name] < num_fakes_per_task:
#             try: x_src_batch = next(source_iter)
#             except StopIteration: source_iter = iter(source_loader); x_src_batch = next(source_iter)
#             except Exception as e_load: print(f"Error loading source batch: {e_load}"); break

#             x_src_batch = x_src_batch.to(device)
#             current_batch_size = x_src_batch.size(0)
#             needed = num_fakes_per_task - generated_counts[target_domain_name]
#             if current_batch_size > needed: x_src_batch = x_src_batch[:needed]; current_batch_size = needed
#             if current_batch_size <= 0: break

#             y_trg_batch = torch.tensor([target_label] * current_batch_size).to(device)
#             z_trg_batch = torch.randn(current_batch_size, latent_dim).to(device)

#             with torch.no_grad():
#                 s_trg_batch = nets_ema.mapping_network(z_trg_batch, y_trg_batch)
#                 x_fake_batch = nets_ema.generator(x_src_batch, s_trg_batch)

#             for i in range(current_batch_size):
#                 if generated_counts[target_domain_name] >= num_fakes_per_task: break
#                 img_fake = denormalize(x_fake_batch[i]) # Denormalize to [0, 1]
#                 try:
#                      filename = task_fake_dir / f"fake_{generated_counts[target_domain_name]:05d}.png"
#                      torchvision.utils.save_image(img_fake.cpu(), filename, nrow=1, padding=0)
#                      generated_counts[target_domain_name] += 1
#                      pbar_task.update(1)
#                 except Exception as e_save: print(f"Warn: Failed to save {filename}: {e_save}")
#         pbar_task.close()
#         if generated_counts[target_domain_name] < num_fakes_per_task: print(f"Warn: Only generated {generated_counts[target_domain_name]} for {target_domain_name}")

#     # --- 6. Calculate FID ---
#     print("\nCalculating FID scores...")
#     fid_values = OrderedDict()
#     fid_batch_size = eval_args.fid_batch_size
#     fid_num_workers = eval_args.num_workers

#     for target_domain_name in target_domains:
#         task = f"{source_domain_name}2{target_domain_name}"
#         path_real = str(real_img_dir / target_domain_name)
#         path_fake = str(tmp_fake_dir / task)

#         if not os.path.exists(path_real) or not os.listdir(path_real): print(f"Warn: Real dir {path_real} empty/missing. Skip FID {task}."); continue
#         if not os.path.exists(path_fake) or not os.listdir(path_fake): print(f"Warn: Fake dir {path_fake} empty/missing. Skip FID {task}."); continue

#         print(f"Calculating FID for {task}...")
#         try:
#             fid_value = calculate_fid_given_paths(
#                 paths=[path_real, path_fake], img_size=img_size_eval, # Use 299 for FID
#                 batch_size=fid_batch_size, device=device,
#                 dims=eval_args.inception_dims, num_workers=fid_num_workers
#             )
#             fid_values[f'FID/{task}'] = fid_value
#             print(f"  FID for {task}: {fid_value:.4f}")
#         except Exception as e_fid: print(f"Error calculating FID for {task}: {e_fid}")

#     if fid_values:
#         fid_mean = np.mean(list(fid_values.values()))
#         fid_values[f'FID/mean'] = fid_mean
#         print(f"Mean FID: {fid_mean:.4f}")
#     else: print("No FID values calculated.")

#     # Save FID results
#     # --- !! 使用 eval_args.resume_iter !! ---
#     fid_filename = eval_output_dir / f"FID_{eval_args.resume_iter:06d}_{mode}.json"
#     try:
#         with open(fid_filename, 'w') as f: json.dump(fid_values, f, indent=4)
#         print(f"FID results saved to {fid_filename}")
#     except Exception as e_json: print(f"Error saving FID JSON: {e_json}")


#     # --- 7. Calculate LPIPS Diversity (Simplified) ---
#     if eval_args.calculate_lpips:
#         print("\nCalculating LPIPS diversity...")
#         lpips_dict = OrderedDict()
#         try:
#             lpips_model = calculate_lpips_given_images.get_lpips_model(device=device)
#         except Exception as e_lpips_load: print(f"Error loading LPIPS model: {e_lpips_load}. Skip LPIPS."); lpips_model = None

#         if lpips_model:
#             # Reload source loader with specific transform for LPIPS if needed (usually [-1, 1])
#             lpips_transform = T.Compose([
#                 T.Resize((img_size_model, img_size_model)), # Model input size
#                 T.ToTensor(),
#                 T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Range [-1, 1]
#             ])
#             lpips_source_dataset = EvalDataset(root=source_img_dir, img_size=img_size_model, transform=lpips_transform, file_limit=50) # Use fewer images for LPIPS
#             lpips_source_loader = DataLoader(lpips_source_dataset, batch_size=eval_args.lpips_batch_size, shuffle=True, num_workers=eval_args.num_workers)

#             for target_domain_name in target_domains:
#                 task = f"{source_domain_name}2{target_domain_name}"
#                 print(f"Calculating LPIPS for {task}...")
#                 lpips_values_task = []
#                 try:
#                     for x_src_batch in tqdm(lpips_source_loader, desc=f" LPIPS {task}"):
#                         x_src_batch = x_src_batch.to(device)
#                         current_batch_size = x_src_batch.size(0)
#                         y_trg_batch = torch.tensor([domain_to_idx[target_domain_name]] * current_batch_size).to(device)
#                         group_of_images = []
#                         with torch.no_grad():
#                             for _ in range(lpips_num_samples):
#                                  z_trg_batch = torch.randn(current_batch_size, latent_dim).to(device)
#                                  s_trg_batch = nets_ema.mapping_network(z_trg_batch, y_trg_batch)
#                                  x_fake = nets_ema.generator(x_src_batch, s_trg_batch)
#                                  group_of_images.append(x_fake) # Keep in [-1, 1] range for LPIPS model

#                         lpips_val = calculate_lpips_given_images(group_of_images, lpips_model)
#                         lpips_values_task.append(lpips_val)

#                     if lpips_values_task:
#                         lpips_mean_task = np.mean(lpips_values_task)
#                         lpips_dict[f'LPIPS/{task}'] = lpips_mean_task
#                         print(f"  LPIPS diversity for {task}: {lpips_mean_task:.4f}")

#                 except Exception as e_lpips: print(f"Error calculating LPIPS for {task}: {e_lpips}")

#             if lpips_dict:
#                 lpips_mean = np.mean(list(lpips_dict.values()))
#                 lpips_dict[f'LPIPS/mean'] = lpips_mean
#                 print(f"Mean LPIPS diversity: {lpips_mean:.4f}")
#             else: print("No LPIPS values calculated.")

#             # Save LPIPS results
#             # --- !! 使用 eval_args.resume_iter !! ---
#             lpips_filename = eval_output_dir / f"LPIPS_{eval_args.resume_iter:06d}_{mode}.json"
#             try:
#                 with open(lpips_filename, 'w') as f: json.dump(lpips_dict, f, indent=4)
#                 print(f"LPIPS results saved to {lpips_filename}")
#             except Exception as e_json: print(f"Error saving LPIPS JSON: {e_json}")
#     else:
#          print("Skipping LPIPS calculation as per arguments.")


#     # --- 8. Clean up temporary fake images (Optional) ---
#     if eval_args.cleanup_fake:
#         print(f"\nCleaning up temporary fake images in {tmp_fake_dir}...")
#         try: shutil.rmtree(tmp_fake_dir); print("Cleanup complete.")
#         except Exception as e_clean: print(f"Error cleaning up fake images: {e_clean}")

#     print("\nEvaluation script finished.")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Evaluate StarGAN v2 Model Quality (FID and LPIPS Diversity)")

#     # Required arguments
#     parser.add_argument('--config_path', type=str, required=True, help='Path to the main stargan_v2_config.yaml file.')
#     parser.add_argument('--resume_iter', type=int, required=True, help='Checkpoint iteration number to evaluate.')
#     parser.add_argument('--real_img_dir', type=str, required=True, help='Path to the directory containing REAL images organized by class subdirectories.')
#     parser.add_argument('--eval_output_dir', type=str, required=True, help='Directory to save evaluation outputs (generated images for FID, metrics JSON).')
#     parser.add_argument('--source_domain_fid', type=str, required=True, help='Name of the source domain (e.g., NV) used to generate fakes for FID.')

#     # Optional arguments
#     parser.add_argument('--mode', type=str, default='latent', choices=['latent'], help='Evaluation mode (currently only latent supported).')
#     parser.add_argument('--batch_size', type=int, default=16, help='Batch size for generating images.')
#     parser.add_argument('--fid_batch_size', type=int, default=32, help='Batch size for FID calculation (InceptionV3 inference).')
#     parser.add_argument('--lpips_batch_size', type=int, default=16, help='Batch size for LPIPS calculation.')
#     parser.add_argument('--num_fakes_for_fid', type=int, default=1000, help='Number of fake images to generate per task for FID calculation.')
#     parser.add_argument('--inception_dims', type=int, default=768, choices=[768, 2048], help='Dimensionality of Inception features to use.')
#     parser.add_argument('--num_samples_per_input', type=int, default=10, help='Number of outputs per input for LPIPS diversity.')
#     parser.add_argument('--num_workers', type=int, default=0, help='Number of dataloader workers (set to 0 if issues occur).') # Default to 0 for safety
#     parser.add_argument('--calculate_lpips', action='store_true', help='Calculate LPIPS diversity score (can be slow).')
#     parser.add_argument('--cleanup_fake', action='store_true', help='Remove temporary generated fake images after evaluation.')


#     args = parser.parse_args()

#     # --- Basic input validation ---
#     if not Path(args.config_path).exists(): raise FileNotFoundError(f"Config file not found: {args.config_path}")
#     if not Path(args.real_img_dir).is_dir(): raise NotADirectoryError(f"Real image directory not found: {args.real_img_dir}")
#     if args.num_fakes_for_fid <= 0: raise ValueError("--num_fakes_for_fid must be positive.")
#     # Add more checks as needed

#     # --- !! 修改这里：调用 evaluate_stargan 而不是 main !! ---
#     evaluate_stargan(args)




# python evaluation\evaluate_stargan_quality.py ^
#     --config_path .\configs\stargan_v2_config.yaml ^
#     --resume_iter 80000 ^
#     --real_img_dir .\data\ISIC2019\real_images_for_eval ^
#     --eval_output_dir .\expr\stargan_v2_isic\evaluation_results_80k ^
#     --source_domain_fid NV ^
#     --num_fakes_for_fid 1000 ^
#     --inception_dims 2048 ^
#     --batch_size 16 ^
#     --fid_batch_size 32 ^
#     --calculate_lpips ^
#     --lpips_batch_size 8 ^
#     --num_images_lpips 50 ^
#     --num_batches_lpips 5 ^
#     --num_samples_per_input 5 ^
#     --cleanup_fake ^
#     --num_workers 0










# pytorch_project/evaluation/evaluate_stargan_quality.py

import os
import shutil
import random
from pathlib import Path
import yaml
from munch import Munch
import json
from collections import OrderedDict, defaultdict
import argparse

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
import torchvision.transforms as T
import torchvision
import sys # Import sys for path manipulation

# --- 添加项目根目录到 sys.path ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
    print(f"Added {PROJECT_ROOT} to sys.path")

# --- 导入必要的模块 ---
try:
    from model.stargan_v2 import Generator, MappingNetwork, StyleEncoder
    from utils.helpers import load_config, load_checkpoint
    from utils.stargan_utils import denormalize
    # --- 导入核心 FID 和 LPIPS 计算函数 (从我们复制并适配的文件) ---
    from metrics.fid import calculate_fid_given_paths
    from metrics.lpips import calculate_lpips_given_images # <<< 使用您提供的 lpips.py 中的函数
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Ensure models, utils, helpers, and adapted metrics (fid.py, lpips.py) exist.")
    sys.exit(1)
except FileNotFoundError as e_fnf:
     print(f"Error: A required file (like lpips_weights.ckpt in metrics/) might be missing: {e_fnf}")
     sys.exit(1)


# --- 简单的评估数据集 ---
class EvalDataset(Dataset):
    """Simple Dataset for loading images for evaluation."""
    def __init__(self, root, img_size=299, transform=None, file_limit=None): # Default img_size to 299
        print(f"Scanning directory: {root}")
        root_path = Path(root)
        if not root_path.is_dir():
             raise FileNotFoundError(f"Directory not found: {root}")

        self.files = list(root_path.rglob('*.[pP][nN][gG]')) + \
                     list(root_path.rglob('*.[jJ][pP][gG]')) + \
                     list(root_path.rglob('*.[jJ][pP][eE][gG]'))
        if not self.files:
            # It's okay for fake image dir to be initially empty during FID calc before generation
            print(f"Warning: No image files found in {root}")
            # raise FileNotFoundError(f"No image files found in {root}") # Don't raise error here

        self.files.sort()
        if file_limit is not None and file_limit > 0:
             if len(self.files) > file_limit:
                  print(f"Limiting dataset from {len(self.files)} to {file_limit} files.")
                  self.files = self.files[:file_limit]
             # else: # No need to print if using all
             #     print(f"Found {len(self.files)} files, less than limit {file_limit}. Using all.")


        self.transform = transform
        if self.transform is None:
            # Default transform for InceptionV3/LPIPS
            self.transform = T.Compose([
                T.Resize((img_size, img_size)), # Use passed img_size
                T.ToTensor(), # Range [0, 1]
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet norm
            ])
        self._length = len(self.files)
        # Only print if files were found
        if self._length > 0:
            print(f"Initialized EvalDataset with {self._length} images from {root}")

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        path = self.files[i]
        try:
            img = Image.open(path).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            return img
        except Exception as e:
            print(f"Warning: Error loading image {path} for evaluation: {e}")
            # Determine output shape based on transform
            c, h, w = 3, 299, 299 # Default
            if isinstance(self.transform, T.Compose):
                 for t in reversed(self.transform.transforms):
                      if isinstance(t, T.Resize):
                           size = t.size
                           if isinstance(size, int): h=w=size
                           elif isinstance(size, (list, tuple)) and len(size) == 2: h, w = size
                           break
            return torch.zeros((c, h, w))


# --- 主评估函数 ---
def evaluate_stargan(eval_args): # Renamed from main
    """Performs StarGAN v2 evaluation (FID and LPIPS Diversity)."""

    # --- 1. 加载主配置文件 ---
    try:
        config = load_config(eval_args.config_path)
        model_cfg = Munch(config['model_params'])
        train_cfg = Munch(config['train_params'])
    except FileNotFoundError: print(f"Error: Main config not found: {eval_args.config_path}"); return
    except Exception as e: print(f"Error loading main config: {e}"); return

    # --- 2. 设置设备 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- 3. 构建并加载 EMA 模型 ---
    print("Building EMA models...")
    nets_ema = Munch()
    try:
        nets_ema.generator = Generator(model_cfg.img_size, model_cfg.style_dim, w_hpf=model_cfg.w_hpf, max_conv_dim=model_cfg.max_conv_dim)
        nets_ema.mapping_network = MappingNetwork(model_cfg.latent_dim, model_cfg.style_dim, model_cfg.num_domains)
        # Load Style Encoder only if needed for LPIPS reference mode (if implemented later)
        # if eval_args.mode == 'reference':
        #     nets_ema.style_encoder = StyleEncoder(model_cfg.img_size, model_cfg.style_dim, model_cfg.num_domains, max_conv_dim=model_cfg.max_conv_dim)
    except AttributeError as e: print(f"Error building models. Missing param in config? {e}"); return

    print("Loading checkpoint...")
    checkpoint_dir = Path(train_cfg.checkpoint_dir)
    checkpoint_path = checkpoint_dir / f"{eval_args.resume_iter:06d}.ckpt"

    try:
        checkpoint = load_checkpoint(checkpoint_path, device)
        if not isinstance(checkpoint, dict): print("Error: Loaded checkpoint not dict."); return

        print("Loading EMA model states...")
        for name, net_ema in nets_ema.items():
             loaded_from = None; state_dict_to_load = None
             # Check for both 'nets_ema' and 'nets' keys
             ema_key = name # Assuming keys in checkpoint match nets_ema keys
             if ema_key in checkpoint.get('nets_ema', {}):
                 state_dict_to_load = checkpoint['nets_ema'][ema_key]; loaded_from = 'nets_ema'
             elif ema_key in checkpoint.get('nets', {}):
                  state_dict_to_load = checkpoint['nets'][ema_key]; loaded_from = 'nets'; print(f"Warn: EMA {name} not found, loading train state.")
             else: print(f"Warn: State for {name} not found in checkpoint.")

             if state_dict_to_load:
                  try: net_ema.load_state_dict(state_dict_to_load); print(f"  Loaded state for {name} from '{loaded_from}'.")
                  except Exception as e_load: print(f"  Error loading state for {name}: {e_load}.")

        for net_ema in nets_ema.values(): net_ema.to(device); net_ema.eval()
        print("EMA models loaded.")

    except FileNotFoundError as e: print(f"Error: {e}"); return
    except Exception as e: print(f"Error during checkpoint loading: {e}"); return

    # --- 4. 定义参数和目录 ---
    real_img_dir = Path(eval_args.real_img_dir)
    eval_output_dir = Path(eval_args.eval_output_dir)
    # Create a dedicated subdir for temporary fake images, named by iteration
    tmp_fake_dir = eval_output_dir / f"fake_images_iter{eval_args.resume_iter}"
    mode = eval_args.mode # 'latent' or 'reference'
    batch_size = eval_args.batch_size
    num_domains = model_cfg.num_domains
    latent_dim = model_cfg.latent_dim
    num_fakes_per_task = eval_args.num_fakes_for_fid
    lpips_num_samples = eval_args.num_samples_per_input
    img_size_eval = 299 # Force 299 for FID/LPIPS eval transforms
    img_size_model = model_cfg.img_size # Model input size

    if not real_img_dir.is_dir(): print(f"Error: Real image dir not found: {real_img_dir}"); return
    try: domains = sorted([d.name for d in real_img_dir.iterdir() if d.is_dir()])
    except Exception as e: print(f"Error listing domains in {real_img_dir}: {e}"); return
    if len(domains) != num_domains: print(f"Error: Found {len(domains)} dirs in {real_img_dir}, expected {num_domains}. Found: {domains}"); return
    print(f"Found domains: {domains}")
    domain_to_idx = {name: i for i, name in enumerate(domains)}
    idx_to_domain = {i: name for i, name in enumerate(domains)}

    eval_output_dir.mkdir(parents=True, exist_ok=True)
    if tmp_fake_dir.exists(): print(f"Removing existing temp dir: {tmp_fake_dir}"); shutil.rmtree(tmp_fake_dir)
    tmp_fake_dir.mkdir(parents=True)

    # Transform for source images fed into generator [-1, 1] range
    source_transform = T.Compose([
        T.Resize((img_size_model, img_size_model)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # --- 5. 生成合成图像 (用于 FID) ---
    print(f"\nGenerating {num_fakes_per_task} fake images per task for FID calculation...")
    source_domain_name = eval_args.source_domain_fid
    if source_domain_name not in domains: print(f"Error: Source domain '{source_domain_name}' not found."); return
    source_img_dir = real_img_dir / source_domain_name
    if not source_img_dir.is_dir(): print(f"Error: Source image dir not found: {source_img_dir}"); return

    print(f"Using source domain: {source_domain_name}")
    try:
        # Load source images using EvalDataset with source_transform
        source_dataset = EvalDataset(root=source_img_dir, img_size=img_size_model, transform=source_transform, file_limit=num_fakes_per_task)
        source_loader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True, num_workers=eval_args.num_workers, pin_memory=True)
    except FileNotFoundError: print(f"Error: No source images found in {source_img_dir}"); return
    except Exception as e: print(f"Error creating source dataloader: {e}"); return

    generated_counts = defaultdict(int)
    target_domains = [d for d in domains if d != source_domain_name]

    for target_domain_name in target_domains:
        target_label = domain_to_idx[target_domain_name]
        task_fake_dir = tmp_fake_dir / f"{source_domain_name}2{target_domain_name}"
        task_fake_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Generating for task: {source_domain_name} -> {target_domain_name}")
        pbar_task = tqdm(total=num_fakes_per_task, desc=f" Task {source_domain_name}2{target_domain_name}")
        source_iter = iter(source_loader)

        while generated_counts[target_domain_name] < num_fakes_per_task:
            try: x_src_batch = next(source_iter)
            except StopIteration: source_iter = iter(source_loader); x_src_batch = next(source_iter)
            except Exception as e_load: print(f"Error loading source batch: {e_load}"); break

            x_src_batch = x_src_batch.to(device)
            current_batch_size = x_src_batch.size(0)
            needed = num_fakes_per_task - generated_counts[target_domain_name]
            if current_batch_size > needed: x_src_batch = x_src_batch[:needed]; current_batch_size = needed
            if current_batch_size <= 0: break

            y_trg_batch = torch.tensor([target_label] * current_batch_size).to(device)
            z_trg_batch = torch.randn(current_batch_size, latent_dim).to(device)

            with torch.no_grad():
                s_trg_batch = nets_ema.mapping_network(z_trg_batch, y_trg_batch)
                x_fake_batch = nets_ema.generator(x_src_batch, s_trg_batch)

            for i in range(current_batch_size):
                if generated_counts[target_domain_name] >= num_fakes_per_task: break
                img_fake = denormalize(x_fake_batch[i]) # Denormalize to [0, 1]
                try:
                     filename = task_fake_dir / f"fake_{generated_counts[target_domain_name]:05d}.png"
                     torchvision.utils.save_image(img_fake.cpu(), filename, nrow=1, padding=0)
                     generated_counts[target_domain_name] += 1
                     pbar_task.update(1)
                except Exception as e_save: print(f"Warn: Failed to save {filename}: {e_save}")
        pbar_task.close()
        if generated_counts[target_domain_name] < num_fakes_per_task: print(f"Warn: Only generated {generated_counts[target_domain_name]} for {target_domain_name}")

    # --- 6. Calculate FID ---
    print("\nCalculating FID scores...")
    fid_values = OrderedDict()
    fid_batch_size = eval_args.fid_batch_size
    fid_num_workers = eval_args.num_workers

    for target_domain_name in target_domains:
        task = f"{source_domain_name}2{target_domain_name}"
        path_real = str(real_img_dir / target_domain_name)
        path_fake = str(tmp_fake_dir / task)

        if not os.path.exists(path_real) or not os.listdir(path_real): print(f"Warn: Real dir {path_real} empty/missing. Skip FID {task}."); continue
        if not os.path.exists(path_fake) or not os.listdir(path_fake): print(f"Warn: Fake dir {path_fake} empty/missing. Skip FID {task}."); continue

        print(f"Calculating FID for {task}...")
        try:
            # Call the function from metrics.fid
            fid_value = calculate_fid_given_paths(
                paths=[path_real, path_fake], img_size=img_size_eval, # Use 299 for FID
                batch_size=fid_batch_size, device=device,
                dims=eval_args.inception_dims, num_workers=fid_num_workers
            )
            fid_values[f'FID/{task}'] = fid_value
            print(f"  FID for {task}: {fid_value:.4f}")
        except Exception as e_fid: print(f"Error calculating FID for {task}: {e_fid}")

    if fid_values:
        fid_mean = np.mean(list(fid_values.values()))
        fid_values[f'FID/mean'] = fid_mean
        print(f"Mean FID: {fid_mean:.4f}")
    else: print("No FID values calculated.")

    # Save FID results
    fid_filename = eval_output_dir / f"FID_{eval_args.resume_iter:06d}_{mode}.json"
    try:
        with open(fid_filename, 'w') as f: json.dump(fid_values, f, indent=4)
        print(f"FID results saved to {fid_filename}")
    except Exception as e_json: print(f"Error saving FID JSON: {e_json}")


    # --- 7. Calculate LPIPS Diversity ---
    if eval_args.calculate_lpips:
        print("\nCalculating LPIPS diversity...")
        lpips_dict = OrderedDict()

        # --- !! LPIPS Data Loading (using [-1, 1] normalization) !! ---
        lpips_transform = T.Compose([
            T.Resize((model_cfg.img_size, model_cfg.img_size)), # Model input size
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Range [-1, 1]
        ])
        lpips_source_loader = None
        try:
            # Load fewer source images for LPIPS
            lpips_source_dataset = EvalDataset(root=source_img_dir, img_size=model_cfg.img_size, transform=lpips_transform, file_limit=eval_args.num_images_lpips)
            lpips_source_loader = DataLoader(lpips_source_dataset, batch_size=eval_args.lpips_batch_size, shuffle=True, num_workers=eval_args.num_workers)
        except FileNotFoundError: print(f"Error: Source images not found for LPIPS at {source_img_dir}. Skip LPIPS."); lpips_source_loader=None
        except Exception as e_load_lpips: print(f"Error creating LPIPS dataloader: {e_load_lpips}. Skip LPIPS."); lpips_source_loader=None
        # --- !! LPIPS Data Loading End !! ---

        if lpips_source_loader:
            for target_domain_name in target_domains:
                task = f"{source_domain_name}2{target_domain_name}"
                print(f"Calculating LPIPS for {task}...")
                lpips_values_task = []
                try:
                    num_lpips_batches = min(eval_args.num_batches_lpips, len(lpips_source_loader))
                    pbar_lpips = tqdm(enumerate(lpips_source_loader), total=num_lpips_batches, desc=f" LPIPS {task}")

                    for batch_count, x_src_batch in pbar_lpips:
                        if batch_count >= num_lpips_batches: break

                        x_src_batch = x_src_batch.to(device)
                        current_batch_size = x_src_batch.size(0)
                        y_trg_batch = torch.tensor([domain_to_idx[target_domain_name]] * current_batch_size).to(device)

                        # --- Generate multiple styles ---
                        group_of_images = []
                        with torch.no_grad():
                            for _ in range(lpips_num_samples):
                                 z_trg_batch = torch.randn(current_batch_size, latent_dim).to(device)
                                 s_trg_batch = nets_ema.mapping_network(z_trg_batch, y_trg_batch)
                                 x_fake = nets_ema.generator(x_src_batch, s_trg_batch)
                                 group_of_images.append(x_fake) # Keep in [-1, 1] range
                        # --- Generation finished ---

                        # --- !! Call calculate_lpips_given_images from metrics/lpips.py !! ---
                        if len(group_of_images) >= 2:
                             try:
                                 # Pass the list of generated tensors directly
                                 lpips_val = calculate_lpips_given_images(group_of_images)
                                 lpips_values_task.append(lpips_val)
                             except Exception as e_lpips_calc:
                                 print(f"Error during LPIPS calculation call: {e_lpips_calc}")
                        # --- !! Call finished !! ---

                    if lpips_values_task:
                        lpips_mean_task = np.mean(lpips_values_task)
                        lpips_dict[f'LPIPS/{task}'] = lpips_mean_task
                        print(f"  LPIPS diversity for {task}: {lpips_mean_task:.4f}")

                except Exception as e_lpips_task: print(f"Error during LPIPS task {task}: {e_lpips_task}")

            # Calculate and save mean LPIPS
            if lpips_dict:
                lpips_mean = np.mean(list(lpips_dict.values()))
                lpips_dict[f'LPIPS/mean'] = lpips_mean
                print(f"Mean LPIPS diversity: {lpips_mean:.4f}")
            else: print("No LPIPS values calculated.")

            lpips_filename = eval_output_dir / f"LPIPS_{eval_args.resume_iter:06d}_{mode}.json"
            try:
                with open(lpips_filename, 'w') as f: json.dump(lpips_dict, f, indent=4)
                print(f"LPIPS results saved to {lpips_filename}")
            except Exception as e_json: print(f"Error saving LPIPS JSON: {e_json}")
    else:
         print("Skipping LPIPS calculation as per arguments.")


    # --- 8. Clean up temporary fake images (Optional) ---
    if eval_args.cleanup_fake:
        print(f"\nCleaning up temporary fake images in {tmp_fake_dir}...")
        try: shutil.rmtree(tmp_fake_dir); print("Cleanup complete.")
        except Exception as e_clean: print(f"Error cleaning up fake images: {e_clean}")

    print("\nEvaluation script finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate StarGAN v2 Model Quality (FID and LPIPS Diversity)")

    # Required arguments
    parser.add_argument('--config_path', type=str, required=True, help='Path to the main stargan_v2_config.yaml file.')
    parser.add_argument('--resume_iter', type=int, required=True, help='Checkpoint iteration number to evaluate.')
    parser.add_argument('--real_img_dir', type=str, required=True, help='Path to the directory containing REAL images organized by class subdirectories.')
    parser.add_argument('--eval_output_dir', type=str, required=True, help='Directory to save evaluation outputs (generated images for FID, metrics JSON).')
    parser.add_argument('--source_domain_fid', type=str, required=True, help='Name of the source domain (e.g., NV) used to generate fakes for FID.')

    # Optional arguments for fine-tuning evaluation
    parser.add_argument('--mode', type=str, default='latent', choices=['latent'], help='Evaluation mode (currently only latent supported).')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for generating images during evaluation.')
    parser.add_argument('--fid_batch_size', type=int, default=32, help='Batch size for FID calculation (InceptionV3 inference).')
    parser.add_argument('--num_fakes_for_fid', type=int, default=1000, help='Number of fake images to generate per task for FID calculation.')
    parser.add_argument('--inception_dims', type=int, default=2048, choices=[768, 2048], help='Dimensionality of Inception features to use (default: 2048 for pool3).') # Default to 2048
    parser.add_argument('--calculate_lpips', action='store_true', help='Calculate LPIPS diversity score (can be slow).')
    parser.add_argument('--lpips_batch_size', type=int, default=8, help='Batch size used when generating images for LPIPS calculation.') # Smaller default maybe
    parser.add_argument('--num_images_lpips', type=int, default=50, help='Number of source images to use for LPIPS calculation.')
    parser.add_argument('--num_batches_lpips', type=int, default=5, help='Number of source batches to process for LPIPS.')
    parser.add_argument('--num_samples_per_input', type=int, default=5, help='Number of outputs per input for LPIPS diversity.') # Default to 5 for speed
    parser.add_argument('--num_workers', type=int, default=0, help='Number of dataloader workers (set to 0 if issues occur).')
    parser.add_argument('--cleanup_fake', action='store_true', help='Remove temporary generated fake images after evaluation.')


    args = parser.parse_args()

    # --- Basic input validation ---
    if not Path(args.config_path).exists(): raise FileNotFoundError(f"Config file not found: {args.config_path}")
    if not Path(args.real_img_dir).is_dir(): raise NotADirectoryError(f"Real image directory not found: {args.real_img_dir}")
    if args.num_fakes_for_fid <= 0: raise ValueError("--num_fakes_for_fid must be positive.")
    # Add more checks as needed

    # --- Call the main evaluation function ---
    evaluate_stargan(args) # Call the renamed function