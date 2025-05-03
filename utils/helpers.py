# pytorch_project/utils/helpers.py

import yaml
import torch
from pathlib import Path

def load_config(config_path):
    """加载 YAML 配置文件，并指定使用 UTF-8 编码""" # <--- 修改了注释
    try:
        # --- 主要修改点：添加 encoding='utf-8' ---
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"Config loaded from: {config_path}")
        return config
    except FileNotFoundError:
        print(f"Error: Config file not found at {config_path}")
        raise
    except Exception as e:
        print(f"Error loading or parsing config file {config_path}: {e}")
        raise

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    """
    保存检查点。
    Args:
        state (dict): 包含模型和/或优化器 state_dict 的字典。
                      例如: {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
        filename (str or Path): 保存检查点的文件路径。
    """
    filename = Path(filename)
    print(f"=> Saving checkpoint '{filename}'")
    # 确保目录存在
    filename.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, filename)
    print(f"Checkpoint saved successfully.")

# def load_checkpoint(model, optimizer=None, filename="my_checkpoint.pth.tar", device='cpu'):
#     """
#     加载检查点。可以只加载模型，或者同时加载模型和优化器。
#     Args:
#         model (torch.nn.Module): 要加载状态的模型。
#         optimizer (torch.optim.Optimizer, optional): 要加载状态的优化器。默认为 None。
#         filename (str or Path): 检查点文件的路径。
#         device (str or torch.device): 加载到的设备。
#     Returns:
#         int: 检查点中可能保存的起始 epoch 或迭代次数 (如果存在)，否则返回 0。
#              (注意: StarGAN v2 的原始 checkpoint 可能不存这个，需要适配)
#     """
#     filename = Path(filename)
#     if filename.is_file():
#         print(f"=> Loading checkpoint '{filename}'")
#         # 加载到指定设备
#         checkpoint = torch.load(filename, map_location=device)

#         # -- 加载模型状态 --
#         # 尝试匹配 StarGAN v2 可能的保存格式
#         if 'state_dict' in checkpoint: # 常见的 PyTorch 保存格式
#              model.load_state_dict(checkpoint['state_dict'])
#         elif 'model' in checkpoint: # 如果直接保存了 model state_dict
#             model.load_state_dict(checkpoint['model'])
#         elif 'generator' in checkpoint and isinstance(model, type(checkpoint['generator'])): # 如果保存了名为 generator 的 state_dict
#             model.load_state_dict(checkpoint['generator'])
#         elif 'nets' in checkpoint and 'generator' in checkpoint['nets']: # 适配可能的 nets 字典
#              model.load_state_dict(checkpoint['nets']['generator'])
#         elif 'nets_ema' in checkpoint and 'generator' in checkpoint['nets_ema']: # 适配可能的 nets_ema 字典
#              model.load_state_dict(checkpoint['nets_ema']['generator'])
#         else:
#             # 如果以上都不匹配，尝试直接加载整个 checkpoint (假设它就是 state_dict)
#             try:
#                 model.load_state_dict(checkpoint)
#                 print("Loaded state_dict directly from checkpoint object.")
#             except:
#                 print("Error: Could not determine model state_dict key in checkpoint.")
#                 # raise ValueError("Could not load model state_dict from checkpoint")


#         # -- 加载优化器状态 (如果提供了 optimizer) --
#         if optimizer is not None:
#             if 'optimizer' in checkpoint:
#                 try:
#                     optimizer.load_state_dict(checkpoint['optimizer'])
#                     print("Optimizer state loaded.")
#                 except:
#                      print("Warning: Could not load optimizer state_dict, perhaps it's incompatible.")
#             elif 'optims' in checkpoint and 'generator' in checkpoint['optims']: # 适配 StarGAN v2 的 optims 结构
#                  # 注意：这里只加载了 generator 的优化器，你需要根据实际保存的结构适配
#                  try:
#                       # 你需要确定哪个优化器对应传入的 optimizer
#                       # 这里假设传入的是 generator 的优化器
#                       optimizer.load_state_dict(checkpoint['optims']['generator'])
#                       print("Optimizer state loaded (assuming generator optimizer).")
#                  except:
#                      print("Warning: Could not load optimizer state_dict from optims structure.")

#             else:
#                 print("Warning: Optimizer state not found in checkpoint.")

#         # -- 获取起始迭代次数 (如果存在) --
#         start_iter = checkpoint.get('iter', 0) # 假设迭代次数保存在 'iter' 键中
#         start_epoch = checkpoint.get('epoch', 0) # 或者保存在 'epoch' 键中

#         print(f"=> Loaded checkpoint '{filename}' (iter {start_iter}, epoch {start_epoch})")
#         return start_iter or start_epoch # 返回迭代次数或 epoch
#     else:
#         print(f"=> No checkpoint found at '{filename}'")
#         return 0



def load_checkpoint(filename, device='cpu'):
    """Loads a checkpoint file."""
    filename = Path(filename)
    print(f"--- DEBUG: Inside MODIFIED load_checkpoint ---") # 确认调用的是新函数
    print(f"--- DEBUG: Checking if file exists: {filename} ---") # 确认路径参数正确
    if not filename.is_file():
        print(f"--- DEBUG: File not found! Raising FileNotFoundError. ---") # 确认进入了正确的分支
        raise FileNotFoundError(f"Checkpoint file not found at '{filename}'") # <--- 必须是抛出异常

    print(f"=> Loading checkpoint '{filename}'") # 只有文件存在时才打印这个
    try:
        checkpoint = torch.load(filename, map_location=device)
        print("Checkpoint loaded successfully from file.")
        return checkpoint
    except Exception as e:
        print(f"Error reading checkpoint file {filename}: {e}")
        raise