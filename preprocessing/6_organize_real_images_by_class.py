# pytorch_project/preprocessing/6_organize_real_images_by_class.py

import pandas as pd
from pathlib import Path
import shutil
from tqdm import tqdm
import sys

# --- 添加项目根目录到 sys.path (如果需要导入其他项目模块) ---
# PROJECT_ROOT = Path(__file__).resolve().parent.parent
# if str(PROJECT_ROOT) not in sys.path:
#     sys.path.append(str(PROJECT_ROOT))
#     print(f"Added {PROJECT_ROOT} to sys.path")

# --- 配置 (请根据您的实际情况修改) ---

# 1. K-Fold CSV 文件所在的目录
#    (应包含 train_fold_0.csv, train_fold_1.csv, ...)
FOLDS_DIR = Path("./data/ISIC2019/folds_pytorch")

# 2. 包含修复后 (inpainted) 图像的根目录
#    (这是步骤 3_inpaint.py 的输出目录)
INPAINTED_IMG_ROOT = Path(".data/ISIC2019/inpainted_pytorch/train")

# 3. 输出目录: 用于存放按类别整理好的真实图像
#    脚本会自动创建这个目录及其子目录
OUTPUT_REAL_DIR = Path("./data/ISIC2019/real_images_for_eval")

# 4. CSV 文件中的列名
LABEL_COLUMN = 'integer_label'    # 包含整数标签的列名
IMAGE_PATH_COLUMN = 'image_path'  # 包含图像路径的列名

# 5. 类别标签到目录名的映射 (确保与您的设置一致)
INT_TO_CLASS_MAPPING = {
    0: 'MEL', 1: 'NV', 2: 'BCC', 3: 'AK',
    4: 'BKL', 5: 'DF', 6: 'VASC', 7: 'SCC'
}
# 要处理的类别标签列表 (0 到 7)
TARGET_LABELS = list(INT_TO_CLASS_MAPPING.keys())

# 6. 您使用的 K-Fold 折数
NUM_FOLDS = 5 # 例如 5 折，请修改为您实际使用的折数

# --- 主逻辑 ---

def organize_images():
    """读取 K-Fold CSV，将修复后的图像按类别复制到新目录。"""
    print("Organizing real inpainted images by class for evaluation...")
    OUTPUT_REAL_DIR.mkdir(parents=True, exist_ok=True) # 创建输出根目录

    all_files_df_list = []
    print("Loading K-Fold CSV files...")
    for k in range(NUM_FOLDS):
        csv_path = FOLDS_DIR / f"train_fold_{k}.csv"
        if csv_path.exists():
            try:
                df_fold = pd.read_csv(csv_path)
                # --- 关键: 确认 CSV 中的 image_path 列内容 ---
                # 假设 CSV 中的路径是相对于项目根目录的，或者需要基于文件名重新构建
                # 我们需要确保能定位到 INPAINTED_IMG_ROOT 下的文件
                # 这里假设 CSV 中的 image_path 列包含了可以直接使用的、指向修复后图像的路径
                # 如果不是，您需要在这里修改路径构建逻辑
                print(f"  Loaded {len(df_fold)} entries from {csv_path.name}")
                all_files_df_list.append(df_fold)
            except KeyError as e:
                 print(f"Error: Column {e} not found in {csv_path}. Please check LABEL_COLUMN and IMAGE_PATH_COLUMN settings.")
                 return # Exit if column names are wrong
            except Exception as e:
                print(f"Error reading {csv_path}: {e}")
        else:
            print(f"Warning: CSV file not found: {csv_path}")

    if not all_files_df_list:
        print("Error: No K-Fold CSV files were loaded. Please check FOLDS_DIR and NUM_FOLDS settings.")
        return

    # 合并所有 folds 的信息到一个 DataFrame
    df_all = pd.concat(all_files_df_list, ignore_index=True)
    # 按图像路径去重，以防同一个图像在不同 fold 的训练集出现（理论上不应发生）
    df_all = df_all.drop_duplicates(subset=[IMAGE_PATH_COLUMN]).reset_index(drop=True)
    print(f"Processing information for {len(df_all)} unique training images.")

    copied_count = 0
    error_count = 0
    skipped_count = 0
    print("Copying images to class directories...")

    # 使用 tqdm 显示进度条
    for index, row in tqdm(df_all.iterrows(), total=len(df_all), desc="Organizing Images"):
        try:
            # 获取整数标签
            label_int = int(row[LABEL_COLUMN])

            # 只处理我们关心的类别 (0-7)
            if label_int in TARGET_LABELS:
                # 获取类别名称 (目录名)
                class_name = INT_TO_CLASS_MAPPING[label_int]
                # 获取源图像路径 (来自 CSV)
                src_path_str = row[IMAGE_PATH_COLUMN]
                src_path = Path(src_path_str)

                # --- !!! 非常重要：确定源文件路径 !!! ---
                # 检查 CSV 中的路径是否是绝对路径或可以直接使用
                if not src_path.is_absolute():
                    # 如果是相对路径，尝试相对于修复后的图像根目录拼接
                    # 注意：这假设 CSV 中的路径是相对于 INPAINTED_IMG_ROOT 的文件名或相对路径
                    # 如果 CSV 路径已经是正确的绝对路径或相对于项目根目录的路径，则不需要这步
                    src_path_check1 = INPAINTED_IMG_ROOT / src_path.name # 尝试只用文件名拼接
                    src_path_check2 = Path(src_path_str) # 尝试直接使用 CSV 中的路径

                    if src_path_check1.exists():
                         src_path = src_path_check1
                    elif src_path_check2.exists():
                         src_path = src_path_check2 # 假设 CSV 路径相对于运行目录有效
                    else:
                         # 如果都找不到，再尝试拼接原始相对路径（如果适用）
                         src_path_check3 = INPAINTED_IMG_ROOT / src_path_str
                         if src_path_check3.exists():
                              src_path = src_path_check3
                         else:
                              # 如果各种尝试都找不到，则跳过
                              print(f"Warning: Source image not found for path '{src_path_str}' in CSV (tried various combinations). Skipping row {index}.")
                              error_count += 1
                              continue
                elif not src_path.exists():
                     # 如果是绝对路径但文件不存在
                     print(f"Warning: Source image (absolute path) not found: {src_path}. Skipping row {index}.")
                     error_count += 1
                     continue


                # 确定目标目录和目标文件路径
                target_class_dir = OUTPUT_REAL_DIR / class_name
                target_class_dir.mkdir(exist_ok=True) # 创建类别子目录
                target_path = target_class_dir / src_path.name # 使用原始文件名

                # 复制文件，如果目标文件已存在则跳过 (避免重复复制)
                if not target_path.exists():
                    try:
                        shutil.copy2(src_path, target_path) # copy2 保留元数据
                        copied_count += 1
                    except Exception as e_copy:
                         print(f"Error copying {src_path} to {target_path}: {e_copy}")
                         error_count += 1
                # else: # Optional: track skipped files
                #     skipped_count += 1

            else: # 如果标签不在 TARGET_LABELS 中 (例如 -1 或其他)
                skipped_count += 1

        except KeyError as e:
             print(f"Error: Column {e} not found in DataFrame. Please check column names in CSV and script config.")
             error_count += 1
             # Potentially break or return if column names are wrong
        except ValueError:
             print(f"Error: Could not convert label '{row[LABEL_COLUMN]}' to integer for row {index}.")
             error_count += 1
        except Exception as e:
            # 捕获其他意外错误
            print(f"Error processing row {index} (path: {row.get(IMAGE_PATH_COLUMN, 'N/A')}, label: {row.get(LABEL_COLUMN, 'N/A')}): {e}")
            error_count += 1

    print("\nFinished organizing images.")
    print(f"Successfully copied: {copied_count} images.")
    if skipped_count > 0:
        print(f"Skipped (label not in target or file already exists): {skipped_count} entries.")
    if error_count > 0:
        print(f"Errors or files not found: {error_count}")
    print(f"Real images organized by class in: {OUTPUT_REAL_DIR}")


if __name__ == "__main__":
    organize_images()