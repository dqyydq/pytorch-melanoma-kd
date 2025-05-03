import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from pathlib import Path
import os

# -----------------------------------------------------------------------------
# 配置参数 (直接在这里修改)
# -----------------------------------------------------------------------------
# 基础数据目录，包含 ISIC 2019 数据集
BASE_DATA_DIR = 'D:/python_code/pytorch_melanama_kd/data/ISIC2019'

# ISIC 2019 训练集元数据 CSV 文件名 (相对于 BASE_DATA_DIR)
# !!! 请确保这是你实际的 CSV 文件名，格式如您截图所示 !!!
METADATA_CSV_FILENAME = 'ISIC_2019_Training_GroundTruth.csv' # 可能需要修改文件名

# 包含预处理后图像的目录名 (相对于 BASE_DATA_DIR)
# (这是上一步 1_crop_resize.py 的输出目录)
PROCESSED_IMAGE_DIR_NAME = 'centre_square_cropped_pytorch/train/'

# 保存输出的 K-Fold CSV 文件的目录名 (相对于 BASE_DATA_DIR)
OUTPUT_CSV_DIR_NAME = 'folds_pytorch/'

# K-Fold 交叉验证的折数
NUM_FOLDS = 5 # 你可以改成 10, 15 或其他值

# 用于复现的随机种子
RANDOM_SEED = 42

# ISIC 2019 元数据 CSV 文件中包含图像 ID/名称的列名
IMAGE_ID_COLUMN = 'image' # 这个看起来是正确的

# !!! 定义 CSV 中代表类别的列名 (根据您的截图) !!!
# !!! 检查是否需要包含 'UNK' 类，如果不需要，可以从列表中移除 !!!
CLASS_COLUMNS = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']

# 定义类别名称到整数标签的映射 (PyTorch 通常需要 0 到 N-1 的整数标签)
# 顺序可以自定义，但这将决定哪个整数代表哪个类
# 确保这个映射包含了 CLASS_COLUMNS 中的所有类别
CLASS_TO_INT_MAPPING = {
    'MEL': 0,
    'NV': 1,
    'BCC': 2,
    'AK': 3,
    'BKL': 4,
    'DF': 5,
    'VASC': 6,
    'SCC': 7
}
# 我们将创建的新整数标签列的名称
NEW_INTEGER_LABEL_COLUMN = 'integer_label'

# 图像文件的扩展名
IMAGE_EXTENSION = '.jpg'
# -----------------------------------------------------------------------------

def create_dir(dir_path):
    """如果目录不存在，则创建它。"""
    Path(dir_path).mkdir(parents=True, exist_ok=True)

def main():
    """主函数，执行分层 K 折拆分并创建 CSV 文件。"""

    # --- 使用上面定义的变量构建路径 ---
    base_data_path = Path(BASE_DATA_DIR)
    metadata_csv_path = base_data_path / METADATA_CSV_FILENAME
    processed_images_dir = base_data_path / PROCESSED_IMAGE_DIR_NAME
    output_csv_dir = base_data_path / OUTPUT_CSV_DIR_NAME

    create_dir(output_csv_dir)

    print(f"正在从以下路径读取元数据: {metadata_csv_path}")
    try:
        df = pd.read_csv(metadata_csv_path)
    except FileNotFoundError:
        print(f"错误: 在 {metadata_csv_path} 未找到元数据 CSV 文件")
        return
    except Exception as e:
        print(f"读取 CSV 文件时出错: {e}")
        return

    # --- 检查必要的列是否存在 ---
    if IMAGE_ID_COLUMN not in df.columns:
        print(f"错误: 图像 ID 列 '{IMAGE_ID_COLUMN}' 在 CSV 中未找到。")
        return
    for col in CLASS_COLUMNS:
        if col not in df.columns:
            print(f"错误: 类别列 '{col}' 在 CSV 中未找到。")
            return
    print("必要的列已找到。")

    # --- 将 One-Hot 编码转换为整数标签 ---
    print("正在将 One-Hot 编码转换为整数标签...")
    # 使用 idxmax 找到每行中值为 1 的列名（即类别名称）
    # axis=1 表示按行操作
    df_class_columns = df[CLASS_COLUMNS]
    # 确保每一行只有一个 1 (如果有多于一个或者全是0，idxmax 会有问题，需要额外处理)
    # 这里假设数据是干净的 one-hot 编码
    if not (df_class_columns.sum(axis=1) == 1).all():
         print("警告: CSV 中的类别列似乎不是严格的 One-Hot 编码 (每行不都只有一个 1)。")
         # 在这里可以添加错误处理或数据清理逻辑
         # 例如，过滤掉全 0 的行或有多个 1 的行
         # df = df[df_class_columns.sum(axis=1) == 1]

    class_names = df_class_columns.idxmax(axis=1)
    # 使用之前定义的映射将类别名称转换为整数
    df[NEW_INTEGER_LABEL_COLUMN] = class_names.map(CLASS_TO_INT_MAPPING)

    # 验证转换是否成功 (可选)
    if df[NEW_INTEGER_LABEL_COLUMN].isnull().any():
        print("错误: 转换整数标签时出现问题，可能存在未映射的类别名称。")
        # 打印出有问题的行或类别名称以帮助调试
        print("出现问题的类别名称:", class_names[df[NEW_INTEGER_LABEL_COLUMN].isnull()].unique())
        return
    print(f"整数标签已创建在列 '{NEW_INTEGER_LABEL_COLUMN}' 中。")

    # --- 构建完整的图像文件路径 ---
    print("正在构建完整的图像路径...")
    df['image_path'] = df[IMAGE_ID_COLUMN].apply(
        lambda x: str(processed_images_dir / f"{x}{IMAGE_EXTENSION}")
    )

    # 可选: 验证一个示例路径是否存在
    if not df.empty and not Path(df['image_path'].iloc[0]).exists():
         print(f"警告: 示例图像路径不存在: {df['image_path'].iloc[0]}")
         print(f"请确保 '{PROCESSED_IMAGE_DIR_NAME}' 目录名正确且图像已存在于 '{processed_images_dir}'。")

    # --- 执行分层 K 折拆分 ---
    print(f"正在执行分层 {NUM_FOLDS} 折拆分...")
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    # 使用图像 ID 作为 X，新创建的整数标签作为 y 进行分层
    X = df[IMAGE_ID_COLUMN]
    y = df[NEW_INTEGER_LABEL_COLUMN] # <--- 使用新的整数标签列

    fold_num = 0
    for train_index, val_index in skf.split(X, y):
        df_train = df.iloc[train_index].copy()
        df_val = df.iloc[val_index].copy()

        # --- 选择相关列并保存 CSV 文件 ---
        output_columns = ['image_path', NEW_INTEGER_LABEL_COLUMN] # <-- 输出包含整数标签的列

        train_csv_path = output_csv_dir / f"train_fold_{fold_num}.csv"
        val_csv_path = output_csv_dir / f"val_fold_{fold_num}.csv"

        # 保存 CSV，不包含索引列
        df_train[output_columns].to_csv(train_csv_path, index=False)
        df_val[output_columns].to_csv(val_csv_path, index=False)

        print(f"  第 {fold_num} 折: 训练集={len(df_train)}, 验证集={len(df_val)}")
        # 可选: 打印每折的类别分布以供验证
        # print(f"    训练集分布:\n{df_train[NEW_INTEGER_LABEL_COLUMN].value_counts(normalize=True)}")
        # print(f"    验证集分布:\n{df_val[NEW_INTEGER_LABEL_COLUMN].value_counts(normalize=True)}")

        fold_num += 1

    print(f"\n成功创建 {NUM_FOLDS} 个分层折。")
    print(f"CSV 文件保存在: {output_csv_dir}")

if __name__ == "__main__":
    main()