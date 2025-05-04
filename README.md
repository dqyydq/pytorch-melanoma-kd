
## 安装

1.  **克隆仓库:**
    ```bash
    git clone https://github.com/dqyydq/pytorch-melanoma-kd.git # 替换为您的仓库 URL
    cd pytorch-melanoma-kd
    ```
2.  **创建 Python 环境 (推荐):**
    ```bash
    conda create -n melanoma_kd python=3.9 # 或您使用的 Python 版本
    conda activate melanoma_kd
    ```
3.  **安装依赖:**
    ```bash
    pip install -r requirements.txt
    # 确保安装了支持 CUDA 的正确 PyTorch 版本, 例如:
    # pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    # 如果需要进行 LPIPS 评估，安装 LPIPS 库
    # pip install lpips
    ```
   
## 数据准备

1.  **下载 ISIC 2019 数据:** 获取 ISIC 2019 训练数据集（图像和 `ISIC_2019_Training_GroundTruth.csv` 元数据）。将它们按脚本期望的结构放置（例如放在 `data/ISIC2019/` 下）。
    *   训练图像: `data/ISIC2019/ISIC_2019_Training_Input/`
    *   元数据 CSV: `data/ISIC2019/ISIC_2019_Training_GroundTruth.csv`
    *   (如果需要单独处理测试集，也需准备相应数据)

2.  **运行预处理脚本:** **依次**执行 `preprocessing/` 目录下的脚本：
    *   **裁剪与缩放:**
        ```bash
        python preprocessing/1_crop_resize.py
        ```
        (输出到: `data/ISIC2019/centre_square_cropped_pytorch/`)
    *   **图像修复 (去毛发):**
        ```bash
        python preprocessing/3_inpaint.py # 确保修复函数已正确实现并导入
        ```
        (输出到: `data/ISIC2019/inpainted_pytorch/`)
    *   **创建 K 折交叉验证划分 (初始 - 用于 GAN 训练):**
        ```bash
        python preprocessing/2_create_splits.py
        ```
        (读取原始元数据，输出 K-Fold CSV 文件到 `data/ISIC2019/folds_pytorch/`，CSV 中的路径应指向**修复后**的图像)

3.  **(可选，但推荐用于评估) 按类别整理真实图像:** 准备用于计算 FID 的数据。
    ```bash
    python preprocessing/6_organize_real_images_by_class.py
    ```
    (读取 K-Fold CSVs，将**修复后**的图像按类别名称复制到 `data/ISIC2019/real_images_for_eval/`)

## 训练模型

在运行训练脚本前，请配置 `configs/` 目录下对应的 YAML 文件。确保路径和超参数设置正确。建议从项目根目录运行脚本。

1.  **训练 StarGAN v2:**
    ```bash
    python training/train_stargan.py --config configs/stargan_v2_config.yaml # 或使用脚本内默认路径
    ```
    *   使用 TensorBoard 监控训练: `tensorboard --logdir expr/stargan_v2_isic/logs`
    *   检查点保存在 `expr/stargan_v2_isic/checkpoints/`。

2.  **训练教师模型:** *(需要 `training/train_teacher.py` 和相应配置)*
    ```bash
    # 示例命令 (需实现对应脚本)
    # python training/train_teacher.py --config configs/teacher_config.yaml --fold 0
    ```

3.  **训练学生模型 (使用知识蒸馏):** *(需要 `training/train_student_kd.py` 和相应配置)*
    ```bash
    # 示例命令 (需实现对应脚本)
    # python training/train_student_kd.py --config configs/student_config.yaml --fold 0 --teacher_ckpt path/to/teacher.pth
    ```

## 生成合成数据 (使用训练好的 StarGAN v2)

1.  **配置生成参数:** 编辑 `configs/generation_config.yaml` 文件，指定要加载的 StarGAN v2 检查点 (`resume_iter`)、源域/目标域标签、要生成的图像数量、输出目录等。
2.  **运行生成脚本:**
    ```bash
    python generation/generate_synthetic_stargan.py --stargan_config configs/stargan_v2_config.yaml --gen_config configs/generation_config.yaml # 或使用默认路径
    ```
    (输出: 合成图像保存在 `generation_config.yaml` 中指定的目录)
3.  **创建平衡后的 CSV 文件:** 合并原始数据信息和合成数据信息。
    ```bash
    python preprocessing/5_create_stargan_balanced_csv.py --stargan_config configs/stargan_v2_config.yaml --gen_config configs/generation_config.yaml
    ```
    (输出: `isic2019_train_stargan_balanced.csv` 文件保存在 `data/ISIC2019/` 目录下)
4.  **为平衡数据集创建 K 折划分:** 修改并运行 `preprocessing/2_create_splits.py` 脚本，使其读取上一步生成的平衡 CSV 文件作为输入。

## 评估

1.  **评估 StarGAN v2 生成质量 (FID/LPIPS):**
    *   确保真实图像已按类别整理好 (参见数据准备步骤 3)。
    *   运行评估脚本:
        ```bash
        python evaluation/evaluate_stargan_quality.py ^
            --config_path configs/stargan_v2_config.yaml ^
            --resume_iter <ITERATION> ^
            --real_img_dir data/ISIC2019/real_images_for_eval ^
            --eval_output_dir expr/stargan_v2_isic/evaluation_results_<ITERATION> ^
            --source_domain_fid NV ^
            --num_fakes_for_fid 1000 ^
            --inception_dims 2048 ^
            --calculate_lpips
        ```
    *   评估结果（FID 和 LPIPS 值）将保存在 `--eval_output_dir` 指定目录下的 JSON 文件中。

2.  **评估分类器性能:**
    *   使用**平衡后数据集**的 K 折划分来训练教师/学生模型。
    *   在独立的测试集或 K 折验证集上评估训练好的分类器。
    *   计算多类别评估指标（准确率、各类别的 F1 分数、混淆矩阵等）。
    *   将结果与未使用 GAN 增强的基线模型进行比较。

## 结果展示

*(本部分需在获得实验结果后填写)*

*   **StarGAN v2 评估:**
    *   展示关键的 FID/LPIPS 分数 (例如 Mean FID)。
    *   展示一些有代表性的生成样本图像。
    *   (可选) 提供公开托管的 TensorBoard 日志链接。
*   **分类性能:**
    *   用表格展示关键指标（例如 准确率、宏平均 F1、MEL F1、BCC F1 等）的对比：
        *   基线模型（例如，在不平衡数据上使用加权损失/采样训练）
        *   使用 StarGAN v2 平衡数据训练的模型
    *   如果混淆矩阵有助于分析，也一并展示。

## 引用 / 致谢

*   本项目适配了论文 "[Melanoma classification from dermatoscopy images using knowledge distillation for highly imbalanced data](https://doi.org/10.1016/j.compbiomed.2023.106571)" 中的概念。
*   StarGAN v2 的实现基于官方仓库: [https://github.com/clovaai/stargan-v2](https://github.com/clovaai/stargan-v2)
*   数据集来源: ISIC Archive (本项目使用 ISIC 2019 挑战赛数据集)。如果需要，请按 ISIC 要求进行引用。
*   提及使用的其他重要库或资源。
