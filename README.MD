## Project Overview / 项目概述
This repository hosts the source code for the research titled **"Long-tailed Mineral Identification based on Multi-stage Training"**, submitted to the journal *Computers & Geosciences*. The code implements a multi-stage rebalancing strategy to address the data imbalance (long-tailed distribution) in mineral image recognition, significantly improving the recognition accuracy of tail-class (rare) minerals while enhancing overall model generalization.  
本仓库存储题为《基于多阶段训练的长尾矿物识别》（投稿至期刊《Computers & Geosciences》）的研究源代码。代码实现了多阶段重平衡策略，用于解决矿物图像识别中的数据不平衡（长尾分布）问题，在显著提升尾部类别（稀有）矿物识别精度的同时，增强模型整体泛化能力。

## Dataset / 数据集说明
The experiments are conducted on the **Minerals-36 dataset**, an enhanced version of existing mineral image datasets (Wang et al., 2024; Wu et al., 2022; Zeng et al., 2021).  
- **Composition**: 113,569 images covering 36 mineral categories, collected via web crawling from the Mindat website and filtered manually to remove low-quality samples.  
- **Distribution**: Long-tailed (imbalance factor = 81.4, meaning the most frequent class has 81.4× more samples than the least frequent one).  
- **Split**: Randomly divided into training (80%), validation (10%), and test (10%) sets.  

实验基于**Minerals-36数据集**开展，该数据集是现有矿物图像数据集（Wang et al., 2024; Wu et al., 2022; Zeng et al., 2021）的增强版本：  
- **构成**：包含113,569张图像，覆盖36种矿物类别，通过从Mindat网站爬虫获取并人工筛选低质量样本得到。  
- **分布**：呈长尾特征（不平衡因子=81.4，即样本最多类别数量是最少类别数量的81.4倍）。  
- **划分**：随机分为训练集（80%）、验证集（10%）和测试集（10%）。

## Method Introduction / 方法简介
The proposed multi-stage training framework consists of three core components, implemented in sequence:  
所提多阶段训练框架包含三个核心模块，按以下流程实现：  
1. **Feature Extractor Adjustment (FEA)**  
   Modifies the softmax cross-entropy loss into a **frequency-aware loss function**, which amplifies the gradient of tail-class samples by adding a class-frequency term (logπᵧ), ensuring the model focuses on underrepresented classes during training.  
   将softmax交叉熵损失改进为**频率感知损失函数**，通过添加类别频率项（logπᵧ）放大尾部类别样本的梯度，确保模型在训练中关注少数类。  

2. **Core Feature Learning (CFL)**  
   Introduces **inverse-probability sampling** to increase the sampling frequency of low-confidence samples, and adds **center loss** to guide the model to learn stable class core features, mitigating intra-class attribute imbalance.  
   引入**逆概率采样**提高低置信度样本的采样频率，并添加**中心损失**引导模型学习稳定的类别核心特征，缓解类内属性不平衡。  

3. **Classifier Re-training (cRT)**  
   Freezes the parameters of the feature extractor, re-trains the classifier with **class-balanced sampling** (equal sampling probability for each class), and corrects the classification boundary bias caused by long-tailed data.  
   冻结特征提取器参数，采用**类别平衡采样**（每类采样概率相等）重新训练分类器，修正长尾数据导致的分类边界偏移。

## Code Structure / 代码结构
```
Long-tailed-Mineral-Identification-based-on-Multi-stage-Training/
├── environment.yml       # Third-party library dependencies (PyTorch, scikit-learn, etc.)
                          # 第三方库依赖（PyTorch、scikit-learn等）
├── data_preprocess.py    # Data loading, partitioning, and augmentation (for Minerals-36)
                          # 数据加载、划分与增强（针对Minerals-36数据集）
├── model.py              # Definition of ResNet-50 backbone and multi-stage loss functions (FEA/CFL)
                          # ResNet-50骨干网络及多阶段损失函数（FEA/CFL）定义
├── train.py              # Full training pipeline (FEA → CFL → cRT) and parameter setting
                          # 完整训练流程（FEA→CFL→cRT）及参数配置
├── evaluate.py           # Evaluation metrics calculation (mAcc, CwASD, CwARange) and result visualization
                          # 评估指标计算（mAcc、CwASD、CwARange）及结果可视化
└── utils.py              # Auxiliary functions (log recording, model saving, etc.)
                          # 辅助函数（日志记录、模型保存等）
```

## Installation & Setup / 安装与配置
### 1. Hardware Requirements / 硬件要求
- GPU: NVIDIA graphics card with CUDA support and video memory ≥ 16G (consistent with the experimental configuration of NVIDIA Tesla P100)  
- CPU: Intel Core i7 or equivalent  
- RAM: ≥ 32G  

- 显卡：支持CUDA的NVIDIA显卡，显存≥16G（与实验所用NVIDIA Tesla P100配置一致）  
- 处理器：Intel Core i7或同等性能  
- 内存：≥32G  

### 2. Software Requirements / 软件要求
- Operating System: CentOS Linux 7.9 (or Windows 10/11, Ubuntu 20.04)  
- Python Version: 3.8–3.10  
- Dependencies: Install via `conda` using the provided `environment.yml`:  
  ```bash
  conda env create -f environment.yml
  conda activate mineral_recog
  ```

- 操作系统：CentOS Linux 7.9（或Windows 10/11、Ubuntu 20.04）  
- Python版本：3.8–3.10  
- 依赖库：通过`environment.yml`文件使用`conda`安装：  
  ```bash
  conda env create -f environment.yml
  conda activate mineral_recog
  ```

## Quick Start / 快速开始
### 1. Dataset Preparation / 数据集准备
- Download the Minerals-36 dataset (contact the corresponding author for access if needed: xhji@cugb.edu.cn).  
- Organize the dataset into the following structure:  
  ```
  Minerals-36/
  ├── train/
  │   ├── Albite/          # Folder for each mineral class (36 folders total)
  │   │   ├── img_1.jpg
  │   │   └── ...
  │   └── ...
  ├── val/
  │   └── ... (same structure as train)
  └── test/
      └── ... (same structure as train)
  ```

- 下载Minerals-36数据集（如需获取，可联系通讯作者：xhji@cugb.edu.cn）。  
- 按以下结构整理数据集：  
  ```
  Minerals-36/
  ├── train/
  │   ├── Albite/          # 每个矿物类别对应一个文件夹（共36个）
  │   │   ├── img_1.jpg
  │   │   └── ...
  │   └── ...
  ├── val/
  │   └── ...（与train结构一致）
  └── test/
      └── ...（与train结构一致）
  ```

### 2. Model Training / 模型训练
Modify the dataset path and hyperparameters (epoch, learning rate, etc.) in `train.py`, then run:  
在`train.py`中修改数据集路径及超参数（epoch、学习率等），然后运行：  
```bash
python train.py
```
- Training stages are automatically executed in sequence: FEA (1–60 epochs) → CFL (61–100 epochs) → cRT (101–110 epochs).  
- 训练阶段将自动按顺序执行：FEA（1–60轮）→ CFL（61–100轮）→ cRT（101–110轮）。

### 3. Model Evaluation / 模型评估
After training, run `evaluate.py` to calculate metrics and generate visualization results (confusion matrix, feature distribution):  
训练完成后，运行`evaluate.py`计算评估指标并生成可视化结果（混淆矩阵、特征分布）：  
```bash
python evaluate.py
```

## Experimental Results / 实验结果
On the Minerals-36 dataset, the proposed method achieves the following performance compared to the ResNet-50 baseline:  
在Minerals-36数据集上，所提方法相较于ResNet-50基线模型的性能如下：  

| Metric                | Baseline | Proposed Method | Improvement |
|-----------------------|----------|-----------------|-------------|
| Mean Accuracy (All)   | 77.27%   | 81.27%          | +4.00%      |
| Mean Accuracy (Tail)  | 67.47%   | 80.06%          | +12.59%     |
| Class-wise ASD (CwASD)| 0.150    | 0.107           | -0.043      |
| Class-wise Range (CwARange)| 0.513 | 0.378       | -0.135      |

- The maximum accuracy improvement for a single tail-class mineral reaches 24.1%.  
- 单个尾部类别矿物的精度最大提升达24.1%。

## Contact & Citation / 联系与引用
### 1. Contact Information / 联系方式
- Corresponding Author: Xiaohui Ji (xhji@cugb.edu.cn)  
- Project Lead: Keyang Hu (hu.keyang@email.cugb.edu.cn)  

- 通讯作者：季晓慧（xhji@cugb.edu.cn）  
- 项目负责人：胡科扬（hu.keyang@email.cugb.edu.cn）  

### 2. Citation / 引用格式
If you use this code or dataset in your research, please cite the following manuscript:  
如在研究中使用本代码或数据集，请引用以下论文：  
```
Hu, K., Ji, X., Yang, M., Xu, B., Lv, G., Liu, M., Zhang, Z., Zeng, S., Li, J. (2024). Long-tailed Mineral Identification based on Multi-stage Training. Computers & Geosciences, [under review].
```

## License / 许可证
This code is for **non-commercial research purposes only**. For commercial use, please contact the corresponding author for permission.  
本代码仅用于**非商业研究用途**。如需商业使用，请联系通讯作者获取许可。