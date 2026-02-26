# 图像修复：FISTA vs. ISTA vs. BM3D 在 Set14 数据集上的比较研究

本仓库包含图像修复算法的比较研究代码和报告。我们实现了带两种不同正则化项（小波域 ℓ₁ 范数和全变分 TV）的 **FISTA**（快速迭代收缩阈值算法）和 **ISTA**，并与当前最先进的 **BM3D** 方法进行对比。实验在 **Set14** 数据集中的五幅图像上进行，随机丢失 50% 的像素。完整的报告（LaTeX 源码和 PDF）也包含在内。

## 仓库结构

```
.
├── README.md
├── utils.py                       # 辅助函数（图像读写、掩码生成、指标计算、绘图等）
├── fista_l1.py                    # ISTA/FISTA 带 ℓ₁ 正则化（小波域）
├── fista_tv.py                     # ISTA/FISTA 带全变分正则化
├── bm3d_inpaint.py                 # 基于 BM3D 的修复（预填充 + 去噪）
├── main.py                          # 主实验脚本：运行所有算法，收集结果并绘图
├── data/                           # 存放 Set14 图像
│   ├── ppt3.png
│   ├── baboon.png
│   ├── barbara.png
│   ├── bridge.png
│   └── coastguard.png
├── results/    # 输出图像和表格                      				           # 报告中使用的图片
              # 保存的收敛曲线图
             # LaTeX 报告源码和编译好的 PDF
```

##  安装方法

克隆本仓库并安装所需的 Python 包。建议使用虚拟环境（例如 `conda` 或 `venv`）。

```bash
# 克隆仓库
git clone https://github.com/yourusername/FISTA-ISTA-BM3D-Inpainting.git
cd FISTA-ISTA-BM3D-Inpainting

# （可选）创建并激活虚拟环境
conda create -n inpainting python=3.9
conda activate inpainting

# 安装依赖
pip install numpy scipy matplotlib scikit-image opencv-python pillow bm3d PyWavelets
```

### 依赖说明

- `numpy` – 数值计算  
- `scipy` – 信号处理工具  
- `matplotlib` – 绘制收敛曲线和视觉结果  
- `scikit-image` – 图像读写、PSNR/SSIM 计算  
- `opencv-python` – 用于 BM3D 流程中的 Telea 修复  
- `pillow` – 图像处理（可选，常与其他包一起使用）  
- `bm3d` – 官方 BM3D Python 包（通过 `pip install bm3d` 安装）  
- `PyWavelets` – 小波变换（用于 ℓ₁ 正则化）  

> **注意**：如果 `bm3d` 包安装出现问题，可以尝试从源码安装，或者使用其他备选方案。导入时使用 `import bm3d`。

## 数据集

**Set14** 数据集可以从https://www.kaggle.com/datasets/ll01dm/set-5-14-super-resolution-dataset/data 或其他可靠来源下载。将选中的图像（`ppt3`、`baboon`、`barbara`、`bridge`、`coastguard` 的灰度版本）放入 `data/` 文件夹。如果想测试更多图像，只需添加它们并更新 `main.py` 中的列表。

##  使用方法

运行主实验脚本：

```bash
python main.py
```

该脚本将：

- 从 `data/` 加载图像并生成随机掩码（50% 像素缺失）。
- 运行全部五种算法：
  - ISTA‑L1  
  - FISTA‑L1  
  - ISTA‑TV  
  - FISTA‑TV  
  - BM3D  
- 计算每个结果的 PSNR 和 SSIM。
- 将收敛曲线图保存到 `convergence/`，视觉对比图保存到 `results/`。
- 打印结果汇总表（类似于报告中的表格）。

所有参数（正则化权重、迭代次数、容差等）均在 `main.py` 开头定义，可根据需要调整。

## 结果摘要

定量结果汇总如下（摘自报告）：

| 图像           | ISTA‑L1 (PSNR/SSIM) | FISTA‑L1 (PSNR/SSIM) | ISTA‑TV (PSNR/SSIM) | FISTA‑TV (PSNR/SSIM) | BM3D (PSNR/SSIM) |
| -------------- | ------------------- | -------------------- | ------------------- | -------------------- | ---------------- |
| ppt3.png       | 20.78 / 0.826       | 20.82 / 0.827        | 25.61 / 0.928       | 25.66 / 0.928        | 23.99 / 0.913    |
| baboon.png     | 20.65 / 0.451       | 20.66 / 0.451        | 22.01 / 0.546       | 22.01 / 0.546        | 23.41 / 0.781    |
| barbara.png    | 23.14 / 0.633       | 23.14 / 0.634        | 25.08 / 0.739       | 25.08 / 0.739        | 26.65 / 0.874    |
| bridge.png     | 22.04 / 0.497       | 22.05 / 0.497        | 24.94 / 0.688       | 24.42 / 0.629        | 25.87 / 0.827    |
| coastguard.png | 23.82 / 0.481       | 23.84 / 0.481        | 25.54 / 0.547       | 25.68 / 0.566        | 27.10 / 0.818    |

**主要发现**：

- 对于自然图像，全变分（TV）正则化始终优于 ℓ₁ 小波稀疏性。
- FISTA 的收敛速度远快于 ISTA，尤其在 TV 情形下。
- BM3D 在纹理丰富的图像（baboon、barbara、bridge、coastguard）上表现最佳，这得益于其非局部自相似性建模。

详细讨论请参阅完整报告 `report/report.pdf`。

## 报告

报告 LaTeX 源码位于 `report/` 文件夹。您可以自行编译（需要 `biblatex` 和较新的 LaTeX 发行版），或直接阅读预编译的 `report.pdf`。
