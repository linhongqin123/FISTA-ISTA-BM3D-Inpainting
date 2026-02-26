import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, metrics
import pywt
import time
import os

def load_image(path, gray=True):
    """读取图像，归一化到[0,1]，转为灰度（可选）"""
    img = io.imread(path)
    if gray and img.ndim == 3:
        img = color.rgb2gray(img)
    img = img.astype(np.float64)
    if img.max() > 1.0:
        img /= 255.0
    return img

def create_mask(shape, keep_ratio=0.5, seed=42):
    """生成随机掩码：1表示保留，0表示丢失"""
    np.random.seed(seed)
    mask = np.random.rand(*shape) < keep_ratio
    return mask.astype(np.float64)

def add_mask(img, mask):
    """应用掩码，得到受损图像"""
    return img * mask

def psnr(img_true, img_test):
    """计算PSNR，假设图像在[0,1]范围"""
    return metrics.peak_signal_noise_ratio(img_true, img_test, data_range=1.0)

def ssim(img_true, img_test):
    """计算SSIM"""
    return metrics.structural_similarity(img_true, img_test, data_range=1.0)

def plot_images(images, titles, filename=None, figsize=None):
    """绘制多张图像对比"""
    n = len(images)
    if figsize is None:
        figsize = (4*n, 4)
    plt.figure(figsize=figsize)
    for i in range(n):
        plt.subplot(1, n, i+1)
        # 判断图像维度以选择colormap
        if images[i].ndim == 2:
            plt.imshow(images[i], cmap='gray', vmin=0, vmax=1)
        else:
            plt.imshow(images[i], vmin=0, vmax=1)
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()

def ensure_dir(directory):
    """确保目录存在"""
    if not os.path.exists(directory):
        os.makedirs(directory)