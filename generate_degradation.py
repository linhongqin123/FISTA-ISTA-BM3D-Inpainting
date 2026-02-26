import os
import numpy as np
import matplotlib.pyplot as plt
from utils import load_image, create_mask, add_mask, plot_images

# 设置路径
data_dir = './data/Set14'
img_file = 'ppt3.png'
img_path = os.path.join(data_dir, img_file)

# 加载图像
img = load_image(img_path, gray=True)

# 生成掩码（与实验相同的 seed=42，keep_ratio=0.5）
mask = create_mask(img.shape, keep_ratio=0.5, seed=42)
damaged = add_mask(img, mask)

# 绘制对比图
plot_images([img, damaged], 
            ['Original', 'Damaged (50% missing)'], 
            filename='./results/ppt3_degradation.png', 
            figsize=(8, 4))

print("De gradation image saved to ./results/ppt3_degradation.png")