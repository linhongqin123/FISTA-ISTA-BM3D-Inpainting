import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import load_image, create_mask, add_mask, psnr, ssim, plot_images, ensure_dir
from fista_l1 import FISTA_L1
from fista_tv import FISTA_TV
from bm3d_inpaint import bm3d_inpaint

# 参数设置
data_dir = './data/Set14'          # 图像文件夹
image_files = ['ppt3.png', 'baboon.png', 'barbara.png', 'bridge.png', 'coastguard.png']  # 根据实际文件名修改
keep_ratio = 0.5                   # 保留像素比例
lam_l1 = 0.1                        # L1正则化参数（需调优）
lam_tv = 0.05                       # TV正则化参数（需调优）
max_iter = 200                      # 最大迭代次数
tol = 1e-5                          # 收敛容忍度

# 创建输出文件夹
ensure_dir('./results')
ensure_dir('./results/images')
ensure_dir('./results/convergence')

# 存储结果的列表
results = []

for img_file in image_files:
    print(f"\nProcessing {img_file}...")
    # 读取图像
    img_path = os.path.join(data_dir, img_file)
    if not os.path.exists(img_path):
        print(f"Warning: {img_path} not found, skipping.")
        continue
    img = load_image(img_path, gray=True)
    H, W = img.shape

    # 生成掩码并创建受损图像
    mask = create_mask((H, W), keep_ratio=keep_ratio, seed=42)
    y = add_mask(img, mask)

    # 初始化算法
    fista_l1_solver = FISTA_L1(wavelet='db1', level=4)
    fista_tv_solver = FISTA_TV()

    # 定义算法列表
    algos = {
        'ISTA-L1': (lambda: fista_l1_solver.solve(y, mask, lam_l1, max_iter, tol, fista=False, return_obj=True)),
        'FISTA-L1': (lambda: fista_l1_solver.solve(y, mask, lam_l1, max_iter, tol, fista=True, return_obj=True)),
        'ISTA-TV': (lambda: fista_tv_solver.solve(y, mask, lam_tv, max_iter, tol, fista=False, return_obj=True)),
        'FISTA-TV': (lambda: fista_tv_solver.solve(y, mask, lam_tv, max_iter, tol, fista=True, return_obj=True)),
        'BM3D': (lambda: (bm3d_inpaint(y, mask, sigma_psd=0.01), None))
    }

    # 创建本图像的结果行
    row = {'Image': img_file}

    # 临时保存恢复图像和obj_vals用于绘图
    restored_imgs = {}
    obj_vals_dict = {}

    for name, func in algos.items():
        print(f"  Running {name}...")
        start = time.time()
        restored, obj_vals = func()
        elapsed = time.time() - start
        psnr_val = psnr(img, restored)
        ssim_val = ssim(img, restored)

        # 保存指标
        row[f'{name}_PSNR'] = psnr_val
        row[f'{name}_SSIM'] = ssim_val
        row[f'{name}_Time'] = elapsed

        # 保存图像和obj_vals用于后续绘图
        restored_imgs[name] = restored
        if obj_vals is not None:
            obj_vals_dict[name] = obj_vals

        print(f"    PSNR: {psnr_val:.4f}, SSIM: {ssim_val:.4f}, Time: {elapsed:.2f}s")

    # 绘制图像对比图
    images_to_plot = [img, y, 
                      restored_imgs['ISTA-L1'], restored_imgs['FISTA-L1'],
                      restored_imgs['ISTA-TV'], restored_imgs['FISTA-TV'],
                      restored_imgs['BM3D']]
    titles = ['Original', 'Damaged', 
              f'ISTA-L1\nPSNR={row["ISTA-L1_PSNR"]:.2f}', f'FISTA-L1\nPSNR={row["FISTA-L1_PSNR"]:.2f}',
              f'ISTA-TV\nPSNR={row["ISTA-TV_PSNR"]:.2f}', f'FISTA-TV\nPSNR={row["FISTA-TV_PSNR"]:.2f}',
              f'BM3D\nPSNR={row["BM3D_PSNR"]:.2f}']
    plot_images(images_to_plot, titles, 
                filename=f'./results/images/{img_file}_comparison.png', 
                figsize=(16, 8))

        # 绘制收敛曲线（对L1和TV分别）
    plt.figure()
    if 'ISTA-L1' in obj_vals_dict:
        plt.plot(obj_vals_dict['ISTA-L1'], label='ISTA-L1')
    if 'FISTA-L1' in obj_vals_dict:
        plt.plot(obj_vals_dict['FISTA-L1'], label='FISTA-L1')
    plt.xlabel('Iteration')
    plt.ylabel('Objective Value')
    plt.title(f'{img_file} - L1 Convergence')
    plt.legend()
    plt.savefig(f'./results/convergence/{img_file}_L1_convergence.png', dpi=150)
    plt.close()

    plt.figure()
    if 'ISTA-TV' in obj_vals_dict:
        plt.plot(obj_vals_dict['ISTA-TV'], label='ISTA-TV')
    if 'FISTA-TV' in obj_vals_dict:
        plt.plot(obj_vals_dict['FISTA-TV'], label='FISTA-TV')
    plt.xlabel('Iteration')
    plt.ylabel('Objective Value')
    plt.title(f'{img_file} - TV Convergence')
    plt.legend()
    plt.savefig(f'./results/convergence/{img_file}_TV_convergence.png', dpi=150)
    plt.close()

    # ========== 在这里插入合并收敛图的代码 ==========
    plt.figure(figsize=(10, 6))
    if 'ISTA-L1' in obj_vals_dict:
        plt.plot(obj_vals_dict['ISTA-L1'], 'b-', label='ISTA-L1')
    if 'FISTA-L1' in obj_vals_dict:
        plt.plot(obj_vals_dict['FISTA-L1'], 'b--', label='FISTA-L1')
    if 'ISTA-TV' in obj_vals_dict:
        plt.plot(obj_vals_dict['ISTA-TV'], 'r-', label='ISTA-TV')
    if 'FISTA-TV' in obj_vals_dict:
        plt.plot(obj_vals_dict['FISTA-TV'], 'r--', label='FISTA-TV')
    plt.xlabel('Iteration')
    plt.ylabel('Objective Value')
    plt.title(f'{img_file} - Convergence Comparison (All Algorithms)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'./results/convergence/{img_file}_all_convergence.png', dpi=150)
    plt.close()
    # =============================================

    results.append(row)

# 转换为DataFrame并保存
df = pd.DataFrame(results)
print("\nFinal Results:")
print(df.to_string(index=False))

# 保存为CSV
df.to_csv('./results/results.csv', index=False)

# 也可以生成LaTeX表格（用于报告）
with open('./results/results_table.tex', 'w') as f:
    f.write(df.to_latex(index=False, float_format="%.4f"))

print("\nAll done! Results saved in ./results/")