import numpy as np
import cv2
import bm3d

def bm3d_inpaint(y, mask, sigma_psd=0.01):
    """
    使用BM3D进行图像修复
    y: 受损图像 (H,W)，值域[0,1]
    mask: 掩码 (1保留，0丢失)
    sigma_psd: BM3D去噪的噪声标准差
    """
    # 将图像转为uint8范围[0,255]
    y_uint8 = (y * 255).astype(np.uint8)
    mask_uint8 = (1 - mask).astype(np.uint8) * 255  # 缺失区域为255

    # OpenCV快速行进法修复
    inpainted = cv2.inpaint(y_uint8, mask_uint8, 3, cv2.INPAINT_TELEA)
    inpainted = inpainted.astype(np.float64) / 255.0

    # BM3D去噪（尝试两种常见接口）
    try:
        # 尝试标准接口
        denoised = bm3d.bm3d(inpainted, sigma_psd=sigma_psd)
    except AttributeError:
        # 如果失败，尝试另一种接口
        denoised = bm3d.bm3d(inpainted, sigma_psd=sigma_psd, stage_arg=bm3d.BM3DStages.ALL_STAGES)

    # 强制保留已知像素
    result = y * mask + denoised * (1 - mask)
    return np.clip(result, 0, 1)