import numpy as np
import pywt

def soft_threshold(x, thresh):
    """软阈值算子"""
    return np.sign(x) * np.maximum(np.abs(x) - thresh, 0)

class FISTA_L1:
    def __init__(self, wavelet='db1', level=4):
        self.wavelet = wavelet
        self.level = level
        self.coeffs_shape = None

    def _forward(self, img):
        """小波正变换，返回系数数组（展平）"""
        coeffs = pywt.wavedec2(img, self.wavelet, level=self.level)
        arr, self.coeffs_shape = pywt.coeffs_to_array(coeffs)
        return arr

    def _backward(self, arr):
        """小波逆变换，从系数数组恢复图像"""
        coeffs = pywt.array_to_coeffs(arr, self.coeffs_shape, output_format='wavedec2')
        return pywt.waverec2(coeffs, self.wavelet)

    def solve(self, y, mask, lam, max_iter=200, tol=1e-5, fista=True, return_obj=False):
        """
        y: 观测图像 (H,W)
        mask: 掩码 (H,W)
        lam: 正则化参数
        fista: True使用FISTA，False使用ISTA
        return_obj: 是否返回目标函数值序列
        返回：恢复图像，以及可选的目标函数值
        """
        H, W = y.shape
        # 初始化
        x_img = y.copy()
        x_coeff = self._forward(x_img)
        y_coeff = x_coeff.copy() if fista else None
        t = 1.0

        obj_vals = []  # 始终初始化

        for i in range(max_iter):
            # 图像域的梯度：mask * (x_img - y)
            grad_img = mask * (x_img - y)
            # 变换到小波域
            grad_coeff = self._forward(grad_img)

            step = 1.0  # 固定步长

            if fista:
                # FISTA
                y_coeff_new = y_coeff - step * grad_coeff
                x_coeff_new = soft_threshold(y_coeff_new, lam)
                # 动量更新
                t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
                y_coeff = x_coeff_new + ((t - 1) / t_new) * (x_coeff_new - x_coeff)
                t = t_new
                x_coeff = x_coeff_new
            else:
                # ISTA
                x_coeff = soft_threshold(x_coeff - step * grad_coeff, lam)

            # 更新图像
            x_img = self._backward(x_coeff)

            if x_img.shape != y.shape:
                # 裁剪或填充到与 y 相同尺寸
                if x_img.shape[0] >= y.shape[0] and x_img.shape[1] >= y.shape[1]:
                    x_img = x_img[:y.shape[0], :y.shape[1]]
                else:
                    padded = np.zeros(y.shape)
                    padded[:x_img.shape[0], :x_img.shape[1]] = x_img
                    x_img = padded

            # 如果需要返回目标函数值，则计算并记录
            if return_obj:
                data_term = np.sum((mask * (x_img - y))**2)
                reg_term = np.sum(np.abs(x_coeff))
                obj = 0.5 * data_term.item() + lam * reg_term.item()
                obj_vals.append(obj)

                # 检查收敛
                if i > 0 and abs(obj_vals[-1] - obj_vals[-2]) < tol:
                    break

        if return_obj:
            return x_img, obj_vals
        else:
            return x_img