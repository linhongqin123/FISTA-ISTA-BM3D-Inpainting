import numpy as np
from skimage.restoration import denoise_tv_chambolle

def total_variation(img):
    """
    手动计算各向同性全变分 (Total Variation)
    TV = sum sqrt( (dx)^2 + (dy)^2 )
    """
    # 计算水平方向差分（右侧减当前，最后一个元素差分为0）
    dx = np.diff(img, axis=1, append=0)
    # 计算垂直方向差分（下侧减当前，最后一行差分为0）
    dy = np.diff(img, axis=0, append=0)
    return np.sum(np.sqrt(dx**2 + dy**2))

def prox_tv(x, lam, max_iter=20):
    """TV近端映射（近似），使用Chambolle算法"""
    return denoise_tv_chambolle(x, weight=lam, eps=1e-4, max_num_iter=max_iter)

class FISTA_TV:
    def solve(self, y, mask, lam, max_iter=200, tol=1e-5, fista=True, return_obj=False):
        """
        y: 观测图像 (H,W)
        mask: 掩码 (H,W)
        lam: 正则化参数
        return_obj: 是否返回目标函数值
        """
        x = y.copy()
        y_aux = x.copy() if fista else None
        t = 1.0
        obj_vals = []

        for i in range(max_iter):
            # 梯度：mask*(x - y)
            grad = mask * (x - y)

            if fista:
                # 在辅助变量上梯度下降
                x_temp = y_aux - grad
                # 近端映射：TV去噪
                x_new = prox_tv(x_temp, lam)
                # 动量更新
                t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
                y_aux = x_new + ((t - 1) / t_new) * (x_new - x)
                t = t_new
                x = x_new
            else:
                # ISTA
                x = prox_tv(x - grad, lam)

            if return_obj:
                # 计算目标函数：数据项 + TV项
                data_term = 0.5 * np.sum((mask * (x - y))**2)
                tv_term = lam * total_variation(x)
                obj = data_term + tv_term
                obj_vals.append(obj)
                if i > 0 and abs(obj_vals[-1] - obj_vals[-2]) < tol:
                    break
            elif i > 0 and np.linalg.norm(x - x_prev) < tol:
                break

            x_prev = x.copy()

        if return_obj:
            return x, obj_vals
        else:
            return x