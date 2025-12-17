import torch
import torch.nn as nn


class NWDLoss(nn.Module):
    """
    Normalized Wasserstein Distance (NWD) Loss for Tiny Object Detection.
    论文参考: https://arxiv.org/abs/2110.13389
    """

    def __init__(self, constant=12.5, eps=1e-7, reduction='mean'):
        """
        Args:
            constant (float): NWD公式中的常数C，通常与数据集的目标平均尺寸有关。
                              对于VisDrone等小目标数据集，建议 12.0 - 12.5。
            eps (float): 防止除零的极小值。
            reduction (str): 'mean', 'sum', or 'none'.
        """
        super(NWDLoss, self).__init__()
        self.constant = constant
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred, target):
        """
        Args:
            pred (Tensor): 预测框，形状 (N, 4)，格式通常为 [cx, cy, w, h]
            target (Tensor): 真实框，形状 (N, 4)，格式通常为 [cx, cy, w, h]

        注意：NWD 对尺度敏感。如果输入是归一化(0-1)的坐标，建议先还原回绝对像素坐标，
              或者调整 constant 参数使其适配 0-1 的范围。
        """
        # 确保输入类型一致
        if pred.dim() != 2 or target.dim() != 2:
            raise ValueError(f"Expected input shape (N, 4), got {pred.shape} and {target.shape}")

        # 解析坐标 (假设输入是 cx, cy, w, h)
        p_cx, p_cy, p_w, p_h = pred.unbind(-1)
        t_cx, t_cy, t_w, t_h = target.unbind(-1)

        # 1. 计算 Wasserstein 距离 (W2)
        # 公式: W2^2 = ||m1-m2||^2 + ||Sigma1 - Sigma2||_F^2
        # 对于水平框高斯分布，简化为如下形式：

        w2_square = (p_cx - t_cx).pow(2) + \
                    (p_cy - t_cy).pow(2) + \
                    ((p_w - t_w) / 2.0).pow(2) + \
                    ((p_h - t_h) / 2.0).pow(2)

        # 2. 计算 NWD
        # NWD = exp( - sqrt(W2^2) / C )
        # 添加 eps 防止开根号时梯度为 NaN (当完全重合时)
        w2 = torch.sqrt(torch.clamp(w2_square, min=self.eps))
        nwd = torch.exp(-w2 / self.constant)

        # 3. 计算 Loss (1 - NWD)
        loss = 1.0 - nwd

        # 4. Reduction (归约)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss