import torch
import torch.nn as nn


class DWT(nn.Module):
    """
    离散小波变换 (Discrete Wavelet Transform) 层 - 基于 Haar 小波
    用于将图像无损分解为频域分量：LL (低频), LH, HL, HH (高频)
    创新点：相比于MaxPooling，DWT保留了所有像素信息，只是重排了位置，非常适合小目标。
    """

    def __init__(self):
        super().__init__()
        # 不需要学习参数，纯数学变换
        self.requires_grad = False

    def forward(self, x):
        """
        Input: x (B, C, H, W)
        Output: (B, 4*C, H/2, W/2) -> 包含 LL, LH, HL, HH
        """
        # 切片操作提取像素
        x01 = x[:, :, 0::2, :] / 2
        x02 = x[:, :, 1::2, :] / 2
        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]

        # Haar 小波计算公式
        x_LL = x1 + x2 + x3 + x4  # 低频近似 (Global Context)
        x_LH = -x1 - x2 + x3 + x4  # 垂直细节
        x_HL = -x1 + x2 - x3 + x4  # 水平细节
        x_HH = x1 - x2 - x3 + x4  # 对角细节 (通常包含噪点和微小边缘)

        # 将四个分量拼接：(B, 4C, H/2, W/2)
        return torch.cat([x_LL, x_LH, x_HL, x_HH], dim=1)


class IDWT(nn.Module):
    """
    逆离散小波变换 (Inverse DWT)
    用于在需要恢复分辨率时（如上采样）使用
    """

    def __init__(self):
        super().__init__()
        self.requires_grad = False

    def forward(self, x):
        # x shape: (B, 4C, H/2, W/2)
        r = 2
        in_batch, in_channel, in_height, in_width = x.size()
        out_batch, out_channel, out_height, out_width = in_batch, int(
            in_channel / (r ** 2)), r * in_height, r * in_width

        # 将通道拆回四个分量
        x1 = x[:, 0:out_channel, :, :] / 2
        x2 = x[:, out_channel:out_channel * 2, :, :] / 2
        x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
        x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2

        # 逆变换计算
        h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().to(x.device)

        h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
        h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
        h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
        h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

        return h


# 测试代码
if __name__ == '__main__':
    img = torch.randn(1, 3, 640, 640)
    dwt = DWT()
    out = dwt(img)
    print(f"输入尺寸: {img.shape}")
    print(f"DWT输出尺寸: {out.shape}")  # 应该是 (1, 12, 320, 320)