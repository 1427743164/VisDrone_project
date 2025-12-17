import torch
import torch.nn as nn
import torch.nn.functional as F


class HaarWaveletDownsampling(nn.Module):
    """
    [完全体] Haar Wavelet Downsampling Block

    功能：
    1. 离散小波变换 (DWT)：将空间信息无损转换为频率通道 (C -> 4C, H -> H/2, W -> W/2)
    2. 特征融合 (Fusion)：通过 1x1 卷积将 4C 维度的频率特征映射到用户指定的 out_channels
    3. 激活与归一化：包含 BN 和 SiLU，使其可以直接替代 stride=2 的 Conv 层

    优势：
    相比普通卷积下采样，它保留了所有的高频纹理细节（这对小目标至关重要）。
    """

    def __init__(self, in_channels, out_channels):
        super(HaarWaveletDownsampling, self).__init__()

        # 1. 小波变换部分 (固定参数，无梯度)
        self.register_buffer('haar_weights', self._create_haar_weights())

        # 2. 频率融合部分 (可学习参数)
        # 小波变换后通道数会变成 4倍 (LL, LH, HL, HH)
        # 我们需要用 1x1 卷积把它映射到目标 out_channels
        self.conv = nn.Conv2d(in_channels * 4, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)  # 使用 SiLU (Swish)，RT-DETR 标配

    def _create_haar_weights(self):
        """
        构造 Haar 小波卷积核
        """
        c = 0.5
        # (Out=4, In=1, K=2, K=2)
        # 分别对应: LL(低频), LH(水平高频), HL(垂直高频), HH(对角高频)
        filters = torch.tensor([
            [[[c, c], [c, c]]],  # LL
            [[[-c, -c], [c, c]]],  # LH
            [[[-c, c], [-c, c]]],  # HL
            [[[c, -c], [-c, c]]]  # HH
        ], dtype=torch.float32)
        return filters

    def forward(self, x):
        B, C, H, W = x.shape

        # --- 步骤 1: 鲁棒性 Padding ---
        # 确保输入宽高是偶数，防止下采样丢失边缘信息
        pad_h = H % 2
        pad_w = W % 2
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        # --- 步骤 2: Haar 小波变换 (离散变换) ---
        # 这里的原理是：对每个输入通道，都进行 4 种小波滤波
        # 使用 group convolution 实现通道独立处理
        # weights 形状适配: (4*C, 1, 2, 2)
        weights = self.haar_weights.repeat(C, 1, 1, 1).to(x.device)

        # stride=2 实现下采样
        x_dwt = F.conv2d(x, weights, stride=2, groups=C)
        # 此时 x_dwt 形状: (B, 4*C, H/2, W/2)

        # --- 步骤 3: 频率信息融合 (Learnable) ---
        # 将分离出来的频率层融合，学习哪些频率对检测更重要
        out = self.conv(x_dwt)
        out = self.bn(out)
        out = self.act(out)

        return out


if __name__ == "__main__":
    # --- 单元测试 ---
    print("Testing Complete HaarWaveletDownsampling...")
    # 模拟 RT-DETR Backbone 第一层的常见输入
    # 输入: Batch=2, Channel=3 (RGB), H=640, W=640
    dummy_input = torch.randn(2, 3, 640, 640)

    # 我们希望替代原本的 Conv(3, 64, stride=2)
    # 所以初始化参数为 (3, 64)
    model = HaarWaveletDownsampling(in_channels=3, out_channels=64)

    output = model(dummy_input)

    print(f"Input: {dummy_input.shape}")
    print(f"Output: {output.shape}")

    # 验证尺寸是否减半
    assert output.shape[2] == 320 and output.shape[3] == 320
    # 验证通道数是否正确映射
    assert output.shape[1] == 64
    print("Test Passed! This module is ready for backbone integration.")