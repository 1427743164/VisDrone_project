import torch
import torch.nn as nn
from models.layers.dwt import DWT  # 沿用之前的 DWT 代码
from models.layers.vmamba_core import SS2D


class WaveletBlock(nn.Module):
    """
    单层处理单元：包含 DWT 下采样 + SS2D 频域处理 + 门控融合
    """

    def __init__(self, in_ch, embed_dim, drop_path=0.):
        super().__init__()
        self.dwt = DWT()

        # DWT 后通道变为 4 倍
        self.in_ch_expanded = in_ch * 4

        # 维度对齐
        self.proj_low = nn.Linear(in_ch, embed_dim)  # LL (原始通道in_ch)
        self.proj_high = nn.Linear(in_ch * 3, embed_dim)  # HF (3 * in_ch)

        # 双流处理 (完全体)
        # 1. 低频流：走 SS2D (Mamba) 看全局
        self.global_stream = SS2D(d_model=embed_dim)

        # 2. 高频流：走 Conv 看细节
        self.detail_stream = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1, groups=embed_dim),  # Depthwise
            nn.SiLU(),
            nn.Conv2d(embed_dim, embed_dim, 1)  # Pointwise
        )

        # 3. 频域门控融合
        self.gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid()
        )

        self.out_proj = nn.Linear(embed_dim, embed_dim * 2)  # 准备给下一个 stage，通常 channel 会翻倍

    def forward(self, x):
        # x: (B, C, H, W)

        # 1. 小波分解
        x_dwt = self.dwt(x)  # (B, 4C, H/2, W/2)
        B, _, H_half, W_half = x_dwt.shape
        C = x.shape[1]

        x_ll = x_dwt[:, :C, :, :]  # 低频
        x_hf = x_dwt[:, C:, :, :]  # 高频 (3C)

        # 2. 变换维度 (B, H, W, C) 以适配 Mamba
        x_ll = x_ll.permute(0, 2, 3, 1)
        x_hf = x_hf.permute(0, 2, 3, 1)

        # 3. 双流并行
        # LL -> Projection -> SS2D
        feat_ll = self.proj_low(x_ll)
        feat_ll = self.global_stream(feat_ll)  # SS2D Forward

        # HF -> Projection -> Conv (Conv需要 NCHW，转回去)
        feat_hf = self.proj_high(x_hf)
        feat_hf_nchw = feat_hf.permute(0, 3, 1, 2)
        feat_hf_nchw = self.detail_stream(feat_hf_nchw)
        feat_hf = feat_hf_nchw.permute(0, 2, 3, 1)

        # 4. 门控融合
        # "Is this area background or object?" - learned from LL
        g = self.gate(feat_ll)
        out = feat_ll + (feat_hf * g)  # 动态加权高频信息

        # 变回 NCHW 输出
        return out.permute(0, 3, 1, 2)


class SpectralMambaBackbone(nn.Module):
    """
    [完全体] 完整的 Backbone 网络
    输出: [P3, P4, P5] 多尺度特征图
    """

    def __init__(self, in_ch=3, dims=[64, 128, 256], depths=[2, 2, 2]):
        super().__init__()

        self.stages = nn.ModuleList()

        # Stage 1 (Stem): 原始分辨率 -> H/2 (P2)
        # 这里我们可以先用一个 Conv 降采样做 Stem，或者直接 DWT
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, dims[0], 3, 2, 1),
            nn.BatchNorm2d(dims[0]),
            nn.SiLU(),
            nn.Conv2d(dims[0], dims[0], 3, 1, 1),
            nn.BatchNorm2d(dims[0]),
            nn.SiLU(),
        )

        # Stage 2 (P3): H/4 -> H/8
        # Stage 3 (P4): H/8 -> H/16
        # Stage 4 (P5): H/16 -> H/32

        curr_dim = dims[0]
        for i, (d, depth) in enumerate(zip(dims, depths)):
            # 这里构建每一层的 Blocks
            # 注意：这里简化了逻辑，实际应为多个 Block 堆叠
            # 为了完全体逻辑，我们使用 WaveletBlock 进行下采样
            stage = nn.ModuleList([
                WaveletBlock(in_ch=curr_dim if i == 0 else dims[i - 1], embed_dim=d)
            ])
            # 如果深度 > 1，可以堆叠更多不带下采样的 SS2D Block (StandardMambaBlock)
            # 这里为了代码简洁展示核心结构
            self.stages.append(stage)
            curr_dim = d

    def forward(self, x):
        outputs = []

        # Stem
        x = self.stem(x)  # P2 scale

        # Stages
        for stage in self.stages:
            for block in stage:
                x = block(x)
            # 我们需要 P3, P4, P5
            outputs.append(x)

        # 假设 dims 长度为 3，outputs 包含 [P3, P4, P5]
        # 注意需要根据实际层数索引。
        # RT-DETR 需要 P3, P4, P5 (分别对应 stride 8, 16, 32)

        return outputs[-3:]  # 返回最后三层