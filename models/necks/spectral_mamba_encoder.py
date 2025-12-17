import torch
import torch.nn as nn
import torch.nn.functional as F
from src.core import register


class SpectralGatedMambaBlock(nn.Module):
    """
    [完全体] Spectral Gated Mamba Block (SGM)

    创新点：
    1. 纯 PyTorch 实现，无需编译 mamba-ssm 库（对 Windows 极度友好）。
    2. 频域闭环：利用 FFT 在频域进行特征混合，完美呼应 Backbone 的小波变换。
    3. 线性复杂度：避免了 Transformer 的 N^2 矩阵计算，适合处理 VisDrone 的大分辨率特征图。
    """

    def __init__(self, dim, expansion_factor=2.0):
        super().__init__()
        hidden_dim = int(dim * expansion_factor)
        self.dim = dim

        # 1. 输入投影
        self.in_proj = nn.Linear(dim, hidden_dim * 2)

        # 2. 局部上下文增强 (Depthwise Conv)
        self.conv = nn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, padding=1, groups=hidden_dim * 2)

        # 3. 谱门控参数 (Spectral Gating)
        # 这是一个可学习的频域滤波器，类似于 Mamba 的 SSM 状态矩阵 A
        # 使用 Complex 参数来处理频域信息
        self.complex_weight = nn.Parameter(torch.randn(hidden_dim, 2, dtype=torch.float32) * 0.02)

        # 4. 输出投影
        self.out_proj = nn.Linear(hidden_dim, dim)
        self.act = nn.SiLU()

    def forward(self, x):
        # x shape: [B, H, W, C] (注意：这里输入通常是 channel last)
        B, H, W, C = x.shape

        # --- Phase 1: Expansion & Local Mixing ---
        x = self.in_proj(x)
        # Permute for Conv2d: [B, H, W, 2C] -> [B, 2C, H, W]
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = self.act(x)

        # Split into Value (v) and Gate (g)
        # 类似于 Mamba 的 x * sigmoid(Linear(x)) 结构
        x = x.permute(0, 2, 3, 1)  # Back to [B, H, W, 2C]
        hidden_dim = x.shape[-1] // 2
        u, gate = torch.split(x, hidden_dim, dim=-1)

        # --- Phase 2: Spectral State Space Mixing (The "Mamba" Magic) ---
        # 1. 转换到频域 (FFT)
        # rfft2 只计算一半的频率，节省计算量
        u_f = torch.fft.rfft2(u.float(), dim=(1, 2), norm='ortho')

        # 2. 频域滤波 (Spectral Gating)
        # 这里的 weight 充当了 SSM 中的全局卷积核
        weight = torch.view_as_complex(self.complex_weight)  # [C]
        # 广播乘法: [B, H, W/2+1, C] * [C]
        u_f = u_f * weight

        # 3. 转换回空域 (IFFT)
        u_out = torch.fft.irfft2(u_f, s=(H, W), dim=(1, 2), norm='ortho')

        # 4. Gating 机制 (Element-wise multiplication)
        out = u_out * gate

        # --- Phase 3: Projection ---
        out = self.out_proj(out)
        return out

@register
class SpectralMambaEncoder(nn.Module):
    """
    [完全体] VisDrone 专用：谱状态空间编码器
    替代原版 HybridEncoder，专门处理密集小目标。
    """

    def __init__(self, in_channels=[128, 256, 512], hidden_dim=256, encoder_idx=[1, 2]):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoder_idx = encoder_idx

        # 1. Channel Projection
        self.input_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim)
            ) for ch in in_channels
        ])

        # 2. Spectral Mamba Layers
        # 为每个选中的尺度构建 Mamba Block
        self.encoders = nn.ModuleDict()
        for idx in encoder_idx:
            # 使用 LayerNorm 保证训练稳定性
            self.encoders[str(idx)] = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                SpectralGatedMambaBlock(hidden_dim)
            )

        # 3. Bi-directional Fusion (CCFM - 保持原版优秀的融合结构)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0) for _ in range(len(in_channels) - 1)
        ])
        self.fpn_blocks = nn.ModuleList([
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1) for _ in range(len(in_channels) - 1)
        ])
        self.downsample_convs = nn.ModuleList([
            nn.Conv2d(hidden_dim, hidden_dim, 3, 2, 1) for _ in range(len(in_channels) - 1)
        ])
        self.pan_blocks = nn.ModuleList([
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1) for _ in range(len(in_channels) - 1)
        ])

    def forward(self, feats):
        # 1. Projection
        projs = [proj(f) for proj, f in zip(self.input_proj, feats)]

        # 2. Spectral Mamba Encoding (Core Innovation)
        for i, proj in enumerate(projs):
            if i in self.encoder_idx:
                B, C, H, W = proj.shape
                # Mamba block expects [B, H, W, C]
                src = proj.permute(0, 2, 3, 1).contiguous()

                # Apply Spectral Mamba
                src = self.encoders[str(i)](src)

                # Back to [B, C, H, W]
                projs[i] = src.permute(0, 3, 1, 2).contiguous() + proj  # Residual connection

        # 3. Fusion (Top-Down FPN)
        # 这里的实现逻辑与 RT-DETR 标准 CCFM 保持一致，确保融合稳定性
        inner_outs = [projs[-1]]
        for idx in range(len(projs) - 2, -1, -1):
            feat_high = inner_outs[0]
            feat_low = projs[idx]

            lat = self.lateral_convs[len(projs) - 2 - idx](feat_low)
            up = F.interpolate(feat_high, scale_factor=2, mode="nearest")
            inner_outs.insert(0, self.fpn_blocks[len(projs) - 2 - idx](lat + up))

        # 4. Fusion (Bottom-Up PAN)
        outs = [inner_outs[0]]
        for idx in range(len(inner_outs) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]

            down = self.downsample_convs[idx](feat_low)
            outs.append(self.pan_blocks[idx](down + feat_high))

        return outs