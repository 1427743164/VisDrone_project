import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleHybridEncoder(nn.Module):
    """
    [完全体] Multi-Scale Hybrid Encoder for Tiny Object Detection

    创新点：
    1. Multi-Scale AIFI: 原版 RT-DETR 只对最后一层(S5)做 Intra-scale 交互。
       本模块支持对 [S4, S5] 甚至 [S3, S4, S5] 进行 Transformer 编码，
       显著增强了小目标（存在于 S3/S4 层）的语义上下文理解能力。
    2. 鲁棒的 CSP-Fusion: 采用 CSPNet 思想进行跨尺度特征融合 (CCFM)。
    """

    def __init__(self,
                 in_channels=[128, 256, 512],
                 hidden_dim=256,
                 expansion=1.0,
                 depth_mult=1.0,
                 # 关键参数: 决定哪些层通过 Transformer Encoder (0=S3, 1=S4, 2=S5)
                 # 对于 VisDrone，建议开启 [1, 2] 即 S4 和 S5
                 encoder_layer_indices=[2],
                 nhead=8,
                 dim_feedforward=1024,
                 dropout=0.0,
                 enc_act='gelu',
                 use_encoder_idx=[2],  # 兼容旧参数名
                 num_encoder_layers=1,
                 pe_temperature=10000,
                 eval_spatial_size=None):
        super(MultiScaleHybridEncoder, self).__init__()

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        # 优先使用新参数名，兼容旧配置
        self.encoder_layer_indices = encoder_layer_indices if encoder_layer_indices else use_encoder_idx

        # --- 1. Channel Projection (维度对齐) ---
        # 将 Backbone 输出的不同通道数统一映射到 hidden_dim (通常 256)
        self.input_proj = nn.ModuleList()
        for ch in in_channels:
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2d(ch, hidden_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(hidden_dim)
                )
            )

        # --- 2. Transformer Encoder (AIFI) ---
        # 针对被选中的层构建 Encoder
        # 这是一个 ModuleDict，键是层的索引 (str)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=enc_act,
            batch_first=True
        )

        self.encoders = nn.ModuleDict()
        for idx in self.encoder_layer_indices:
            # 这里的 num_layers 通常设为 1 以节省显存，小目标 1 层足矣
            self.encoders[str(idx)] = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        # --- 3. CCFM (Cross-Scale Feature-Fusion Module) ---
        # 融合路径: 包含了自顶向下和自底向上的路径
        self.fusion_blocks = nn.ModuleList()
        # 这里使用简单的 RepConv 或 Conv 块进行融合，为保证通用性，使用标准 ConvBNReLU
        for _ in range(len(in_channels) - 1):
            self.fusion_blocks.append(ConvNormLayer(hidden_dim * 2, hidden_dim, 1, 1))

        # 额外的融合层，用于自底向上的路径
        self.pan_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1):
            self.pan_blocks.append(ConvNormLayer(hidden_dim * 2, hidden_dim, 1, 1))

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.0, device=None):
        """生成 2D 绝对位置编码"""
        grid_w = torch.arange(int(w), dtype=torch.float32, device=device)
        grid_h = torch.arange(int(h), dtype=torch.float32, device=device)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='xy')
        assert embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32, device=device) / pos_dim
        omega = 1. / (temperature ** omega)

        out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
        out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])

        pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)[None, :, :]
        return pos_emb

    def forward(self, feats):
        """
        feats: List[Tensor], shape usually [B, C, H, W]
        """
        # 1. 投影到统一维度
        projs = [proj(feat) for proj, feat in zip(self.input_proj, feats)]

        # 2. Multi-Scale AIFI (核心魔改部分)
        # 对选定索引的特征层应用 Transformer Encoder
        for i, proj in enumerate(projs):
            if i in self.encoder_layer_indices:
                B, C, H, W = proj.shape
                # Flatten: [B, C, H, W] -> [B, H*W, C]
                src = proj.flatten(2).permute(0, 2, 1)

                # 生成位置编码
                pos_embed = self.build_2d_sincos_position_embedding(W, H, C, device=src.device)

                # Transformer 编码
                encoder = self.encoders[str(i)]
                src = encoder(src + pos_embed)

                # 还原形状
                projs[i] = src.permute(0, 2, 1).reshape(B, C, H, W)

        # 3. CCFM Fusion (类似于 PANet/FPN)
        # 自顶向下融合 (Top-Down)
        # projs 顺序通常是 [S3, S4, S5] (从小尺度到大尺度)
        inner_outs = [projs[-1]]  # 先放入 S5
        for idx in range(len(projs) - 2, -1, -1):  # 遍历 S4, S3
            feat_high = inner_outs[0]  # 上一层的高级特征
            feat_low = projs[idx]  # 当前层的低级特征

            # 上采样
            upsample_feat = F.interpolate(feat_high, scale_factor=2.0, mode='nearest')

            # 拼接 + 卷积融合
            concat_feat = torch.cat([upsample_feat, feat_low], dim=1)
            fused_feat = self.fusion_blocks[len(projs) - 2 - idx](concat_feat)

            inner_outs.insert(0, fused_feat)  # 插入到列表前面

        # 自底向上融合 (Bottom-Up)
        outs = [inner_outs[0]]  # 先放入 S3
        for idx in range(len(inner_outs) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]

            # 下采样 (这里简化用 Conv stride=2，实际可以用更复杂的)
            downsample_feat = F.avg_pool2d(feat_low, kernel_size=2, stride=2)

            concat_feat = torch.cat([downsample_feat, feat_high], dim=1)
            fused_feat = self.pan_blocks[idx](concat_feat)
            outs.append(fused_feat)

        return outs


class ConvNormLayer(nn.Module):
    """辅助用的卷积块"""

    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act='relu'):
        super().__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size, stride, padding, bias=bias)
        self.bn = nn.BatchNorm2d(ch_out)
        self.act = nn.SiLU() if act == 'silu' else nn.ReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


if __name__ == "__main__":
    # --- 单元测试 ---
    print("Testing MultiScaleHybridEncoder...")
    # 模拟 Backbone 输出 (S3, S4, S5)
    # 通道数对应: ResNet50 -> [512, 1024, 2048]
    # 但如果是 W-RT-DETR 且用了通道压缩，可能是 [128, 256, 512]
    # 这里我们模拟 W-RT-DETR 通常的配置
    inputs = [
        torch.randn(1, 128, 80, 80),  # S3 (High Res)
        torch.randn(1, 256, 40, 40),  # S4 (Medium)
        torch.randn(1, 512, 20, 20)  # S5 (Low Res)
    ]

    # 关键测试点: 开启 S4 和 S5 的 Encoder (indices=[1, 2])
    encoder = MultiScaleHybridEncoder(
        in_channels=[128, 256, 512],
        hidden_dim=256,
        encoder_layer_indices=[1, 2]
    )

    outputs = encoder(inputs)

    print("Output shapes:")
    for i, o in enumerate(outputs):
        print(f"  Level {i}: {o.shape}")
        # 验证输出维度是否统一为 256
        assert o.shape[1] == 256

    print("MultiScaleHybridEncoder Test Passed!")