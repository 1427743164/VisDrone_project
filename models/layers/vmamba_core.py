import torch
import torch.nn as nn
from mamba_ssm import Mamba


class SS2D(nn.Module):
    """
    [完全体逻辑] 2D Selective Scan Mechanism (SS2D)
    原理：为了让 1D Mamba 理解 2D 图像，我们必须把图像展开成 4 个方向的序列：
    1. 左上 -> 右下 (原始)
    2. 右下 -> 左上 (翻转)
    3. 右上 -> 左下 (转置)
    4. 左下 -> 右上 (转置+翻转)

    这保证了像素点能感知到全图任何方向的上下文信息。
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.):
        super().__init__()
        self.d_model = d_model
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)

        # 输入投影
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)

        # 核心 Mamba (我们需要共享权重还是独立权重？为了参数效率通常共享核心SSM参数)
        # 这里我们实例化一个标准 Mamba，但会手动控制它的扫描流
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
        )
        self.activation = nn.SiLU()

        # 核心 SSM 算子 (调用 CUDA 优化版)
        self.ssm = Mamba(
            d_model=self.d_inner,  # 这里的 d_model 对应内部维度
            d_state=d_state,
            d_conv=d_conv,
            expand=1  # 已经在外部 expand 过了
        )

        # 输出投影
        self.out_proj = nn.Linear(self.d_inner, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: (B, H, W, C)
        """
        B, H, W, C = x.shape
        input_x = x

        # 1. 线性投影与激活
        x = self.in_proj(x)  # (B, H, W, 2*d_inner)
        (x, z) = x.chunk(2, dim=-1)  # x: signal, z: gate

        # 2. 4向展开 (Scan Expanding)
        # 这是一个显存密集操作，但为了逻辑完整性必须这么做
        x_flatten = x.view(B, -1, self.d_inner).permute(0, 2, 1)  # (B, D, L)

        # 为了模拟 SS2D，我们构造 4 个序列
        # 注意：这里为了代码简洁，使用了近似实现。
        # 真正的完全体需要编写 CUDA kernel 来合并这4个流以避免显存爆炸。
        # 这里使用 PyTorch 原生操作实现逻辑等价：

        # 正常流
        feat1 = self.ssm(x.view(B, -1, self.d_inner))  # (B, L, D)

        # 翻转流 (模拟从右下到左上)
        x_flip = torch.flip(x, [1, 2]).view(B, -1, self.d_inner)
        feat2 = torch.flip(self.ssm(x_flip), [1]).view(B, H, W, self.d_inner)
        feat2 = feat2.view(B, -1, self.d_inner)

        # 转置流 (模拟纵向扫描)
        x_trans = x.transpose(1, 2).contiguous().view(B, -1, self.d_inner)
        feat3 = self.ssm(x_trans).view(B, W, H, self.d_inner).transpose(1, 2).contiguous().view(B, -1, self.d_inner)

        # 转置翻转流
        x_trans_flip = torch.flip(x.transpose(1, 2), [1, 2]).contiguous().view(B, -1, self.d_inner)
        feat4 = self.ssm(x_trans_flip).view(B, W, H, self.d_inner)
        feat4 = torch.flip(feat4, [1, 2]).transpose(1, 2).contiguous().view(B, -1, self.d_inner)

        # 3. 融合 4 个方向的信息 (Cross-Scan Merge)
        y = feat1 + feat2 + feat3 + feat4
        y = y * self.activation(z.view(B, -1, self.d_inner))  # Gating

        # 4. 输出投影
        out = self.out_proj(y)  # (B, L, C)
        out = out.view(B, H, W, C)

        return out + input_input  # Residual connection