import torch
import torch.nn as nn
import torch.nn.functional as F

# 引入我们上一阶段写好的完全体小波模块
from models.layers.wavelet import HaarWaveletDownsampling


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        # width 这里的实现简化了 group conv 等变体，专注于标准 ResNet 结构
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class WaveletPResNet(nn.Module):
    """
    [完全体] Wavelet-Integrated PResNet Backbone

    核心改动：
    将原本的第一层 Conv7x7 替换为 HaarWaveletDownsampling。
    这种设计保留了 0-1 阶段下采样时的所有频域信息，极大提升小目标检测能力。
    """

    def __init__(self, depth, variant='d', num_stages=4, return_idx=[0, 1, 2, 3], freeze_at=0, freeze_norm=True):
        super(WaveletPResNet, self).__init__()

        # 配置 ResNet 的层数结构
        blocks_config = {
            18: (BasicBlock, [2, 2, 2, 2]),
            34: (BasicBlock, [3, 4, 6, 3]),
            50: (Bottleneck, [3, 4, 6, 3]),
            101: (Bottleneck, [3, 4, 23, 3]),
        }

        if depth not in blocks_config:
            raise ValueError(f"Unsupported depth: {depth}. Choose from 18, 34, 50, 101.")

        block, layers = blocks_config[depth]
        self.inplanes = 64
        self.variant = variant
        self.return_idx = return_idx
        self.freeze_at = freeze_at
        self.freeze_norm = freeze_norm

        # --- 核心创新点：Stem Layer (入口层) ---
        # 原版: self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # 魔改版: 使用 HaarWaveletDownsampling
        self.conv1 = HaarWaveletDownsampling(in_channels=3, out_channels=64)

        # 注意：Wavelet 模块内部已经包含了 BN 和 SiLU，所以这里不需要再加 BN/ReLU
        # 但为了保持 ResNet 结构后续兼容性，我们保留 MaxPool
        # (有些激进的修改会连 MaxPool 也去掉，但为了保证输出尺寸为 H/4，这里保留 MaxPool 是最稳妥的)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # --- ResNet Stages ---
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # 初始化权重 (除了 Wavelet 部分，它有自己的初始化)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and not hasattr(m, 'haar_weights'):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        # 处理残差连接的维度匹配
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # 1. Stem (Wavelet + MaxPool) -> H/4
        x = self.conv1(x)
        # 此时 x 已经是 64通道，尺寸 H/2

        x = self.maxpool(x)
        # 此时 x 尺寸 H/4

        res = []

        # 2. Stages
        x = self.layer1(x)  # H/4
        if 0 in self.return_idx: res.append(x)

        x = self.layer2(x)  # H/8
        if 1 in self.return_idx: res.append(x)

        x = self.layer3(x)  # H/16
        if 2 in self.return_idx: res.append(x)

        x = self.layer4(x)  # H/32
        if 3 in self.return_idx: res.append(x)

        return res


if __name__ == "__main__":
    # --- 单元测试 ---
    print("Testing WaveletPResNet...")
    model = WaveletPResNet(depth=50)
    inputs = torch.randn(1, 3, 640, 640)
    outputs = model(inputs)

    print(f"Input: {inputs.shape}")
    for i, out in enumerate(outputs):
        print(f"Output[{i}]: {out.shape}")

    # 验证输出层级 (RT-DETR 通常使用最后3层，即 stride 8, 16, 32)
    # Output[0] (stride 8):  [1, 512, 80, 80]
    # Output[1] (stride 16): [1, 1024, 40, 40]
    # Output[2] (stride 32): [1, 2048, 20, 20]
    assert len(outputs) == 4  # 默认返回所有4个stage
    print("WaveletPResNet Test Passed!")