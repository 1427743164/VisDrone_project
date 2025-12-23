import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# 导入你的模型定义
sys.path.append(os.getcwd())
from models.backbones.presnet_wavelet import WaveletPResNet


def normalize_feat(feat):
    """将特征图归一化到 0-1 以便可视化"""
    feat = feat.detach().cpu().float()
    min_v = feat.min()
    max_v = feat.max()
    return (feat - min_v) / (max_v - min_v + 1e-6)


def main():
    # --- 配置 ---
    img_path = "datasets/VisDrone2019-DET-val/images/0000001_02999_d_0000005.jpg"  # 替换为你的测试图
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not os.path.exists(img_path):
        print(f"❌ 图片不存在: {img_path}")
        return

    # 1. 准备图片
    raw_img = Image.open(img_path).convert('RGB')
    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])
    img_tensor = transforms(raw_img)[None].to(device)

    # 2. 加载标准 ResNet-50 (Baseline)
    print("Loading Standard ResNet-50...")
    resnet_base = torchvision.models.resnet50(pretrained=True).to(device)
    # 提取第一层 conv1 的输出 (Standard Conv 7x7)
    # 使用 hook
    conv_feats = {}

    def hook_conv(module, input, output):
        conv_feats['out'] = output

    h1 = resnet_base.conv1.register_forward_hook(hook_conv)
    resnet_base(img_tensor)
    h1.remove()

    # 3. 加载你的 WaveletPResNet (Ours)
    print("Loading WaveletPResNet...")
    # 注意：这里不需要加载训练权重，因为我们主要看 WaveletDownsampling 的初始物理特性
    # (小波变换本身是固定参数的数学变换，conv融合层即使随机初始化也能看出纹理保留特性，当然如果有训练权重更好)
    wavelet_model = WaveletPResNet(depth=50).to(device)
    wavelet_model.eval()

    wavelet_feats = {}

    def hook_wavelet(module, input, output):
        wavelet_feats['out'] = output

    # 你的模型第一层是 conv1 (HaarWaveletDownsampling)
    h2 = wavelet_model.conv1.register_forward_hook(hook_wavelet)
    wavelet_model(img_tensor)
    h2.remove()

    # 4. 特征处理
    # Standard Conv1 输出: [B, 64, 320, 320]
    feat_base = conv_feats['out'][0]
    # 取平均通道展示结构，或者取能量最大的通道
    viz_base = torch.mean(feat_base, dim=0)

    # Wavelet Conv1 输出: [B, 64, 320, 320]
    feat_wavelet = wavelet_feats['out'][0]
    viz_wavelet = torch.mean(feat_wavelet, dim=0)

    # 5. 绘图对比
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(raw_img)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(normalize_feat(viz_base), cmap='gray')
    plt.title("Standard Conv7x7 (Blurred)", fontsize=14)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(normalize_feat(viz_wavelet), cmap='gray')
    plt.title("Ours Wavelet Stem (Sharp Details)", fontsize=14, color='blue', fontweight='bold')
    plt.axis('off')

    save_path = "feature_comparison.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"✅ 特征对比图已保存: {save_path}")


if __name__ == '__main__':
    main()