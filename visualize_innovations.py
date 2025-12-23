import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import sys

# 确保能导入项目模块
sys.path.append(os.getcwd())
from src.core import YAMLConfig


class FeatureExtractor:
    def __init__(self, model):
        self.features = {}
        self.hooks = []

        # Hook 1: 提取小波 Backbone 第一层的输出
        # 这层保留了高频边缘信息
        self.hooks.append(model.backbone.conv1.register_forward_hook(self.hook_fn('wavelet')))

        # Hook 2: 提取 Encoder 输出，用于计算密度图
        self.hooks.append(model.encoder.register_forward_hook(self.hook_fn('encoder')))

    def hook_fn(self, name):
        def fn(module, input, output):
            self.features[name] = output

        return fn

    def remove(self):
        for h in self.hooks:
            h.remove()


def compute_density_map(feat_map):
    """复用 DensitySelector 的逻辑计算热力图"""
    # feat_map shape: [B, C, H, W]
    # 计算 L2 能量
    energy = torch.norm(feat_map, dim=1, keepdim=True)
    # 局部平滑
    density = torch.nn.functional.avg_pool2d(energy, kernel_size=3, stride=1, padding=1)
    # 归一化
    density = (density - density.min()) / (density.max() - density.min() + 1e-6)
    return density[0, 0].cpu().numpy()


def main(args):
    # 1. 加载模型
    print(f"Loading config from {args.config}...")
    cfg = YAMLConfig(args.config, resume=args.resume)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        state = checkpoint['model'] if 'model' in checkpoint else checkpoint['ema']['module']
        cfg.model.load_state_dict(state)
    else:
        print("警告: 未提供 checkpoint，使用随机初始化权重（仅供测试）")

    model = cfg.model.deploy().to(args.device)
    model.eval()

    # 2. 准备图片
    raw_img = Image.open(args.image).convert('RGB')
    w, h = raw_img.size
    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])
    img_tensor = transforms(raw_img)[None].to(args.device)
    orig_size = torch.tensor([w, h])[None].to(args.device)

    # 3. 注册 Hook 并推理
    extractor = FeatureExtractor(model)
    print("Running inference to extract features...")
    with torch.no_grad():
        model(img_tensor, orig_size)
    extractor.remove()

    # 4. 提取特征
    # 小波特征: 取 64 个通道的平均值，展示纹理
    wavelet_feat = extractor.features['wavelet'][0]  # [64, 320, 320]
    wavelet_viz = torch.mean(wavelet_feat, dim=0).cpu().numpy()

    # 密度图: 取 Encoder 输出的第一层特征计算
    encoder_out = extractor.features['encoder']  # List of tensor
    density_viz = compute_density_map(encoder_out[0])  # 取 stride=8 的特征层

    # 5. 绘图保存
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(raw_img)
    plt.title("Original Image (VisDrone)")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(wavelet_viz, cmap='gray')
    plt.title("Wavelet Stem Features\n(Rich High-Freq Details)")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(density_viz, cmap='jet')
    plt.title("Density Map\n(Focus on Crowds)")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('innovation_viz.png', dpi=300)
    print("✅ 创新点可视化已保存为: innovation_viz.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='configs/w_rtdetr_visdrone.yml')
    parser.add_argument('-r', '--resume', type=str, default=None, help='训练好的权重路径')
    parser.add_argument('-i', '--image', type=str, required=True, help='测试图片路径')
    parser.add_argument('-d', '--device', type=str, default='cuda')
    args = parser.parse_args()
    main(args)