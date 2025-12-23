import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import os
import sys
import argparse

sys.path.append(os.getcwd())
from src.core import YAMLConfig


class QueryExtractor:
    def __init__(self, model):
        self.ref_points = None
        self.hooks = []
        # 我们 Hook Decoder，因为 Decoder 接收 ref_points 作为输入
        # 在 w_rtdetr.py 中: self.decoder(tgt=tgt, ref_points=ref_points, ...)
        self.hooks.append(model.decoder.register_forward_hook(self.hook_fn))

    def hook_fn(self, module, args, output):
        # args[0] is tgt, args[1] is ref_points
        # 也有可能是 kwargs 传参，保险起见都检查
        # 这里的 ref_points 是 (cx, cy) 格式，范围 0-1
        if len(args) > 1:
            self.ref_points = args[1]

        # 针对 kwargs 的情况 (RTDETRTransformerDecoder 可能用 kwargs 调用)
        # 但在 w_rtdetr.py 中是位置参数调用，所以上面的应该有效。
        # 如果代码改动过，可以打印 args 查看。

    def remove(self):
        for h in self.hooks:
            h.remove()


def main(args):
    # 1. 加载模型
    cfg = YAMLConfig(args.config, resume=args.resume)
    checkpoint = torch.load(args.resume, map_location='cpu')
    state = checkpoint['model'] if 'model' in checkpoint else checkpoint['ema']['module']
    cfg.model.load_state_dict(state)

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

    # 3. 推理并提取 Query 点
    extractor = QueryExtractor(model)
    print("Running inference to extract queries...")
    with torch.no_grad():
        model(img_tensor, orig_size)
    extractor.remove()

    if extractor.ref_points is None:
        print("❌ 提取失败，未能 Hook 到 ref_points。请检查 w_rtdetr.py 中 decoder 的调用方式。")
        return

    # ref_points shape: [Batch, NumQueries, 2] (cx, cy)
    points = extractor.ref_points[0].cpu().numpy()

    # 4. 绘图 (散点图)
    plt.figure(figsize=(10, 10))
    plt.imshow(raw_img)

    # 将 0-1 坐标映射回原图尺寸
    img_pts_x = points[:, 0] * w
    img_pts_y = points[:, 1] * h

    # 画散点：红色圆点，半透明
    plt.scatter(img_pts_x, img_pts_y, c='red', s=15, alpha=0.6, label='Selected Queries')

    plt.title(f"Density-Guided Query Distribution (Top {len(points)})", fontsize=15)
    plt.axis('off')
    plt.legend(loc='upper right')

    save_path = "query_distribution.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"✅ Query 分布图已保存: {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='configs/w_rtdetr_visdrone.yml')
    parser.add_argument('-r', '--resume', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('-i', '--image', type=str, required=True, help='Path to test image')
    parser.add_argument('-d', '--device', type=str, default='cuda')
    args = parser.parse_args()
    main(args)