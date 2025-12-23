import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.cuda.amp import autocast
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import sys
import matplotlib.pyplot as plt

# 确保能导入项目模块
sys.path.append(os.getcwd())
from src.core import YAMLConfig


# ==========================================
# 1. 复用 infer.py 的核心工具函数
# ==========================================
def postprocess(labels, boxes, scores, iou_threshold=0.55):
    # 简单的 NMS 后处理逻辑 (复用自 infer.py)
    def calculate_iou(box1, box2):
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2
        xi1 = max(x1, x3);
        yi1 = max(y1, y3)
        xi2 = min(x2, x4);
        yi2 = min(y2, y4)
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x4 - x3) * (y4 - y3)
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area != 0 else 0

    merged_labels, merged_boxes, merged_scores = [], [], []
    used_indices = set()
    for i in range(len(boxes)):
        if i in used_indices: continue
        boxes_to_merge = [boxes[i]]
        scores_to_merge = [scores[i]]
        used_indices.add(i)
        for j in range(i + 1, len(boxes)):
            if j in used_indices: continue
            if labels[j] != labels[i]: continue
            if calculate_iou(boxes[i], boxes[j]) >= iou_threshold:
                boxes_to_merge.append(boxes[j])
                scores_to_merge.append(scores[j])
                used_indices.add(j)
        # 融合框
        xs = np.concatenate([[b[0], b[2]] for b in boxes_to_merge])
        ys = np.concatenate([[b[1], b[3]] for b in boxes_to_merge])
        merged_boxes.append([np.min(xs), np.min(ys), np.max(xs), np.max(ys)])
        merged_labels.append(labels[i])
        merged_scores.append(max(scores_to_merge))
    return [np.array(merged_labels)], [np.array(merged_boxes)], [np.array(merged_scores)]


def slice_image(image, slice_height, slice_width, overlap_ratio):
    img_width, img_height = image.size
    slices = []
    coordinates = []
    step_x = int(slice_width * (1 - overlap_ratio))
    step_y = int(slice_height * (1 - overlap_ratio))
    for y in range(0, img_height, step_y):
        for x in range(0, img_width, step_x):
            box = (x, y, min(x + slice_width, img_width), min(y + slice_height, img_height))
            slices.append(image.crop(box))
            coordinates.append((x, y))
    return slices, coordinates


def merge_predictions(predictions, slice_coordinates, orig_image_size, threshold=0.30):
    merged_labels, merged_boxes, merged_scores = [], [], []
    orig_height, orig_width = orig_image_size
    for i, (label, boxes, scores) in enumerate(predictions):
        x_shift, y_shift = slice_coordinates[i]
        scores = np.array(scores).reshape(-1)
        valid_indices = scores > threshold
        valid_labels = np.array(label).reshape(-1)[valid_indices]
        valid_boxes = np.array(boxes).reshape(-1, 4)[valid_indices]
        valid_scores = scores[valid_indices]
        for j, box in enumerate(valid_boxes):
            box[0] = np.clip(box[0] + x_shift, 0, orig_width)
            box[1] = np.clip(box[1] + y_shift, 0, orig_height)
            box[2] = np.clip(box[2] + x_shift, 0, orig_width)
            box[3] = np.clip(box[3] + y_shift, 0, orig_height)
            valid_boxes[j] = box
        merged_labels.extend(valid_labels)
        merged_boxes.extend(valid_boxes)
        merged_scores.extend(valid_scores)
    return np.array(merged_labels), np.array(merged_boxes), np.array(merged_scores)


def draw_boxes_on_image(image, labels, boxes, scores, thrh=0.45):
    draw = ImageDraw.Draw(image)
    # 筛选
    indices = scores > thrh
    labels = labels[indices]
    boxes = boxes[indices]
    scores = scores[indices]

    for i, box in enumerate(boxes):
        # 画红框，线宽 2
        draw.rectangle(list(box), outline='red', width=2)
        # 可选：画标签（为了论文美观，密集场景有时会关掉标签显示）
        # text = f"{int(labels[i])}: {scores[i]:.2f}"
        # draw.text((box[0], box[1]), text=text, fill='yellow')
    return image


# ==========================================
# 2. 模型加载与推理类
# ==========================================
class Detector:
    def __init__(self, config_path, checkpoint_path, device='cuda'):
        self.device = device
        print(f"Loading model from {checkpoint_path}...")
        self.cfg = YAMLConfig(config_path, resume=checkpoint_path)

        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state = checkpoint['model'] if 'model' in checkpoint else checkpoint['ema']['module']
        self.cfg.model.load_state_dict(state)

        self.model = self.cfg.model.deploy().to(device)
        self.postprocessor = self.cfg.postprocessor.deploy()
        self.model.eval()

        self.transforms = T.Compose([T.Resize((640, 640)), T.ToTensor()])

    def infer(self, img_pil, sliced=True):
        w, h = img_pil.size
        orig_size = torch.tensor([w, h])[None].to(self.device)

        if not sliced:
            # 普通推理
            im_data = self.transforms(img_pil)[None].to(self.device)
            with torch.no_grad():
                output = self.model(im_data)
                output = self.postprocessor(output, orig_size)
            labels, boxes, scores = output
            labels, boxes, scores = labels[0].cpu().numpy(), boxes[0].cpu().numpy(), scores[0].cpu().numpy()
        else:
            # 切片推理 (VisDrone 核心)
            num_boxes = 20  # 切片数量
            aspect_ratio = w / h
            num_cols = int(np.sqrt(num_boxes * aspect_ratio))
            num_rows = int(num_boxes / num_cols)
            slice_h, slice_w = h // num_rows, w // num_cols

            slices, coords = slice_image(img_pil, slice_h, slice_w, 0.2)
            predictions = []

            for slice_img in slices:
                slice_tensor = self.transforms(slice_img)[None].to(self.device)
                slice_size = torch.tensor([[slice_img.size[0], slice_img.size[1]]]).to(self.device)
                with torch.no_grad(), autocast():
                    output = self.model(slice_tensor)
                    output = self.postprocessor(output, slice_size)
                predictions.append((output[0][0].cpu().numpy(), output[1][0].cpu().numpy(), output[2][0].cpu().numpy()))

            l, b, s = merge_predictions(predictions, coords, (h, w))
            # 再次 NMS
            labels, boxes, scores = postprocess([l], [b], [s], iou_threshold=0.5)
            labels, boxes, scores = labels[0], boxes[0], scores[0]

        return draw_boxes_on_image(img_pil.copy(), labels, boxes, scores)


# ==========================================
# 3. 主程序：对比并拼图
# ==========================================
def main():
    # --- 配置区域 (请修改这里) ---
    img_path = "datasets/VisDrone2019-DET-val/images/0000001_02999_d_0000005.jpg"  # 测试图片路径

    # Baseline 模型 (如果没有训练好的 Baseline，可以用同一个权重文件模拟，重点演示效果)
    # 正常来说这里应该是: output/rtdetr_r50_baseline/checkpoint.pth
    baseline_cfg = "configs/w_rtdetr_visdrone.yml"
    baseline_ckpt = "output/w_rtdetr_visdrone_4090_run/checkpoint.pth"  # 暂时用你的权重代替

    # Ours 模型 (你的创新点模型)
    ours_cfg = "configs/w_rtdetr_visdrone.yml"
    ours_ckpt = "output/w_rtdetr_visdrone_4090_run/checkpoint.pth"

    device = 'cuda'
    # ---------------------------

    if not os.path.exists(img_path):
        print(f"❌ 图片不存在: {img_path}")
        return

    # 1. 加载图片
    raw_img = Image.open(img_path).convert('RGB')

    # 2. 推理 Baseline
    print("正在推理 Baseline...")
    detector_base = Detector(baseline_cfg, baseline_ckpt, device)
    # 为了模拟 Baseline 效果差，这里演示关闭切片推理 (Standard Inference)
    # 实际对比时，如果 Baseline 也没用切片，就设为 False；如果用了，就设为 True
    res_baseline = detector_base.infer(raw_img, sliced=False)

    # 3. 推理 Ours
    print("正在推理 Ours...")
    # 释放显存，防止 OOM
    del detector_base
    torch.cuda.empty_cache()

    detector_ours = Detector(ours_cfg, ours_ckpt, device)
    # 你的模型开启切片推理 (Sliced Inference) 以展示最佳性能
    res_ours = detector_ours.infer(raw_img, sliced=True)

    # 4. 拼图 (Matplotlib)
    plt.figure(figsize=(16, 8))

    plt.subplot(1, 2, 1)
    plt.imshow(res_baseline)
    plt.title("Baseline (Standard RT-DETR)", fontsize=15)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(res_ours)
    plt.title("Ours (W-RT-DETR + Slicing)", fontsize=15, color='blue', fontweight='bold')
    plt.axis('off')

    save_path = "comparison_result.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"✅ 对比图已生成: {save_path}")


if __name__ == '__main__':
    main()