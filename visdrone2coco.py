import os
import json
import cv2
from tqdm import tqdm
from pathlib import Path

# VisDrone 类别映射 (只保留 10 个有效类别)
# 原始 ID: 0-ignored, 1-pedestrian, 2-people, 3-bicycle, 4-car, 5-van,
#          6-truck, 7-tricycle, 8-awning-tricycle, 9-bus, 10-motor, 11-others
# 我们只取 1-10，并映射到 0-9
VISDRONE_CLASSES = {
    1: 0,  # pedestrian
    2: 1,  # people
    3: 2,  # bicycle
    4: 3,  # car
    5: 4,  # van
    6: 5,  # truck
    7: 6,  # tricycle
    8: 7,  # awning-tricycle
    9: 8,  # bus
    10: 9  # motor
}

CATEGORIES = [
    {"id": 0, "name": "pedestrian"},
    {"id": 1, "name": "people"},
    {"id": 2, "name": "bicycle"},
    {"id": 3, "name": "car"},
    {"id": 4, "name": "van"},
    {"id": 5, "name": "truck"},
    {"id": 6, "name": "tricycle"},
    {"id": 7, "name": "awning-tricycle"},
    {"id": 8, "name": "bus"},
    {"id": 9, "name": "motor"},
]


def convert_to_coco(root_dir, split):
    """
    root_dir: 数据集根目录 (例如 ./datasets)
    split: 分割集名称 (例如 VisDrone2019-DET-train)
    """
    image_dir = os.path.join(root_dir, split, 'images')
    label_dir = os.path.join(root_dir, split, 'annotations')
    output_file = os.path.join(root_dir, split, f'{split.split("-")[-1]}.json')  # 生成 train.json 或 val.json

    print(f"正在转换 {split} ...")

    # 检查路径
    if not os.path.exists(image_dir) or not os.path.exists(label_dir):
        print(f"Error: 找不到目录 {image_dir} 或 {label_dir}")
        return

    images = []
    annotations = []
    ann_id = 0
    img_id = 0

    # 获取所有图片文件
    img_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]
    img_files.sort()

    for img_file in tqdm(img_files):
        # 1. 处理图片信息
        img_path = os.path.join(image_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: 无法读取图片 {img_path}")
            continue

        height, width = img.shape[:2]

        images.append({
            "id": img_id,
            "file_name": img_file,
            "height": height,
            "width": width
        })

        # 2. 处理对应的 TXT 标注
        txt_file = img_file.replace('.jpg', '.txt').replace('.png', '.txt')
        txt_path = os.path.join(label_dir, txt_file)

        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                lines = f.readlines()

            for line in lines:
                data = line.strip().split(',')
                # VisDrone 格式: <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
                if len(data) < 8:
                    continue

                # 提取信息
                bbox_left = int(data[0])
                bbox_top = int(data[1])
                bbox_width = int(data[2])
                bbox_height = int(data[3])
                category = int(data[5])

                # 过滤掉不需要的类别 (0:ignored, 11:others)
                if category not in VISDRONE_CLASSES:
                    continue

                # 转换为 COCO 类别 ID (0-9)
                coco_cat_id = VISDRONE_CLASSES[category]

                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": coco_cat_id,
                    "bbox": [bbox_left, bbox_top, bbox_width, bbox_height],
                    "area": bbox_width * bbox_height,
                    "iscrowd": 0
                })
                ann_id += 1

        img_id += 1

    # 保存 JSON
    coco_format = {
        "images": images,
        "annotations": annotations,
        "categories": CATEGORIES
    }

    with open(output_file, 'w') as f:
        json.dump(coco_format, f)

    print(f"转换完成！保存至: {output_file}")
    print(f"图片数: {len(images)}, 标注数: {len(annotations)}")


if __name__ == "__main__":
    # 假设你的目录结构是 ./datasets/VisDrone2019-DET-train
    ROOT_DIR = "./datasets"

    # 转换训练集
    convert_to_coco(ROOT_DIR, "VisDrone2019-DET-train")

    # 转换验证集
    convert_to_coco(ROOT_DIR, "VisDrone2019-DET-val")