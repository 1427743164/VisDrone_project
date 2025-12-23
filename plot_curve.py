import matplotlib.pyplot as plt
import os
import argparse


def plot_logs(log_path, save_path='training_curves.png'):
    if not os.path.exists(log_path):
        print(f"错误：找不到日志文件 {log_path}")
        return

    epochs = []
    losses = []
    maps = []

    with open(log_path, 'r') as f:
        lines = f.readlines()

    # 解析日志
    for line in lines:
        # 提取 Loss
        if "Averaged stats:" in line and "loss:" in line:
            try:
                # 格式示例: Averaged stats: lr: 0.000010  loss: 12.3456 ...
                parts = line.split()
                loss_idx = parts.index("loss:") + 1
                loss_val = float(parts[loss_idx])
                losses.append(loss_val)
            except:
                pass

        # 提取 mAP
        if "best_stat:" in line:
            try:
                # 格式示例: best_stat:  {'epoch': 0, 'coco_eval_bbox': 0.123...}
                epoch_str = line.split("'epoch': ")[1].split(",")[0]
                epochs.append(int(epoch_str))

                # 提取 mAP
                map_part = line.split("'coco_eval_bbox': ")[1]
                # 处理可能是列表 [0.12, ...] 或数值 0.12 的情况
                if map_part.startswith("["):
                    map_val = float(map_part.split(",")[0].strip("["))
                else:
                    map_val = float(map_part.split("}")[0])
                maps.append(map_val)
            except:
                pass

    # 开始绘图
    plt.figure(figsize=(12, 5))

    # 1. Loss 曲线
    plt.subplot(1, 2, 1)
    # 这里的 x 轴是 Epoch，但因为 Loss 每个 Epoch 打印一次，直接用 range 即可
    plt.plot(range(len(losses)), losses, label='Total Loss', color='red', linewidth=2)
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # 2. mAP 曲线
    plt.subplot(1, 2, 2)
    if len(epochs) > 0:
        plt.plot(epochs, maps, label='mAP 50:95', color='blue', marker='o', linewidth=2)
        plt.title('Validation mAP Curve')
        plt.xlabel('Epoch')
        plt.ylabel('mAP')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
    else:
        plt.text(0.5, 0.5, 'No mAP data yet', ha='center', va='center')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"✅ 训练曲线已保存为: {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 默认指向你 4090 运行的日志目录，请根据实际情况修改
    parser.add_argument('--log', type=str, default='output/w_rtdetr_visdrone_4090_run/log.txt')
    args = parser.parse_args()
    plot_logs(args.log)