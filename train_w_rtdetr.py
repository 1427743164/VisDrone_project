import sys
import os

# 假设你的核心训练代码在 src 或 similar 目录下，确保能引用到
# sys.path.append(...)

from ultralytics import RTDETR  # 或者你所使用的具体库/类的导入方式


def main():
    # 1. 指定配置文件路径 (保持当前路径，不新建文件夹)
    # 假设配置文件就在当前目录下，名为 'rtdetr_w_rtdetr_config.yaml' (请根据实际文件名修改)
    config_path = 'rtdetr_w_rtdetr_config.yaml'

    # 确保配置文件存在
    if not os.path.exists(config_path):
        print(f"Error: 配置文件 {config_path} 未找到，请检查路径。")
        return

    # 2. 加载模型
    # 这里根据你的具体项目结构，可能是加载预训练权重，也可能是从配置加载
    # 如果是 ultralytics 的用法：
    model = RTDETR('rtdetr-l.pt')  # 或者你的 .yaml 结构文件

    # 3. 开始训练
    # 对应命令行参数: model.train(data=..., epochs=..., ...)
    # 这里我们将配置文件的内容传进去，或者如果你的脚本已经内部读取该 yaml，直接运行即可

    print(f"开始使用配置文件: {config_path} 进行训练...")

    results = model.train(
        data=config_path,  # 这里直接传入配置文件路径作为 data 参数
        epochs=100,  # 训练轮数
        imgsz=640,  # 输入图片大小
        device='0',  # 显卡设备
        project='runs/train',  # 结果保存路径
        name='exp',  # 实验名称
        exist_ok=True,  # 如果存在则覆盖/追加，不报错
        # batch=16,         # 根据显存调整
        # workers=4         # 根据CPU核心数调整
    )

    print("训练完成！")


if __name__ == '__main__':
    main()