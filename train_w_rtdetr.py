import sys
import os
import argparse

# ----------------------------------------------------
# 1. 环境与路径设置
# ----------------------------------------------------
# 获取当前脚本所在目录（根目录）
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

# ----------------------------------------------------
# 2. 核心模块导入
# ----------------------------------------------------
# ⚠️ 关键点：必须导入 models，否则相关类（如 WRTDETR, Backbone 等）不会注册到注册表中，
# 会导致配置文件读取时报错 "KeyError" 或 "Class not found"。
import models

# 导入项目内部的核心配置和分布式工具
import src.misc.dist as dist
from src.core import YAMLConfig
from src.solver import TASKS


def main(args):
    # 初始化分布式环境（即使是单卡训练也建议保留此初始化）
    dist.init_distributed()

    # 设置随机种子
    if args.seed is not None:
        dist.set_seed(args.seed)

    # 检查参数冲突
    assert not all([args.tuning, args.resume]), \
        'Only support from_scratch or resume or tuning at one time'

    # ----------------------------------------------------
    # 3. 加载配置
    # ----------------------------------------------------
    # 使用项目自带的 YAMLConfig 解析器，它会处理 __include__ 依赖和参数合并
    cfg = YAMLConfig(
        args.config,
        resume=args.resume,
        use_amp=args.amp,
        tuning=args.tuning
    )

    # ----------------------------------------------------
    # 4. 实例化 Solver 并开始训练
    # ----------------------------------------------------
    # 根据配置中的 task（如 'detection'）从 TASKS 注册表中获取对应的 Solver 类
    solver = TASKS[cfg.yaml_cfg['task']](cfg)

    if args.test_only:
        solver.val()
    else:
        solver.fit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="RT-DETR Training Script")

    # 默认指向你最新的配置文件 configs/w_rtdetr_visdrone.yml
    parser.add_argument('--config', '-c', type=str, default='configs/w_rtdetr_visdrone.yml',
                        help='path to configuration file')

    parser.add_argument('--resume', '-r', type=str, default=None,
                        help='path to resume checkpoint')
    parser.add_argument('--tuning', '-t', type=str, default=None,
                        help='path to tuning checkpoint')
    parser.add_argument('--test-only', action='store_true', default=False,
                        help='only run validation')
    parser.add_argument('--amp', action='store_true', default=False,
                        help='enable Automatic Mixed Precision (AMP)')
    parser.add_argument('--seed', type=int, default=None,
                        help='random seed')

    args = parser.parse_args()

    print(f"Using configuration: {args.config}")
    main(args)