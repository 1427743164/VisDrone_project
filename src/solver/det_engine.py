"""
by lyuwenyu
"""

import math
import sys
from typing import Iterable

import torch
import torch.amp
from torch.utils.data import DataLoader

from src.misc import logger
from src.core import yaml_utils
from src.zoo.rtdetr.utils import inverse_sigmoid

# [关键修改] 显式导入 CocoEvaluator，而不是通过 logger 导入
from src.data.coco.coco_eval import CocoEvaluator

from tqdm import tqdm

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, **kwargs):
    model.train()
    criterion.train()

    metric_logger = logger.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', logger.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    header = 'Epoch: [{}]'.format(epoch)
    print_freq = kwargs.get('print_freq', 10)

    ema = kwargs.get('ema', None)
    scaler = kwargs.get('scaler', None)

    pbar = tqdm(data_loader, desc=header, file=sys.stdout)

    for samples, targets in pbar:
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if scaler is not None:
            with torch.amp.autocast('cuda', enabled=True):
                outputs = model(samples, targets)

            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            scaler.scale(losses).backward()

            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        else:
            outputs = model(samples, targets)
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            optimizer.zero_grad()
            losses.backward()

            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()

        if ema is not None:
            ema.update(model)

        loss_dict_reduced = logger.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        pbar.set_postfix({
            'loss': f"{loss_value:.4f}",
            'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
        })

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir):
    model.eval()
    criterion.eval()

    metric_logger = logger.MetricLogger(delimiter="  ")
    header = 'Test:'

    pbar = tqdm(data_loader, desc=header, file=sys.stdout)

    if hasattr(postprocessors, 'keys'):
        iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    else:
        iou_types = ('bbox',)

    # [关键修改] 直接使用导入的 CocoEvaluator 类，而不是 logger.CocoEvaluator
    coco_evaluator = CocoEvaluator(base_ds, iou_types)

    panoptic_evaluator = None

    for samples, targets in pbar:
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        if hasattr(postprocessors, 'keys') and 'bbox' in postprocessors.keys():
            results = postprocessors['bbox'](outputs, orig_target_sizes)
        else:
            results = postprocessors(outputs, orig_target_sizes)

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}

        if coco_evaluator is not None:
            coco_evaluator.update(res)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    if coco_evaluator is not None:
        if 'bbox' in iou_types:
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in iou_types:
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()

    return stats, coco_evaluator