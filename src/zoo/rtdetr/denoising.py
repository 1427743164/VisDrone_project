"""by lyuwenyu
"""

import torch 

from .utils import inverse_sigmoid
from .box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh


def get_contrastive_denoising_training_group(targets,
                                             num_denoising,
                                             num_classes,
                                             hidden_dim,
                                             label_noise_ratio,
                                             box_noise_scale,
                                             class_embed,
                                             num_anchors=None): # [关键修复] 补上 num_anchors 参数
    """
    完全体 Contrastive DeNoising (CDN) 逻辑
    作用：在训练阶段生成带噪声的 Ground Truth Queries，加速模型收敛。
    """
    if num_denoising <= 0:
        return None, None, None, None

    # 如果没有显式传入 num_anchors，则默认使用 300 (RT-DETR 默认值)
    # W-RT-DETR 会传入实际的 topk 数量
    num_queries = num_anchors if num_anchors is not None else 300

    num_gts = [len(t['labels']) for t in targets]
    device = targets[0]['labels'].device

    max_gt_num = max(num_gts)
    if max_gt_num == 0:
        return None, None, None, None

    # 计算组数：每一组包含正样本和负样本
    num_group = num_denoising // max_gt_num
    num_group = 1 if num_group == 0 else num_group

    bs = len(num_gts)

    # 初始化噪声 Query (类别填充为背景类 num_classes)
    input_query_class = torch.full([bs, max_gt_num], num_classes, dtype=torch.int32, device=device)
    input_query_bbox = torch.zeros([bs, max_gt_num, 4], device=device)
    pad_gt_mask = torch.zeros([bs, max_gt_num], dtype=torch.bool, device=device)

    # 将真实的 GT 信息填入
    for i in range(bs):
        num_gt = num_gts[i]
        if num_gt > 0:
            input_query_class[i, :num_gt] = targets[i]['labels']
            input_query_bbox[i, :num_gt] = targets[i]['boxes']
            pad_gt_mask[i, :num_gt] = 1

    # 扩展组数
    input_query_class = input_query_class.tile([1, 2 * num_group])
    input_query_bbox = input_query_bbox.tile([1, 2 * num_group, 1])
    pad_gt_mask = pad_gt_mask.tile([1, 2 * num_group])

    # 构建正负样本掩码
    negative_gt_mask = torch.zeros([bs, max_gt_num * 2, 1], device=device)
    negative_gt_mask[:, max_gt_num:] = 1
    negative_gt_mask = negative_gt_mask.tile([1, num_group, 1])
    positive_gt_mask = 1 - negative_gt_mask

    positive_gt_mask = positive_gt_mask.squeeze(-1) * pad_gt_mask
    dn_positive_idx = torch.nonzero(positive_gt_mask)[:, 1]
    dn_positive_idx = torch.split(dn_positive_idx, [n * num_group for n in num_gts])

    num_denoising = int(max_gt_num * 2 * num_group)

    # 1. 添加类别噪声 (Label Noise)
    if label_noise_ratio > 0:
        mask = torch.rand_like(input_query_class, dtype=torch.float) < (label_noise_ratio * 0.5)
        new_label = torch.randint_like(mask, 0, num_classes, dtype=input_query_class.dtype)
        input_query_class = torch.where(mask & pad_gt_mask, new_label, input_query_class)

    # 2. 添加位置噪声 (Box Noise)
    if box_noise_scale > 0:
        known_bbox = box_cxcywh_to_xyxy(input_query_bbox)
        diff = torch.tile(input_query_bbox[..., 2:] * 0.5, [1, 1, 2]) * box_noise_scale
        rand_sign = torch.randint_like(input_query_bbox, 0, 2) * 2.0 - 1.0
        rand_part = torch.rand_like(input_query_bbox)
        rand_part = (rand_part + 1.0) * negative_gt_mask + rand_part * (1 - negative_gt_mask)
        rand_part *= rand_sign
        known_bbox += rand_part * diff
        known_bbox.clip_(min=0.0, max=1.0)
        input_query_bbox = box_xyxy_to_cxcywh(known_bbox)
        input_query_bbox = inverse_sigmoid(input_query_bbox)

    # 3. 映射到特征空间
    input_query_class = class_embed(input_query_class)

    # 4. 构建 Attention Mask (关键逻辑：防止泄露)
    tgt_size = num_denoising + num_queries
    attn_mask = torch.full([tgt_size, tgt_size], False, dtype=torch.bool, device=device)

    # 正常匹配 Query 不能看到去噪 Query
    attn_mask[num_denoising:, :num_denoising] = True

    # 不同组的去噪 Query 之间互不可见
    for i in range(num_group):
        if i == 0:
            attn_mask[max_gt_num * 2 * i: max_gt_num * 2 * (i + 1), max_gt_num * 2 * (i + 1): num_denoising] = True
        elif i == num_group - 1:
            attn_mask[max_gt_num * 2 * i: max_gt_num * 2 * (i + 1), :max_gt_num * i * 2] = True
        else:
            attn_mask[max_gt_num * 2 * i: max_gt_num * 2 * (i + 1), max_gt_num * 2 * (i + 1): num_denoising] = True
            attn_mask[max_gt_num * 2 * i: max_gt_num * 2 * (i + 1), :max_gt_num * 2 * i] = True

    dn_meta = {
        "dn_positive_idx": dn_positive_idx,
        "dn_num_group": num_group,
        "dn_num_split": [num_denoising, num_queries]
    }

    return input_query_class, input_query_bbox, attn_mask, dn_meta