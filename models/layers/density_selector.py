import torch
import torch.nn as nn
import torch.nn.functional as F


class DensityGuidedQuerySelection(nn.Module):
    """
    [完全体 - 修正版] Density-Guided Query Selection

    修复问题：
    之前的版本假设图片是正方形 (H=W)，这在 VisDrone (长方形图) 上会崩溃。
    新版本引入了 spatial_shapes 参数，能够精确还原长方形特征图的 2D 结构，
    确保密度计算在正确的空间位置上进行。
    """

    def __init__(self, num_queries=300, alpha=0.4):
        super().__init__()
        self.num_queries = num_queries
        self.alpha = alpha  # 密度权重的超参数

    def forward(self, enc_outputs, enc_logits, spatial_shapes=None):
        """
        Args:
            enc_outputs: Encoder输出的特征 [B, Total_L, C] (所有尺度的特征展平拼接)
            enc_logits: 预测的分类分数 [B, Total_L, NumClasses]
            spatial_shapes: 特征图尺寸信息 [N_levels, 2] (H, W)。
                            如果是单尺度，也可以推导，但建议传入以支持多尺度。
        """
        # 1. 计算基础分类分数 (Confidence Score)
        # shape: [B, Total_L]
        class_prob = enc_logits.sigmoid().max(dim=-1)[0]

        # 2. 计算密度图 (Density Map)
        # 如果没有提供 spatial_shapes，我们尝试推导 (仅限单尺度且为正方形的兜底逻辑)
        if spatial_shapes is None:
            # 这是一个不安全的假设，仅作为 fallback
            B, L, C = enc_outputs.shape
            H = int(L ** 0.5)
            W = L // H
            spatial_shapes = torch.tensor([[H, W]], device=enc_outputs.device)

        # 我们需要计算每个 token 位置的密度
        # 由于 enc_outputs 可能是多尺度拼接的，我们需要拆分处理，或者只处理最高分辨率层
        # 策略：为了计算效率，我们计算所有层的密度，然后拼接回来

        density_maps = []
        start_idx = 0

        for (h, w) in spatial_shapes:
            h, w = int(h), int(w)
            end_idx = start_idx + h * w

            # 提取当前尺度的特征 [B, h*w, C]
            feat_flat = enc_outputs[:, start_idx:end_idx, :]

            # 还原为 2D [B, C, h, w]
            feat_map = feat_flat.transpose(1, 2).view(-1, feat_flat.shape[-1], h, w)

            # --- 密度计算核心逻辑 ---
            # 1. 计算能量图 (L2 Norm): 反应特征强度
            energy = torch.norm(feat_map, dim=1, keepdim=True)  # [B, 1, h, w]

            # 2. 局部聚合 (AvgPool 3x3): 反应局部聚集程度
            # padding=1 保证尺寸不变
            local_density = F.avg_pool2d(energy, kernel_size=3, stride=1, padding=1)

            # 3. 展平回 [B, h*w]
            density_maps.append(local_density.flatten(1))

            start_idx = end_idx

        # 拼接所有层的密度图 [B, Total_L]
        full_density_map = torch.cat(density_maps, dim=1)

        # 归一化到 0~1 (Instance-level Normalization)
        # 避免某些样本整体特征值过大影响训练
        min_v = full_density_map.min(dim=1, keepdim=True)[0]
        max_v = full_density_map.max(dim=1, keepdim=True)[0]
        full_density_map = (full_density_map - min_v) / (max_v - min_v + 1e-6)

        # 3. 融合分数 (Score Fusion)
        # Final Score = (1 - alpha) * ClassScore + alpha * DensityScore
        # 这样既考虑了“像不像物体”，也考虑了“是不是在密集区”
        final_score = class_prob * (1 - self.alpha) + full_density_map * self.alpha

        # 4. Top-K 选择
        topk_scores, topk_indexes = torch.topk(final_score, self.num_queries, dim=1)

        return topk_indexes, topk_scores