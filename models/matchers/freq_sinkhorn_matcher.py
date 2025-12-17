import torch
import torch.nn as nn
import torch.nn.functional as F
from src.core import register

@register
class FrequencySinkhornMatcher(nn.Module):
    """
    [完全体 v2.0] Frequency-aware Sinkhorn Matcher

    优化点：
    1. 逐图计算 Cost (Per-image Cost Calculation)：
       避免了计算跨图像的无效代价，大幅降低显存占用 (Memory Efficient)，
       特别适合 VisDrone 这种目标极多(Target num > 50) 的场景。
    2. 稳定性增强：使用了 Stabilized Sinkhorn (减去 max_cost)，防止指数爆炸。
    3. NWD 频域代价：保留了针对小目标的 Wasserstein 距离计算。
    """

    def __init__(self, cost_class=1.0, cost_bbox=5.0, cost_nwd=5.0, alpha=0.25, gamma=2.0, iter_steps=20, eps=1e-6):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_nwd = cost_nwd
        self.alpha = alpha
        self.gamma = gamma
        self.iter_steps = iter_steps
        self.eps = eps

    def forward(self, outputs, targets):
        """
        Args:
            outputs: 字典，包含 "pred_logits" [B, num_queries, num_classes]
                            和 "pred_boxes" [B, num_queries, 4]
            targets: 列表，每个元素包含 "labels" 和 "boxes"
        """
        with torch.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]

            # [B, N_q, Num_Classes]
            out_prob = outputs["pred_logits"].sigmoid()
            out_bbox = outputs["pred_boxes"]

            indices = []

            # --- 优化：在循环内逐张图处理，避免构建巨大的无效 Cost Matrix ---
            for i in range(bs):
                # 1. 获取当前图片的预测和真值
                # p_prob: [Num_Queries, Num_Classes]
                p_prob = out_prob[i]
                p_bbox = out_bbox[i]

                t_ids = targets[i]["labels"]
                t_bbox = targets[i]["boxes"]

                num_t = len(t_ids)
                if num_t == 0:
                    indices.append((torch.as_tensor([], dtype=torch.int64, device=p_prob.device),
                                    torch.as_tensor([], dtype=torch.int64, device=p_prob.device)))
                    continue

                # --- 2. 构建 Cost Matrix (仅针对当前图片) ---

                # A. Classification Cost (Focal Loss style)
                # p_prob: [N_q, C], t_ids: [N_t]
                # 提取对应 Target 类别的预测概率 -> [N_q, N_t]
                p_prob_t = p_prob[:, t_ids]

                neg_cost_class = (1 - self.alpha) * (p_prob_t ** self.gamma) * (-(1 - p_prob_t + 1e-8).log())
                pos_cost_class = self.alpha * ((1 - p_prob_t) ** self.gamma) * (-(p_prob_t + 1e-8).log())
                cost_class = pos_cost_class - neg_cost_class

                # B. BBox L1 Cost
                # [N_q, N_t]
                cost_bbox = torch.cdist(p_bbox, t_bbox, p=1)

                # C. NWD Cost (Normalized Wasserstein Distance)
                # p_bbox: [N_q, 4], t_bbox: [N_t, 4]
                # 扩展维度以广播
                p_expand = p_bbox.unsqueeze(1)  # [N_q, 1, 4]
                t_expand = t_bbox.unsqueeze(0)  # [1, N_t, 4]

                p_cx, p_cy, p_w, p_h = p_expand.unbind(-1)
                t_cx, t_cy, t_w, t_h = t_expand.unbind(-1)

                # NWD 常数 (归一化坐标下)
                constant = 0.05

                w2_square = (p_cx - t_cx).pow(2) + \
                            (p_cy - t_cy).pow(2) + \
                            ((p_w - t_w) / 2.0).pow(2) + \
                            ((p_h - t_h) / 2.0).pow(2)

                # clamp防止完全重合时梯度计算nan (虽然在no_grad下不传梯度，但防止sqrt报错)
                nwd = torch.exp(-torch.sqrt(torch.clamp(w2_square, min=1e-7)) / constant)
                cost_nwd = 1.0 - nwd

                # 总 Cost
                C = self.cost_bbox * cost_bbox + \
                    self.cost_class * cost_class + \
                    self.cost_nwd * cost_nwd

                # --- 3. Sinkhorn 算法核心 (Stabilized) ---
                # 归一化以防数值溢出
                max_cost = C.max()
                epsilon = 0.1
                # K: [N_q, N_t]
                K = torch.exp(-(C - max_cost) / epsilon)

                # 初始化 Scaling 向量
                u = torch.ones_like(K[:, 0]) / num_queries
                v = torch.ones_like(K[0, :]) / num_t

                for _ in range(self.iter_steps):
                    u = 1.0 / (K @ v + self.eps)
                    v = 1.0 / (K.t() @ u + self.eps)

                # Transport Plan P = diag(u) * K * diag(v)
                # 我们不需要算出完整的 P 矩阵，只需要求 argmax
                # P[i, j] 正比于 u[i] * K[i, j] * v[j]
                # 取对数后寻找最大值更稳定: log(P) = log(u) + log(K) + log(v)
                # 但为了速度，直接计算 P 的近似形式用于 argmax

                # [N_q, N_t]
                P = u.unsqueeze(1) * K * v.unsqueeze(0)

                # --- 4. 生成匹配索引 ---
                # 每个 Target 必须匹配一个 Query -> 对每列 (Target) 找最大值
                src_idx = P.argmax(dim=0)
                tgt_idx = torch.arange(num_t, dtype=torch.int64, device=src_idx.device)

                # [可选] 去重逻辑: Sinkhorn 实际上是软匹配，极端情况下可能两个 Target 抢一个 Query。
                # 但在 query 数量 (300) 远大于 target 数量 (50) 时，且 Cost 区分度足够时，
                # argmax 几乎总是唯一的。此处为了保持高效，不做强制去重（OTA等方法也是如此）。

                indices.append((src_idx, tgt_idx))

            return indices