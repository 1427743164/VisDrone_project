import torch
import torch.nn as nn
import torch.nn.functional as F
from src.core import register

# --- 引入之前的魔改模块 ---
from models.backbones.presnet_wavelet import WaveletPResNet
from models.necks.spectral_mamba_encoder import SpectralMambaEncoder
from models.layers.density_selector import DensityGuidedQuerySelection
from models.matchers.freq_sinkhorn_matcher import FrequencySinkhornMatcher

# --- 引入基础组件 ---
try:
    from src.zoo.rtdetr.rtdetr_decoder import RTDETRTransformerDecoder
    from src.zoo.rtdetr.rtdetr_postprocessor import RTDETRPostProcessor
    from src.zoo.rtdetr.rtdetr_criterion import SetCriterion
    # CDN 核心工具
    from src.zoo.rtdetr.denoising import get_contrastive_denoising_training_group
except ImportError:
    print("Warning: Standard RT-DETR modules not found. Ensure you are in the project root.")


@register
class WRTDETR(nn.Module):
    """
    [终极完全体 v3.0] W-RT-DETR

    集大成者：
    1. Wavelet Backbone (保留微小细节)
    2. Spectral Mamba Encoder (全局感受野 + 显存优化)
    3. Density Query Selection (强制关注密集区)
    4. CDN (Contrastive Denoising)
    5. Deep Supervision (Aux Loss 全开) - 真正的完全体逻辑
    """

    def __init__(self,
                 backbone_depth=50,
                 num_classes=80,
                 hidden_dim=256,
                 num_queries=300,
                 num_decoder_layers=6,
                 eval_spatial_size=[640, 640]):
        super().__init__()

        # 1. Backbone
        self.backbone = WaveletPResNet(
            depth=backbone_depth,
            return_idx=[1, 2, 3],
            freeze_at=0,
            freeze_norm=True
        )

        # 2. Encoder (Mamba)
        self.encoder = SpectralMambaEncoder(
            in_channels=[512, 1024, 2048],
            hidden_dim=hidden_dim,
            encoder_idx=[1, 2]
        )

        # 3. Query Selector (Density Guided)
        self.query_selector = DensityGuidedQuerySelection(
            num_queries=num_queries,
            alpha=0.4
        )

        # 4. Heads (Shared for Encoder/Decoder/CDN)
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
            nn.Sigmoid()
        )

        # 5. Decoder (完全体: 返回 Stack [L, B, N, C])
        self.decoder = RTDETRTransformerDecoder(
            hidden_dim=hidden_dim,
            num_decoder_layers=num_decoder_layers,
            num_head=8
        )

        # 6. Denoising Config (CDN 参数)
        self.num_denoising = 100
        self.label_noise_ratio = 0.5
        self.box_noise_scale = 1.0
        self.num_queries = num_queries  # 存一下，后面 split 用

        # Init bias
        prior_prob = 0.01
        bias_value = -torch.log(torch.tensor((1 - prior_prob) / prior_prob))
        nn.init.constant_(self.class_embed.bias, bias_value)

    def forward(self, x, targets=None):
        # x: [B, 3, H, W]

        # --- 1. Backbone ---
        feats = self.backbone(x)

        # --- 2. Encoder (Mamba) ---
        feats = self.encoder(feats)

        # --- 3. Prepare Features ---
        multi_scale_feats = []
        spatial_shapes = []
        for feat in feats:
            B, C, H, W = feat.shape
            spatial_shapes.append([H, W])
            multi_scale_feats.append(feat.flatten(2).transpose(1, 2))

        # [B, Total_Tokens, C]
        enc_memory = torch.cat(multi_scale_feats, dim=1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=x.device)

        # Encoder Prediction (for selection)
        enc_outputs_class = self.class_embed(enc_memory)
        enc_outputs_coord = self.bbox_embed(enc_memory)

        # --- 4. Density Guided Query Selection ---
        # 选出 Top-K Query
        topk_ind, topk_score = self.query_selector(enc_memory, enc_outputs_class, spatial_shapes)

        # Gather Content & Position
        batch_idx = torch.arange(len(topk_ind), device=topk_ind.device).unsqueeze(1)
        ref_points = enc_outputs_coord[batch_idx, topk_ind]  # [B, Nq, 4]
        tgt = enc_memory[batch_idx, topk_ind]  # [B, Nq, C]

        # --- 5. Contrastive DeNoising (CDN) 核心逻辑 ---
        dn_meta = None
        if self.training and targets is not None:
            # 生成带噪声的 GT 查询 (Noisy Queries)
            dn_output = get_contrastive_denoising_training_group(
                targets,
                self.num_denoising,
                self.num_classes,
                self.hidden_dim,
                self.label_noise_ratio,
                self.box_noise_scale,
                self.class_embed,
                num_anchors=topk_ind.shape[1]  # num_queries
            )

            # 解包 CDN 数据
            input_query_class, input_query_bbox, attn_mask, dn_meta = dn_output

            # 将 CDN Query 拼接到 正常 Query 前面
            tgt = torch.cat([input_query_class, tgt], dim=1)
            ref_points = torch.cat([input_query_bbox, ref_points], dim=1)

        else:
            attn_mask = None  # 推理模式不需要 mask

        # --- 6. Decoder ---
        # [完全体关键] dec_out_stack: [Layers, B, Total_Queries, C]
        dec_out_stack, dec_ref_points = self.decoder(
            tgt=tgt,
            ref_points=ref_points,
            memory=enc_memory,
            spatial_shapes=spatial_shapes,
            tgt_mask=attn_mask
        )

        # --- 7. Prediction Heads (对每一层都预测!) ---
        # outputs_class: [Layers, B, Total_Queries, Num_Classes]
        outputs_class = self.class_embed(dec_out_stack)
        outputs_coord = self.bbox_embed(dec_out_stack) + dec_ref_points  # Broadcasting

        # --- 8. Split CDN outputs & Aux Loss ---
        if self.training and dn_meta is not None:
            # 切分 CDN 部分和 真实 Query 部分
            # 注意维度 dim=2 是 Query 维度 (因为 dim=0 是 Layers)
            dn_out_class, outputs_class = torch.split(outputs_class, [dn_meta['dn_num_split'], self.num_queries], dim=2)
            dn_out_coord, outputs_coord = torch.split(outputs_coord, [dn_meta['dn_num_split'], self.num_queries], dim=2)

            # 主输出：取最后一层 [-1]
            out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}

            # [完全体关键] 记录 Auxiliary Outputs (前 L-1 层)
            out['aux_outputs'] = self._set_aux_loss(outputs_class[:-1], outputs_coord[:-1])

            # CDN 输出也需要 Aux
            out['dn_output'] = {'pred_logits': dn_out_class[-1], 'pred_boxes': dn_out_coord[-1], 'dn_meta': dn_meta}
            out['dn_aux_outputs'] = self._set_aux_loss(dn_out_class[:-1], dn_out_coord[:-1])

        else:
            # 推理模式，只取最后一层
            out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}

        # 记录 Encoder 输出 (用于 Auxiliary Loss 和 Density Selection Loss)
        out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # 辅助函数：将 stack 转为 list of dict，供 loss 计算使用
        return [{'pred_logits': a, 'pred_boxes': b} for a, b in zip(outputs_class, outputs_coord)]