import torch
import torch.nn as nn
import torch.nn.functional as F
from src.core import register

# 引入工具
from src.zoo.rtdetr.utils import inverse_sigmoid  # [关键] 引入 inverse_sigmoid

from models.backbones.presnet_wavelet import WaveletPResNet
from models.necks.spectral_mamba_encoder import SpectralMambaEncoder
from models.layers.density_selector import DensityGuidedQuerySelection
from models.matchers.freq_sinkhorn_matcher import FrequencySinkhornMatcher

try:
    from src.zoo.rtdetr.rtdetr_decoder import RTDETRTransformerDecoder
    from src.zoo.rtdetr.rtdetr_postprocessor import RTDETRPostProcessor
    from src.zoo.rtdetr.rtdetr_criterion import SetCriterion
    from src.zoo.rtdetr.denoising import get_contrastive_denoising_training_group
except ImportError:
    print("Warning: Standard RT-DETR modules not found.")


@register
class WRTDETR(nn.Module):
    """
    [终极完全体 v3.5] W-RT-DETR
    修复内容：
    1. 修正 Box 解码逻辑：使用 (delta + inv(ref)).sigmoid()，防止出现负框导致的 AssertionError
    2. bbox_embed 去除 Sigmoid，输出 Logits
    """

    def __init__(self,
                 backbone_depth=50,
                 num_classes=80,
                 hidden_dim=256,
                 num_queries=300,
                 num_decoder_layers=6,
                 eval_spatial_size=[640, 640]):
        super().__init__()

        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.num_decoder_layers = num_decoder_layers

        self.backbone = WaveletPResNet(
            depth=backbone_depth,
            return_idx=[1, 2, 3],
            freeze_at=0,
            freeze_norm=True
        )

        self.encoder = SpectralMambaEncoder(
            in_channels=[512, 1024, 2048],
            hidden_dim=hidden_dim,
            encoder_idx=[1, 2]
        )

        self.query_selector = DensityGuidedQuerySelection(
            num_queries=num_queries,
            alpha=0.4
        )

        self.class_embed = nn.Linear(hidden_dim, num_classes)

        # [关键修改] 移除 Sigmoid，因为我们需要预测未归一化的偏移量 (Delta)
        self.bbox_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)
        )

        self.denoising_class_embed = nn.Embedding(num_classes + 1, hidden_dim, padding_idx=num_classes)

        self.decoder = RTDETRTransformerDecoder(
            hidden_dim=hidden_dim,
            num_decoder_layers=num_decoder_layers,
            num_head=8
        )

        self.num_denoising = 100
        self.label_noise_ratio = 0.5
        self.box_noise_scale = 1.0

        prior_prob = 0.01
        bias_value = -torch.log(torch.tensor((1 - prior_prob) / prior_prob))
        nn.init.constant_(self.class_embed.bias, bias_value)

    def forward(self, x, targets=None):
        feats = self.backbone(x)
        feats = self.encoder(feats)

        multi_scale_feats = []
        spatial_shapes = []
        for feat in feats:
            B, C, H, W = feat.shape
            spatial_shapes.append([H, W])
            multi_scale_feats.append(feat.flatten(2).transpose(1, 2))

        enc_memory = torch.cat(multi_scale_feats, dim=1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=x.device)

        enc_outputs_class = self.class_embed(enc_memory)

        # [关键逻辑] Encoder 输出也需要 Sigmoid (因为 bbox_embed 现在没有 Sigmoid 了)
        enc_outputs_coord = self.bbox_embed(enc_memory).sigmoid()

        topk_ind, topk_score = self.query_selector(enc_memory, enc_outputs_class, spatial_shapes)

        batch_idx = torch.arange(len(topk_ind), device=topk_ind.device).unsqueeze(1)
        ref_points = enc_outputs_coord[batch_idx, topk_ind]
        tgt = enc_memory[batch_idx, topk_ind]

        dn_meta = None
        if self.training and targets is not None:
            dn_output = get_contrastive_denoising_training_group(
                targets,
                self.num_denoising,
                self.num_classes,
                self.hidden_dim,
                self.label_noise_ratio,
                self.box_noise_scale,
                self.denoising_class_embed,
                num_anchors=topk_ind.shape[1]
            )
            input_query_class, input_query_bbox, attn_mask, dn_meta = dn_output
            tgt = torch.cat([input_query_class, tgt], dim=1)
            # 这里的 ref_points 都是 [0, 1] 区间，可以直接拼接
            ref_points = torch.cat([input_query_bbox, ref_points], dim=1)
        else:
            attn_mask = None

        dec_out_stack, dec_ref_points = self.decoder(
            tgt=tgt,
            ref_points=ref_points,
            memory=enc_memory,
            spatial_shapes=spatial_shapes,
            tgt_mask=attn_mask
        )

        # --- 6. Prediction Heads (修正后的坐标解码) ---
        outputs_class = self.class_embed(dec_out_stack)

        # [核心修复] 使用标准的 DETR 坐标细化公式
        # ref_points 是 [0, 1]，先转回 logits (inverse_sigmoid)
        # 然后加上网络预测的偏移量 (bbox_embed)
        # 最后再 sigmoid 变回 [0, 1]
        # 这样保证了输出框永远在 [0, 1] 范围内，不会出现负数，解决了 AssertionError
        tmp = self.bbox_embed(dec_out_stack)
        tmp += inverse_sigmoid(dec_ref_points)
        outputs_coord = tmp.sigmoid()

        # --- 7. Split & Loss ---
        if self.training and dn_meta is not None:
            dn_out_class, outputs_class = torch.split(outputs_class, dn_meta['dn_num_split'], dim=2)
            dn_out_coord, outputs_coord = torch.split(outputs_coord, dn_meta['dn_num_split'], dim=2)

            out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
            out['aux_outputs'] = self._set_aux_loss(outputs_class[:-1], outputs_coord[:-1])
            out['dn_output'] = {'pred_logits': dn_out_class[-1], 'pred_boxes': dn_out_coord[-1], 'dn_meta': dn_meta}
            out['dn_aux_outputs'] = self._set_aux_loss(dn_out_class, dn_out_coord)
            out['dn_meta'] = dn_meta

        else:
            out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}

        out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [{'pred_logits': a, 'pred_boxes': b} for a, b in zip(outputs_class, outputs_coord)]