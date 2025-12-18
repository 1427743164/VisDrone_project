""""by lyuwenyu
"""

import torch
import torch.nn as nn

import torchvision
torchvision.disable_beta_transforms_warning()

# [关键升级] 引入新版 API tv_tensors，替代已废弃的 datapoints
from torchvision import tv_tensors

import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F

from PIL import Image
from typing import Any, Dict, List, Optional

from src.core import register, GLOBAL_CONFIG


__all__ = ['Compose', ]


# 直接注册官方实现的变换，逻辑完全一致
RandomPhotometricDistort = register(T.RandomPhotometricDistort)
RandomZoomOut = register(T.RandomZoomOut)
RandomHorizontalFlip = register(T.RandomHorizontalFlip)
Resize = register(T.Resize)
RandomCrop = register(T.RandomCrop)
Normalize = register(T.Normalize)


# =========================================================================
# [兼容层] 手动定义这些类，确保旧配置文件(YAML)能直接运行，且逻辑正确
# =========================================================================

@register
class ToImageTensor(T.ToImage):
    """
    将 PIL Image 或 ndarray 转换为 tv_tensors.Image (Tensor)。
    替代旧版的 ToTensor。注意：它只转类型，不除以255。
    """
    def __init__(self):
        super().__init__()

@register
class ConvertDtype(T.ToDtype):
    """
    将数值类型转换（如 uint8 -> float32），并进行归一化（scale=True）。
    [关键] 必须开启 scale=True，否则 0-255 的数值不会变成 0-1，导致模型无法训练。
    """
    def __init__(self, dtype=torch.float32, scale=True):
        super().__init__(dtype=dtype, scale=scale)

@register
class SanitizeBoundingBox(T.SanitizeBoundingBoxes):
    """
    清理无效的边界框（如宽高为负或极小的框），防止 Loss 计算 NaN。
    """
    pass

# =========================================================================


@register
class Compose(T.Compose):
    def __init__(self, ops) -> None:
        transforms = []
        if ops is not None:
            for op in ops:
                if isinstance(op, dict):
                    name = op.pop('type')
                    # 动态加载注册的类
                    transfom = getattr(GLOBAL_CONFIG[name]['_pymodule'], name)(**op)
                    transforms.append(transfom)
                elif isinstance(op, nn.Module):
                    transforms.append(op)
                else:
                    raise ValueError(f'Unknown transform type: {type(op)}')
        else:
            transforms =[EmptyTransform(), ]

        super().__init__(transforms=transforms)


@register
class EmptyTransform(T.Transform):
    def __init__(self, ) -> None:
        super().__init__()

    def forward(self, *inputs):
        inputs = inputs if len(inputs) > 1 else inputs[0]
        return inputs


@register
class PadToSize(T.Pad):
    # [关键升级] 更新支持的类型列表，使用 tv_tensors
    _transformed_types = (
        Image.Image,
        tv_tensors.Image,
        tv_tensors.Video,
        tv_tensors.Mask,
        tv_tensors.BoundingBoxes,
    )

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        # 计算需要的 padding 大小 (将图像 padding 到固定尺寸)
        sz = F.get_spatial_size(flat_inputs[0])
        h, w = self.spatial_size[0] - sz[0], self.spatial_size[1] - sz[1]
        self.padding = [0, 0, w, h] # left, top, right, bottom
        return dict(padding=self.padding)

    def make_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        return self._get_params(flat_inputs)

    def __init__(self, spatial_size, fill=0, padding_mode='constant') -> None:
        if isinstance(spatial_size, int):
            spatial_size = (spatial_size, spatial_size)

        self.spatial_size = spatial_size
        super().__init__(0, fill, padding_mode)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        # 根据输入类型选择填充值 (如 mask 填 0 或 255)
        fill = self._fill[type(inpt)]
        padding = params['padding']
        return F.pad(inpt, padding=padding, fill=fill, padding_mode=self.padding_mode)

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self._transform(inpt, params)

    def __call__(self, *inputs: Any) -> Any:
        outputs = super().forward(*inputs)
        # 如果输出包含字典（通常是 target），把 padding 信息记录进去，供后处理使用
        if len(outputs) > 1 and isinstance(outputs[1], dict):
            outputs[1]['padding'] = torch.tensor(self.padding)
        return outputs


@register
class RandomIoUCrop(T.RandomIoUCrop):
    def __init__(self, min_scale: float = 0.3, max_scale: float = 1, min_aspect_ratio: float = 0.5, max_aspect_ratio: float = 2, sampler_options: Optional[List[float]] = None, trials: int = 40, p: float = 1.0):
        super().__init__(min_scale, max_scale, min_aspect_ratio, max_aspect_ratio, sampler_options, trials)
        self.p = p

    def __call__(self, *inputs: Any) -> Any:
        # 增加概率控制逻辑
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]

        return super().forward(*inputs)


@register
class ConvertBox(T.Transform):
    _transformed_types = (
        tv_tensors.BoundingBoxes, # [关键升级]
    )
    def __init__(self, out_fmt='', normalize=False) -> None:
        super().__init__()
        self.out_fmt = out_fmt
        self.normalize = normalize

        # 映射新版 API 的格式定义
        self.data_fmt = {
            'xyxy': tv_tensors.BoundingBoxFormat.XYXY,
            'cxcywh': tv_tensors.BoundingBoxFormat.CXCYWH
        }

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if self.out_fmt:
            # [关键] 新版 API 中 spatial_size 变更为 canvas_size
            canvas_size = inpt.canvas_size
            in_fmt = inpt.format.value.lower()
            # 执行转换
            inpt = torchvision.ops.box_convert(inpt, in_fmt=in_fmt, out_fmt=self.out_fmt)
            # 重新封装为 tv_tensors，保留 canvas_size
            inpt = tv_tensors.BoundingBoxes(inpt, format=self.data_fmt[self.out_fmt], canvas_size=canvas_size)

        if self.normalize:
            # 归一化处理 (除以宽高)
            inpt = inpt / torch.tensor(inpt.canvas_size[::-1]).tile(2)[None]

        return inpt

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self._transform(inpt, params)