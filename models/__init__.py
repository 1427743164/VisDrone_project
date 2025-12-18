# 手动注册我们的魔改类
from .w_rtdetr import WRTDETR
from .backbones.presnet_wavelet import WaveletPResNet
from .necks.spectral_mamba_encoder import SpectralMambaEncoder
from .matchers.freq_sinkhorn_matcher import FrequencySinkhornMatcher
# ... 其他引用

# 如果你用的是 rtdetr_pytorch 框架，通常有一个 register 装饰器
# 或者你可以直接将它们加入到 registry 字典中，取决于你 clone 的代码库风格
# 简单粗暴的方法：确保上面的 import 语句执行了，
# 然后在 core/yaml_utils.py 里，确保它能 getattr 到这些类。
from .losses.nwd_loss import NWDLoss