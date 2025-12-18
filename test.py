import torch
print(torch.__version__)  # 应输出2.3.0
print(torch.cuda.is_available())  # 应输出True
print(torch.cuda.get_device_name(0))  # 应显示"NVIDIA GeForce RTX 5060"