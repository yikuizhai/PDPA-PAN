import torch

print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA是否可用: {torch.cuda.is_available()}")
print(f"CUDA版本: {torch.version.cuda}")
print(f"可用GPU数量: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"当前GPU: {torch.cuda.get_device_name(0)}")
    # 简单测试GPU计算
    a = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    b = torch.tensor([4.0, 5.0, 6.0], device='cuda')
    print(f"GPU计算测试: {a + b}")
    print("CUDA测试成功!")
else:
    print("无法检测到GPU!")