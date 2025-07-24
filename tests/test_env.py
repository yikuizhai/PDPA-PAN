import os
import torch
import mmcv
import numpy as np
from PIL import Image

# 打印环境信息
print("环境信息:")
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA是否可用: {torch.cuda.is_available()}")
print(f"CUDA版本: {torch.version.cuda}")
print(f"mmcv版本: {mmcv.__version__}")
print(f"Pillow版本: {Image.__version__}")

# 测试图像加载和处理
print("\n测试图像处理:")
try:
    # 使用mmcv加载图像
    img_path = 'demo/demo.jpg'  # OBBDetection中自带的示例图像
    if not os.path.exists(img_path):
        print(f"找不到图像文件: {img_path}")
    else:
        img = mmcv.imread(img_path)
        print(f"图像加载成功，形状: {img.shape}")
        
        # 简单的图像变换测试
        img_resized = mmcv.imresize(img, (300, 300))
        print(f"图像调整大小成功，新形状: {img_resized.shape}")
        
        # 保存测试
        mmcv.imwrite(img_resized, 'test_output.jpg')
        print("图像保存成功")
except Exception as e:
    print(f"图像处理出错: {e}")

print("\n环境测试完成")