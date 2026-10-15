import torch

# 有 CUDA 时使用显卡，否则退回 CPU。
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建随机张量而非读取真实图片，均匀分布范围是 [0,1)。
# 四个维度分别是：批次中 4 张图、RGB 三通道、高 32、宽 32。
images = torch.rand(4, 3, 32, 32, device=device)

print("形状:", images.shape)
print("设备:", images.device)
print("显卡:", torch.cuda.get_device_name(0) if images.is_cuda else "CPU")
print("最小值:", images.min().item())
print("最大值:", images.max().item())
