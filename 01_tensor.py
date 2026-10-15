import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

images = torch.rand(4, 3, 32, 32, device=device)

print("形状:", images.shape)
print("设备:", images.device)
print("显卡:", torch.cuda.get_device_name(0) if images.is_cuda else "CPU")
print("最小值:", images.min().item())
print("最大值:", images.max().item())