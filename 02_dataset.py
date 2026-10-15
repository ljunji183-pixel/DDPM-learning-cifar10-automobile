import torch
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.ToTensor()

dataset = datasets.CIFAR10(
    root=r"D:\PythonProject\diffusion-model\my-ddpm\data",
    train=True,
    download=False,
    transform=transform,
)

image, label = dataset[20]

print("数据集大小:", len(dataset))
print("图片形状:", image.shape)
print("图片范围:", image.min().item(), image.max().item())
print("标签:", label)
print("设备:", device)
from torchvision.utils import save_image

save_image(image, "sample.png")
print("图片已保存为: sample.png")
print("类别名称:", dataset.classes[label])