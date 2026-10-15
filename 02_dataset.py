import torch
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.ToTensor()

dataset = datasets.CIFAR10(
    # 这里是本机绝对路径，其他人运行时需改为自己的路径（如 ./data）。
    root=r"D:\PythonProject\diffusion-model\my-ddpm\data",
    train=True,
    download=False,
    transform=transform,
)

automobile_indices = [i for i, label in enumerate(dataset.targets) if label == 1]
# CIFAR-10 的标签 1 是 automobile；取筛选出的第一张车查看，不是生成图片。
image, label = dataset[automobile_indices[0]]

print("汽车图片数量:", len(automobile_indices))
print("图片形状:", image.shape)
print("图片范围:", image.min().item(), image.max().item())
print("标签:", label)
print("设备:", device)
from torchvision.utils import save_image

# ToTensor 已把像素转为 [0,1]，可直接保存；image 的形状为 [3,32,32]。
save_image(image, "automobile_input.png")
print("图片已保存为: automobile_input.png")
print("类别名称:", dataset.classes[label])
