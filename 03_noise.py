#观察噪声
import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image
# 有 CUDA 时使用显卡，否则使用 CPU。
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ToTensor 转为 [0,1]；随后归一化到 [-1,1]，与训练输入范围一致。

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
# 本机绝对路径需按实际环境修改；这里读取整个 CIFAR-10，未筛选汽车类别。
dataset = datasets.CIFAR10(root=r"D:\PythonProject\diffusion-model\my-ddpm\data", train=True, download=False, transform=transform)
x_0, _ = dataset[20]  # 取索引 20 的真实图片，忽略类别标签。
x_0 = x_0.unsqueeze(0).to(device)  # [3,32,32] -> [1,3,32,32]，补上批次维。

# 前向加噪调度，和训练脚本使用相同的 beta 起止值和步数。
T = 1000  # Total number of timesteps
betas = torch.linspace(1e-4, 0.02, T).to(device)  # 线性噪声调度。
alphas = 1 - betas
a_bars = torch.cumprod(alphas, dim=0)  # alpha 的累积乘积，表示原始信号保留比例的平方。
# q(x_t | x_0) 的均值为 sqrt(alpha_bar_t)*x_0，协方差为 (1-alpha_bar_t)*I。
def q_sample(x_0, t,noise=None):
    if noise is None:
        noise = torch.randn_like(x_0).to(device)  # 采样与原图同形状的标准高斯噪声。
    a_bar_t = a_bars[t].view(-1, 1, 1, 1)  # 把时间步系数变成可广播到图像的形状。
    return torch.sqrt(a_bar_t) * x_0 + torch.sqrt(1 - a_bar_t) * noise  # 直接得到指定时刻的带噪图。

# 各时间步从同一原图独立加噪，每次重新抽取 noise；不是连续的一条前向轨迹。
# 索引 0 已经包含微量噪声，不是完全干净的原图。

timesteps = [0, 50, 100, 200, 500, 999]
noisy_images = []
for step in timesteps:
    t=torch.tensor(step, dtype=torch.long).to(device)
    noisy_image = q_sample(x_0, t)
    noisy_image = (noisy_image.clamp(-1, 1)+1)/2  # 为显示裁剪并映射到 [0,1]。
    noisy_images.append(noisy_image)

# 沿批次维拼接各阶段，save_image 默认网格足以将这六张排列成一行。
noisy_row = torch.cat(noisy_images, dim=0)  
save_image(noisy_row, "noisy_images.png")
print("Noisy images saved as: noisy_images.png")
