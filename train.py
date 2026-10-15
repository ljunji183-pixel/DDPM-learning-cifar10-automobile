import os
import torch 
import torch.nn as nn
import torch.utils.data as DataLoader
from torch.utils.data import Subset
from torchvision import datasets, transforms
from model import SimpleUNet

# 从头训练汽车单类别 DDPM；当前不是从第 200 轮继续训练的脚本。
# 请在项目根目录执行。重复运行可能覆盖 checkpoints 中的同名权重。
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
epochs = 200
lr = 2e-3
T = 1000  # 扩散时间步数，和训练轮数 epochs 是两个不同的概念。

os.makedirs("checkpoints", exist_ok=True)


# 读取 CIFAR-10：ToTensor 将像素转为 [0,1]，Normalize 再映射到 [-1,1]。
# download=False 表示只读取本地数据，首次使用需先下载到 ./data。
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])
dataset = datasets.CIFAR10(root="./data", train=True, download=False, transform=transform)
automobile_indices = [i for i, label in enumerate(dataset.targets) if label == 1]
# 标签 1 对应 automobile。这里只筛选训练数据，模型本身没有接收类别标签。
dataset = Subset(dataset, automobile_indices)
print(f"Automobile training images: {len(dataset)}")
dataloader = DataLoader.DataLoader(dataset, batch_size=batch_size, shuffle=True)




# 前向加噪调度：alpha_t = 1-beta_t，alpha_bar_t 为 alpha 的累积乘积。
# 数组索引 0 对应第一次加噪，索引 999 对应第 1000 次。
betas = torch.linspace(1e-4, 0.02, T, device=device)
alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)
def q_sample(x_start, t, noise):
    # 一步直接采样任意时间步的带噪图，无需循环逐步加噪：
    # x_t = sqrt(alpha_bar_t)*x_0 + sqrt(1-alpha_bar_t)*epsilon。
    # 每张图的 t 可以不同；[B,1,1,1] 让系数广播到该图所有通道和像素。
    alpha_bar_t = alpha_bars[t].view(-1, 1, 1, 1)
    return torch.sqrt(alpha_bar_t) * x_start + torch.sqrt(1.0 - alpha_bar_t) * noise

# 三个训练组件：预测噪声的网络、更新参数的优化器、比较噪声的损失函数。
model = SimpleUNet().to(device)                       
optimizer = torch.optim.AdamW(model.parameters(), lr=lr) 
criterion = nn.MSELoss()                              

# 每个 epoch 遍历一遍筛选后的汽车数据；每个 batch 重新抽取时间步和噪声。
for epoch in range(epochs):
    total_loss = 0.0 
    for step, (x, _) in enumerate(dataloader):
        x = x.to(device)
        batch_count = x.size(0)
        t = torch.randint(0, T, (batch_count,), device=device).long()
        # epsilon ~ N(0,I)，形状与图片完全相同 [B,3,32,32]。
        noise = torch.randn_like(x)
        x_noisy = q_sample(x, t, noise)
        
        optimizer.zero_grad()
        # 输入带噪图和时间步，目标是本次真实加入的 noise，而不是原图 x。
        pred_noise = model(x_noisy, t)
        loss = criterion(pred_noise, noise)
        loss.backward()
        # backward 计算梯度；step 才根据梯度修改模型参数。
        optimizer.step()
        
        total_loss += loss.item()

    # 按 batch 平均记录 MSE。它不是 FID，也不能直接等同于视觉质量。
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

    if (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
        # 只保存模型参数，不包含优化器状态或随机数状态，无法精确恢复训练现场。
        torch.save(model.state_dict(), f"checkpoints/automobile_epoch_{epoch+1}.pt")

print("训练完成！模型权重已保存在 checkpoints 文件夹中。")
