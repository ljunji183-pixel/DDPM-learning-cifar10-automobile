import torch
import torchvision
from model import SimpleUNet
import os

# 1. 基础配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T = 1000

# 2. 准备加噪调度表（必须和训练时用的完全一模一样！）
betas = torch.linspace(1e-4, 0.02, T, device=device)
alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)

# 3. 加载我们刚刚训练好的模型权重
model = SimpleUNet().to(device)
# 加载第 20 轮保存的权重文件
model.load_state_dict(torch.load("checkpoints/ddpm_epoch_20.pt", map_location=device))
model.eval()  # 切换到评估模式（关闭 Dropout 等随机行为）
print("已成功加载训练好的 U-Net 模型权重！")

# 4. 反向采样函数（DDPM 的逆向生成核心公式）
@torch.no_grad()
def sample(model, n_samples=16, image_channels=3, img_size=32):
    print(f"开始从纯噪声中无中生有生成 {n_samples} 张图片...")
    
    # 第1步：从纯正的高斯噪声开始 (Batch, 3, 32, 32)
    x = torch.randn((n_samples, image_channels, img_size, img_size), device=device)
    
    # 从 t = 999 倒数到 0
    for i in reversed(range(T)):
        t = torch.full((n_samples,), i, device=device, dtype=torch.long)
        
        # 让 U-Net 预测当前步骤的噪声
        pred_noise = model(x, t)
        
        # 提取当前步需要的数学系数
        alpha = alphas[i]
        alpha_bar = alpha_bars[i]
        beta = betas[i]
        
        if i > 0:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x) # 最后一步不加随机扰动
            
        # 核心逆向递推公式：根据预测的噪声，把图片往清澈的方向推一步
        x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_bar)) * pred_noise) + torch.sqrt(beta) * noise
        
        if i % 200 == 0:
            print(f"正在去噪... 剩余步数: {i}")

    # 把像素值从 [-1, 1] 压回 [0, 1] 方便保存成图片
    x = (x.clamp(-1, 1) + 1) / 2
    return x

# 5. 执行生成并保存结果
with torch.no_grad():
    # 一次性生成 16 张图
    generated_images = sample(model, n_samples=16)
    
    # 把这 16 张图拼成一张 4x4 的大图
    grid = torchvision.utils.make_grid(generated_images, nrow=4)
    
    # 保存为 sample_result.png
    torchvision.utils.save_image(grid, "sample_result.png")
    print("生成完毕！已将生成的图片保存为 sample_result.png")