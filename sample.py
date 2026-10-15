import torch
import torchvision
from model import SimpleUNet

# 1. 基础配置：请在项目根目录运行；本文件只生成图片，不训练模型。
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T = 1000

# 2. 准备加噪调度表（必须和训练时用的完全一模一样！）
betas = torch.linspace(1e-4, 0.02, T, device=device)
alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)

# 3. 加载我们刚刚训练好的模型权重
model = SimpleUNet().to(device)
checkpoint = "checkpoints_continue/automobile_epoch_500.pt"
model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
model.eval()  # 切换评估模式；不会关闭采样中主动生成的随机噪声。
print("已成功加载训练好的 U-Net 模型权重！")

# 4. 反向采样函数（DDPM 的逆向生成核心公式）
@torch.no_grad()
# 生成时不计算梯度，减少显存使用；与上面的 eval() 作用不同。
def sample(model, n_samples=16, image_channels=3, img_size=32):
    print(f"开始从纯噪声中无中生有生成 {n_samples} 张图片...")
    
    # 第1步：从纯正的高斯噪声开始 (Batch, 3, 32, 32)
    x = torch.randn((n_samples, image_channels, img_size, img_size), device=device)
    
    # 从 t = 999 倒数到 0
    for i in reversed(range(T)):
        t = torch.full((n_samples,), i, device=device, dtype=torch.long)
        # 当前批次全部处于同一个时间步，t 的形状为 [n_samples]。
        
        # 让 U-Net 预测当前步骤的噪声
        pred_noise = model(x, t)
        
        # 提取当前步需要的数学系数
        alpha = alphas[i]
        alpha_bar = alpha_bars[i]
        beta = betas[i]
        
        if i > 0:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)  # 最后一步不加随机扰动

        # mu = [x_t - beta_t/sqrt(1-alpha_bar_t)*epsilon_theta(x_t,t)] / sqrt(alpha_t)
        # 下一步状态 = mu + sqrt(beta_t)*z；这是 beta 方差版本，不是后验方差版本。
        x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_bar)) * pred_noise) + torch.sqrt(beta) * noise
        
        if i % 200 == 0:
            print(f"正在去噪... 剩余步数: {i}")

    # 仅在生成结束后裁剪并映射到 [0,1] 供显示，不在循环中裁剪采样状态。
    x = (x.clamp(-1, 1) + 1) / 2
    return x

# 5. 执行生成并保存结果
with torch.no_grad():
    # 一次性生成 16 张图
    generated_images = sample(model, n_samples=16)
    
    # 把这 16 张图拼成一张 4x4 的大图
    grid = torchvision.utils.make_grid(generated_images, nrow=4)
    
    output = "automobile_sample.png"
    # 重复运行会覆盖这张输出图片，不会修改已经加载的权重文件。
    torchvision.utils.save_image(grid, output)
    print(f"生成完毕！已将生成的图片保存为 {output}")
