import os
import torch 
import torch.nn as nn
import torch.utils.data as DataLoader
from torchvision import datasets, transforms
from model import SimpleUNet
# simple training script for the diffusion model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
epochs = 20
lr = 1e-3
T = 1000  # Total number of timesteps

os.makedirs("checkpoints", exist_ok=True)


#load CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])
dataset = datasets.CIFAR10(root="./data", train=True, download=False, transform=transform)
dataloader = DataLoader.DataLoader(dataset, batch_size=batch_size, shuffle=True)




# add noise to the image at timestep t
betas = torch.linspace(1e-4, 0.02, T, device=device)
alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)
def q_sample(x_start, t, noise):
    alpha_bar_t = alpha_bars[t].view(-1, 1, 1, 1)
    return torch.sqrt(alpha_bar_t) * x_start + torch.sqrt(1.0 - alpha_bar_t) * noise

# 4. three components for training
model = SimpleUNet().to(device)                       
optimizer = torch.optim.AdamW(model.parameters(), lr=lr) 
criterion = nn.MSELoss()                              

# training loop
for epoch in range(epochs):
    total_loss = 0.0 
    for step, (x, _) in enumerate(dataloader):
        x = x.to(device)
        batch_size = x.size(0)
        t = torch.randint(0, T, (batch_size,), device=device).long()
        noise = torch.randn_like(x).to(device)
        x_noisy = q_sample(x, t, noise)
        
        optimizer.zero_grad()
        pred_noise = model(x_noisy, t)
        loss = criterion(pred_noise, noise)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), f"checkpoints/ddpm_epoch_{epoch+1}.pt")

print("训练完成！模型权重已保存在 checkpoints 文件夹中。")