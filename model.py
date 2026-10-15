'''这个部分是模型的定义，时间编码，U-Net结构，前向传播等。'''

import math 
import torch 
import torch.nn as nn

#sin time embedding function
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.dim = dim

    def forward(self,t):
        device = t.device   
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    
#convolutional block with time embedding
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.conv1=nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.conv2=nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm=nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()

    def forward(self,x,t):
        h=self.act(self.norm(self.conv1(x)))
        time_feat=self.act(self.time_mlp(t))[:, :, None, None]
        h=h+time_feat
        h=self.act(self.norm(self.conv2(h)))
        return h    

# =====================================================================
# 模块 3：完整的 U-Net 去噪网络
# 作用：输入带噪图片 x (形状 [B, 3, 32, 32]) 和时间步 t (形状 [B])
#       输出预测的噪声 (形状必须也是 [B, 3, 32, 32])
# =====================================================================
class SimpleUNet(nn.Module):
    def __init__(self, in_ch=3, time_emb_dim=64):
        super().__init__()
        
        # 1. 时间编码器：把整数 t 变成 [B, 64] 的向量
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU()
        )
        
        # 2. 左半边：下采样路径 (Encoder) —— 尺寸变小，通道变多
        self.down1 = Block(in_ch, 64, time_emb_dim)         # [B, 3, 32, 32] -> [B, 64, 32, 32]
        self.down2 = Block(64, 128, time_emb_dim)           # [B, 64, 16, 16] -> [B, 128, 16, 16]
        self.pool = nn.MaxPool2d(2)                         # 最大池化：每次把高和宽缩小一半 (32->16->8)
        
        # 3. 底部：瓶颈层 (Bottleneck) —— 看全局特征
        self.bot = Block(128, 256, time_emb_dim)            # [B, 128, 8, 8] -> [B, 256, 8, 8]
        
        # 4. 右半边：上采样路径 (Decoder) —— 尺寸放大，并拼接左边的特征
        # ConvTranspose2d(stride=2) 是反卷积，专门把高宽放大一倍 (8->16, 16->32)
        self.up_trans1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        # 注意！因为要和左边 down2 的输出 (128通道) 拼在一起，所以输入通道是 128 + 128 = 256
        self.up1 = Block(256, 128, time_emb_dim)
        
        self.up_trans2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        # 同理，和左边 down1 的输出 (64通道) 拼在一起，输入通道是 64 + 64 = 128
        self.up2 = Block(128, 64, time_emb_dim)
        
        # 5. 最终输出层：用 1x1 卷积把 64 通道还原回 RGB 的 3 通道，预测最终噪声
        self.out = nn.Conv2d(64, in_ch, kernel_size=1)

    def forward(self, x, t):
        # 先算出时间步条形码: [B] -> [B, 64]
        t_emb = self.time_mlp(t)
        
        # --- 左半边：一路往下走，顺便把特征存下来留给右边用 ---
        h1 = self.down1(x, t_emb)            # h1 形状: [B, 64, 32, 32] (存好备用)
        x_down1 = self.pool(h1)              # 缩小一半: [B, 64, 16, 16]
        
        h2 = self.down2(x_down1, t_emb)      # h2 形状: [B, 128, 16, 16] (存好备用)
        x_down2 = self.pool(h2)              # 再缩小一半: [B, 128, 8, 8]
        
        # --- 底部：最深处 ---
        b = self.bot(x_down2, t_emb)         # b 形状: [B, 256, 8, 8]
        
        # --- 右半边：一路往上放大，并把左边存好的 h2、h1 拼回来 ---
        u1 = self.up_trans1(b)               # 放大回 16x16: [B, 128, 16, 16]
        u1 = torch.cat([u1, h2], dim=1)      # 跳跃连接！在通道维度(dim=1)拼接: [B, 256, 16, 16]
        u1 = self.up1(u1, t_emb)             # 融合特征: [B, 128, 16, 16]
        
        u2 = self.up_trans2(u1)              # 放大回 32x32: [B, 64, 32, 32]
        u2 = torch.cat([u2, h1], dim=1)      # 跳跃连接！拼接 h1: [B, 128, 32, 32]
        u2 = self.up2(u2, t_emb)             # 融合特征: [B, 64, 32, 32]
        
        # 输出预测的噪声: [B, 3, 32, 32]
        return self.out(u2)


# =====================================================================
# 更新底部的测试代码：测试整个 SimpleUNet
# =====================================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    x = torch.randn(4, 3, 32, 32, device=device)
    t = torch.tensor([0, 50, 500, 999], device=device)
    
    model = SimpleUNet().to(device)
    pred_noise = model(x, t)
    
    print("输入带噪图的形状:", x.shape)
    print("U-Net 预测噪声的形状:", pred_noise.shape)
    print("模型总参数量:", sum(p.numel() for p in model.parameters()))