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


#test the sinusoidal time embedding and block
if __name__=="__main__":
    x=torch.randn(4,3,32,32)
    t = torch.tensor([0, 50, 500, 999])
    pos_emb = SinusoidalTimeEmbedding(dim=64)
    block = Block(in_channels=3, out_channels=32, time_emb_dim=64)

    t_emb = pos_emb(t)
    out = block(x, t_emb)
    print("Input shape:", x.shape)
    print("Time embedding shape:", t_emb.shape)
    print("Output shape:", out.shape)