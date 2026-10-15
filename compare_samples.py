"""Compare automobile checkpoints with exactly the same sampling noise."""
import argparse
# 命令行示例：python compare_samples.py --epochs 350 500 --samples 64 --seed 123
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torchvision.utils import save_image

from model import SimpleUNet


@torch.no_grad()
def sample(model, device, count, batch_size, seed):
    # 每换一个权重，就从同一 seed 创建专用随机数生成器。
    # 固定批大小时，初始噪声和每步扰动均相同；只比较模型权重带来的差异。
    generator = torch.Generator(device=device).manual_seed(seed)
    betas = torch.linspace(1e-4, 0.02, 1000, device=device)
    alphas = 1 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    images = []
    for offset in range(0, count, batch_size):
        # 分批生成以降低显存占用；最后一批可能不足 batch_size 张。
        size = min(batch_size, count - offset)
        x = torch.randn((size, 3, 32, 32), device=device, generator=generator)
        for step in reversed(range(1000)):
            t = torch.full((size,), step, device=device, dtype=torch.long)
            predicted_noise = model(x, t)
            alpha, beta, alpha_bar = alphas[step], betas[step], alpha_bars[step]
            noise = (torch.randn(x.shape, device=device, generator=generator)
                     if step > 0 else torch.zeros_like(x))
            # 与原 sample.py 一致：先计算反向均值，再加 sqrt(beta_t)*z。
            x = (x - ((1 - alpha) / torch.sqrt(1 - alpha_bar)) * predicted_noise) / torch.sqrt(alpha)
            x = x + torch.sqrt(beta) * noise
        # 只对最终显示结果裁剪，搬到 CPU 后收集，不改变中间采样状态。
        images.append(((x.clamp(-1, 1) + 1) / 2).cpu())
        print(f"  Generated {offset + size}/{count}", flush=True)
    return torch.cat(images)


def main():
    root = Path(__file__).resolve().parent
    # 自动找到最新续训轮次，默认比较 200、350 和最新轮次，并去掉重复项。
    # 路径以脚本所在目录为基准；不依赖打开终端的位置。
    available = sorted((root / "checkpoints_continue").glob("automobile_epoch_*.pt"),
                       key=lambda p: int(p.stem.rsplit("_", 1)[1]))
    latest = int(available[-1].stem.rsplit("_", 1)[1]) if available else 350
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", nargs="+", type=int, default=list(dict.fromkeys([200, 350, latest])))
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.samples < 1 or args.batch_size < 1:
        parser.error("samples and batch-size must be positive")

    checkpoints = []
    # 先确认所有权重都存在，避免生成到一半才发现缺文件。
    for epoch in args.epochs:
        folder = "checkpoints" if epoch <= 200 else "checkpoints_continue"
        path = root / folder / f"automobile_epoch_{epoch}.pt"
        if not path.is_file():
            parser.error(f"Missing checkpoint: {path}")
        checkpoints.append((epoch, path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False
    # 尽量减少同一环境下的非确定性，但不保证跨平台逐像素一致。
    torch.backends.cudnn.deterministic = True
    output = root / "comparisons" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    # 每次建立带时间戳的新目录，不覆盖之前的对比结果。
    output.mkdir(parents=True, exist_ok=False)
    model = SimpleUNet().to(device)
    panels = []
    print(f"Device: {device}; output: {output}", flush=True)
    for epoch, path in checkpoints:
        print(f"Epoch {epoch}: {path}", flush=True)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        model.eval()
        images = sample(model, device, args.samples, args.batch_size, args.seed)
        grid_path = output / f"epoch_{epoch}_seed_{args.seed}.png"
        save_image(images, grid_path, nrow=8)
        # 最近邻放大方便检查像素，只改变显示尺寸，不生成额外细节。
        with Image.open(grid_path) as grid:
            enlarged = grid.convert("RGB").resize((grid.width * 3, grid.height * 3), Image.Resampling.NEAREST)
        panel = Image.new("RGB", (enlarged.width, enlarged.height + 32), "white")
        ImageDraw.Draw(panel).text((8, 8), f"Epoch {epoch} | seed {args.seed}", fill="black")
        panel.paste(enlarged, (0, 32))
        panels.append(panel)

    # 各轮次的网格横向拼接；相同格子位置对应相同的随机输入序列。
    comparison = Image.new("RGB", (sum(p.width for p in panels), max(p.height for p in panels)), "white")
    left = 0
    for panel in panels:
        comparison.paste(panel, (left, 0))
        left += panel.width
    comparison.save(output / "comparison.png")
    print(f"Done: {output / 'comparison.png'}", flush=True)


if __name__ == "__main__":
    main()
