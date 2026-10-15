"""Show one epoch-500 sample evolving from noise to an automobile."""
import argparse
# 命令行示例：python denoising_process.py --seed 789；只推理，不训练。
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import save_image

from model import SimpleUNet


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    checkpoint = root / "checkpoints_continue" / "automobile_epoch_500.pt"
    if not checkpoint.is_file():
        parser.error(f"Missing checkpoint: {checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = SimpleUNet().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()

    # 调度必须与训练一致：共 1000 个时间步，beta 线性增大。
    total_steps = 1000
    betas = torch.linspace(1e-4, 0.02, total_steps, device=device)
    alphas = 1 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    x = torch.randn((1, 3, 32, 32), device=device, generator=generator)

    # 图中标注“已完成的反向更新次数”，不是训练轮数，也不是数组索引。
    # 使用不等间隔的六个阶段：后期密集取样，更容易观察轮廓和细节的变化。
    capture_steps = {0, 500, 800, 900, 950, 1000}
    # 保存同一条生成轨迹的快照，不是分别生成六张独立图片。
    frames = [(0, x.detach().cpu().clone())]
    print(f"Device: {device}; epoch: 500; seed: {args.seed}", flush=True)
    for step in reversed(range(total_steps)):
        t = torch.full((1,), step, device=device, dtype=torch.long)
        pred_noise = model(x, t)
        alpha, beta, alpha_bar = alphas[step], betas[step], alpha_bars[step]
        noise = (torch.randn(x.shape, device=device, generator=generator)
                 if step > 0 else torch.zeros_like(x))
        # 保持原来的 beta 方差采样公式；可视化不改变生成算法。
        x = (x - ((1 - alpha) / torch.sqrt(1 - alpha_bar)) * pred_noise) / torch.sqrt(alpha)
        x = x + torch.sqrt(beta) * noise
        completed = total_steps - step
        # step=999 时完成第 1 次更新；step=0 时完成第 1000 次更新。
        if completed in capture_steps:
            frames.append((completed, x.detach().cpu().clone()))
            print(f"Captured {completed}/{total_steps}", flush=True)

    output = root / "comparisons" / ("denoising_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    # 每次保存到新目录，保留之前不同种子的生成结果。
    output.mkdir(parents=True, exist_ok=False)
    tile_size, cell_width, cell_height = 192, 216, 238
    canvas = Image.new("RGB", (6 * cell_width, cell_height + 52), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), f"Epoch 500 | seed {args.seed} | one continuous DDPM trajectory", fill="black")
    draw.text((12, 29), "Read left to right. Labels: completed reverse steps / 1000 (uneven intervals).", fill="black")
    gif_frames = []
    for index, (completed, tensor) in enumerate(frames):
        # 仅裁剪显示副本：中间状态可能超出 [-1,1]，不能把显示处理反馈到采样中。
        display = (tensor.clamp(-1, 1) + 1) / 2
        image = to_pil_image(display[0]).resize((tile_size, tile_size), Image.Resampling.NEAREST)
        left = (index % 6) * cell_width + 12
        top = (index // 6) * cell_height + 52
        canvas.paste(image, (left, top))
        label = f"{completed}/1000"
        if completed == 0:
            label += " | pure noise"
        elif completed == 1000:
            label += " | final"
        draw.text((left, top + tile_size + 8), label, fill="black")
        frame = Image.new("RGB", (216, 230), "white")
        frame.paste(image, (12, 8))
        ImageDraw.Draw(frame).text((12, 207), label, fill="black")
        gif_frames.append(frame)
        # 单帧保留原始 32×32 分辨率；拼图和 GIF 才放大显示。
        save_image(display, output / f"step_{completed:04d}.png")

    canvas.save(output / "denoising_process.png")
    # GIF 每帧显示 700 毫秒，仅用于演示，播放速度不代表真实采样耗时。
    gif_frames[0].save(output / "denoising_process.gif", save_all=True,
                       append_images=gif_frames[1:], duration=700, loop=0)
    print(f"Saved: {output / 'denoising_process.png'}", flush=True)
    print(f"Animation: {output / 'denoising_process.gif'}", flush=True)


if __name__ == "__main__":
    main()
