import os
import math
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from tqdm import tqdm 
from PIL import Image
from typing import Tuple

@dataclass
class Config:
    data_root: str = "./data"
    runs_root: str = "./runs"
    batch_size: int = 128
    num_workers: int = 4
    epochs: int = 15
    lr: float = 1e-4
    beta: float = 0.25                # commitment loss 权重
    image_size: int = 28
    in_channels: int = 1              # MNIST 是单通道
    embedding_dim: int = 64           # 码本向量维度 D
    num_codes: int = 512              # 码本大小 K
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_interval: int = 100
    save_interval: int = 1


cfg = Config()
os.makedirs(cfg.runs_root, exist_ok=True)
torch.manual_seed(cfg.seed)
random.seed(cfg.seed)

class Encoder(nn.Module):
    """
    输入: Bx1x28x28 -> 输出: BxDx7x7
    """
    def __init__(self, in_ch=1, hidden=128, embedding_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 4, stride=2, padding=1),  # 28->14
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden, hidden, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden, hidden, 4, stride=2, padding=1), # 14->7
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden, embedding_dim, 1)
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    """
    输入: BxDx7x7 -> 输出: Bx1x28x28
    """
    def __init__(self, out_ch=1, hidden=128, embedding_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(embedding_dim, hidden, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(hidden, hidden, 4, stride=2, padding=1),  # 7->14
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(hidden, out_ch, 4, stride=2, padding=1),  # 14->28
            nn.Tanh()  # 因为输入做了 Normalize((0.5,),(0.5,))，目标是 [-1,1]
        )

    def forward(self, z):
        return self.net(z)
    
class VectorQuantizer(nn.Module):
    """
    - codebook: nn.Embedding(K, D)
    - forward:
        1) flatten z_e -> (BHW, D)
        2) 与 codebook.weight (K, D) 计算 L2 距离，取最近索引
        3) 用 embedding 索引回 (BHW, D)，再 reshape 回 BxDxHxW
        4) 计算三项损失（recon 在外部），返回 z_q 与 stats
    - perplexity:
        使用 one-hot 选择的平均分布计算 exp(H(p))
    """
    def __init__(self, num_codes=512, embedding_dim=64, beta=0.25):
        super().__init__()
        self.num_codes = num_codes
        self.embedding_dim = embedding_dim
        self.beta = beta

        self.codebook = nn.Embedding(num_codes, embedding_dim)
        # 初始化为均匀分布（推荐：正态或均匀，小尺度）
        self.codebook.weight.data.uniform_(-1.0 / num_codes, 1.0 / num_codes)

    @torch.no_grad()
    def _compute_distances(self, z_flat):
        # z_flat: (N, D); codebook: (K, D)
        # ||z - e||^2 = ||z||^2 + ||e||^2 - 2 z·e
        z_sq = (z_flat ** 2).sum(dim=1, keepdim=True)            # (N,1)
        e_sq = (self.codebook.weight ** 2).sum(dim=1)            # (K,)
        ze = z_flat @ self.codebook.weight.t()                   # (N,K)
        dist = z_sq + e_sq.unsqueeze(0) - 2 * ze
        return dist

    def forward(self, z_e):
        # z_e: BxDxHxW
        B, D, H, W = z_e.shape
        assert D == self.embedding_dim

        # 1) flatten
        z_flat = z_e.permute(0, 2, 3, 1).contiguous().view(-1, D)  # (BHW, D)

        # 2) 最近码本索引
        with torch.no_grad():
            dist = self._compute_distances(z_flat)                 # (BHW, K)
            indices = torch.argmin(dist, dim=1)                    # (BHW,)
            encodings = F.one_hot(indices, self.num_codes).type(z_flat.dtype)  # (BHW, K)

        # 3) 查表回量化向量
        z_q_flat = self.codebook(indices)                          # (BHW, D)
        z_q = z_q_flat.view(B, H, W, D).permute(0, 3, 1, 2).contiguous()  # BxDxHxW

        # 4) straight-through trick: 把 z_q 当作 z_e 的前向，但反向给 z_e 传梯度
        # 公式: z_q + (z_e - z_q).detach()
        z_q_st = z_e + (z_q - z_e).detach()

        # 5) 码本与承诺损失
        # codebook loss: ||sg[z_e] - e||^2
        codebook_loss = F.mse_loss(z_q.detach(), z_e)
        # commitment loss: ||z_e - sg[e]||^2
        commitment_loss = F.mse_loss(z_q, z_e.detach())
        vq_loss = codebook_loss + self.beta * commitment_loss

        # 6) perplexity（衡量码本使用多样性）
        avg_probs = encodings.mean(dim=0)                          # (K,)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return z_q_st, vq_loss, indices.view(B, H, W), perplexity
    
class VQVAE(nn.Module):
    def __init__(self, in_ch=1, embedding_dim=64, num_codes=512, beta=0.25, hidden=128):
        super().__init__()
        self.encoder = Encoder(in_ch=in_ch, hidden=hidden, embedding_dim=embedding_dim)
        self.quantizer = VectorQuantizer(num_codes=num_codes, embedding_dim=embedding_dim, beta=beta)
        self.decoder = Decoder(out_ch=in_ch, hidden=hidden, embedding_dim=embedding_dim)

    def forward(self, x):
        z_e = self.encoder(x)                # 连续潜表示
        z_q, vq_loss, indices, ppl = self.quantizer(z_e)  # 量化
        x_rec = self.decoder(z_q)            # 重建
        return x_rec, vq_loss, indices, ppl
    
def get_loaders(batch_size=128, num_workers=4):
    tfm = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),        # => [0,1]
        # 去掉 Normalize((0.5,), (0.5,))
    ])
    train_set = datasets.MNIST(root=cfg.data_root, train=True, download=True, transform=tfm)
    test_set  = datasets.MNIST(root=cfg.data_root, train=False, download=True, transform=tfm)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader

def save_reconstructions(model, loader, epoch, device, max_samples=16):
    model.eval()
    x, _ = next(iter(loader))
    x = x.to(device)[:max_samples]
    with torch.no_grad():
        x_rec, _, _, _ = model(x)

    # 保证可视化范围正确
    x_vis = torch.clamp(x.detach().cpu(), 0.0, 1.0)
    xrec_vis = torch.clamp(x_rec.detach().cpu(), 0.0, 1.0)

    # 分别做成单行网格，再上下拼接成最终图片
    top = utils.make_grid(x_vis, nrow=max_samples, padding=2)
    bottom = utils.make_grid(xrec_vis, nrow=max_samples, padding=2)
    grid = torch.cat([top, bottom], dim=1)  # dim=1 表示按高度方向拼接

    save_path = os.path.join(cfg.runs_root, f"recon_epoch_{epoch:02d}.png")
    utils.save_image(grid, save_path)
    print(f"[Eval] Saved reconstructions to {save_path}")
    # 便于排查：打印数值范围
    print(f"[Debug] x    range: [{x_vis.min():.3f}, {x_vis.max():.3f}]")
    print(f"[Debug] x_rec range: [{xrec_vis.min():.3f}, {xrec_vis.max():.3f}]")


def train():
    device = cfg.device
    train_loader, test_loader = get_loaders(cfg.batch_size, cfg.num_workers)

    model = VQVAE(
        in_ch=cfg.in_channels,
        embedding_dim=cfg.embedding_dim,
        num_codes=cfg.num_codes,
        beta=cfg.beta,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    global_step = 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = {"recon": 0.0, "vq": 0.0, "ppl": 0.0}

        # 用 tqdm 包装 dataloader
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}/{cfg.epochs}")
        for i, (x, _) in pbar:
            x = x.to(device)

            x_rec, vq_loss, _, ppl = model(x)
            recon_loss = F.l1_loss(x_rec, x)  # L1 重建
            loss = recon_loss + vq_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            running["recon"] += recon_loss.item()
            running["vq"] += vq_loss.item()
            running["ppl"] += ppl.item()
            global_step += 1

            # 更新进度条后缀
            avg_recon = running["recon"] / (i + 1)
            avg_vq    = running["vq"] / (i + 1)
            avg_ppl   = running["ppl"] / (i + 1)
            pbar.set_postfix({
                "Recon": f"{avg_recon:.4f}",
                "VQ": f"{avg_vq:.4f}",
                "Perplexity": f"{avg_ppl:.2f}"
            })

        # 每轮保存重建可视化
        if epoch % cfg.save_interval == 0 or epoch == cfg.epochs:
            save_reconstructions(model, test_loader, epoch, device)

            # 保存权重
            ckpt_path = os.path.join(cfg.runs_root, f"vqvae_epoch_{epoch:02d}.pt")
            torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, ckpt_path)
            print(f"[Save] checkpoint to {ckpt_path}")
            
if __name__ == "__main__":
    train()