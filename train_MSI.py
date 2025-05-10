import torch
from torchvision.datasets import CIFAR100
from torchvision import transforms
from torch.utils.data import Dataset
import torch.multiprocessing as mp


import torch.distributed as dist
import os, torch, torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms
import numpy as np
import os
import pickle

class CIFAR100LongTail(Dataset):
    def __init__(self, root, phase='train', imbalance_factor=0.01, transform=None):
        self.root = root
        self.phase = phase
        self.transform = transform
        self.num_classes = 100
        self.imgs, self.labels = self._make_longtail(imbalance_factor)

    def _make_longtail(self, imbalance_factor):
        cifar = CIFAR100(self.root, train=(self.phase == 'train'), download=True)
        data, targets = cifar.data, np.array(cifar.targets)
        cls_num = self.num_classes

        # Long tail class distribution
        cls_counts = []
        img_per_cls_max = len(targets) // cls_num
        for cls_idx in range(cls_num):
            num = img_per_cls_max * (imbalance_factor ** (cls_idx / (cls_num - 1)))
            cls_counts.append(int(num))

        new_data, new_targets = [], []
        for cls_idx, cls_count in enumerate(cls_counts):
            idx = np.where(targets == cls_idx)[0]
            np.random.shuffle(idx)
            sel = idx[:cls_count]
            new_data.append(data[sel])
            new_targets.extend([cls_idx] * cls_count)

        new_data = np.concatenate(new_data)
        return new_data, new_targets

    def __getitem__(self, index):
        img, target = self.imgs[index], self.labels[index]
        img = transforms.ToPILImage()(img)
        if self.transform:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.labels)

# ================================
#         CIFAR10 Dataset
# ================================
class ToMinusOneToOne:
    def __call__(self, x):
        return x * 2. - 1.
    
# ================================
#       Linear Beta Schedule
# ================================
def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)

# ================================
#     Beefier UNet-like Model
# ================================
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb).to(t.device)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()
        print(in_channels)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        self.block1 = nn.Sequential(
            nn.GroupNorm(min(4,in_channels), in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(min(4,out_channels), out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t):
        h = self.block1(x)
        time_emb = self.time_mlp(t).view(t.shape[0], -1, 1, 1)
        h = h + time_emb
        h = self.block2(h)
        return h + self.residual_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, num_classes = 100, time_emb_dim=256):
        super().__init__()
        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU()
        )
        self.label_emb = nn.Embedding(num_classes, time_emb_dim)


        # Down
        self.down1 = ResidualBlock(in_channels, base_channels, time_emb_dim)
        self.down2 = ResidualBlock(base_channels, base_channels * 2, time_emb_dim)
        self.pool = nn.MaxPool2d(2)

        # Middle
        self.middle = ResidualBlock(base_channels * 2, base_channels * 2, time_emb_dim)

        # Up
        self.up1 = ResidualBlock(base_channels * 2 + base_channels * 2, base_channels, time_emb_dim)
        self.up2 = ResidualBlock(base_channels + base_channels, in_channels, time_emb_dim)

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x, label, t):
        label_emb = self.label_emb(label)
        t_emb = self.time_embedding(t)
        t_emb = t_emb + label_emb


        # Down
        x1 = self.down1(x, t_emb)

        x2 = self.pool(x1)
        x2 = self.down2(x2, t_emb)

        x3 = self.pool(x2)

        # Middle
        x3 = self.middle(x3, t_emb)


        # Up
        x = self.upsample(x3)
        x = self.up1(torch.cat([x, x2], dim=1), t_emb)
        x = self.upsample(x)
        x = self.up2(torch.cat([x, x1], dim=1), t_emb)

        return x

# ================================
#      Helper Function
# ================================
def extract(a, t, x_shape):
    return a.gather(-1, t.to(torch.int64)).reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))

def student_t_nll(x, mu, nu=4.0):
    return torch.log(1 + ((x - mu) ** 2) / nu).mean()

# ================================
#     Student-t Based DDPM
# ================================
class StudentTDDPM:
    def __init__(self, model, betas, nu=4.0):
        self.model = model
        self.nu = nu
        self.device = next(model.parameters()).device

        self.timesteps = len(betas)
        self.betas = betas.to(self.device)
        self.alphas = 1. - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x_start, t):
        alpha_bar = extract(self.alpha_bars, t, x_start.shape)
        noise = StudentT(df=self.nu, loc=0, scale=1).sample(x_start.shape).to(self.device)
        x_noisy = torch.sqrt(alpha_bar) * x_start + torch.sqrt(1 - alpha_bar) * noise
        return x_noisy, noise  # Return both!

    def p_losses(self, x_start, label, t):
        x_noisy, noise = self.q_sample(x_start, t)
        predicted = self.model(x_noisy, label, t)
        return student_t_nll(noise, predicted, self.nu)


    def p_sample(self, x, label, t):
        betas_t = extract(self.betas, t, x.shape)
        alphas_t = extract(self.alphas,t,x.shape)
        alpha_bar_t = extract(self.alpha_bars, t, x.shape)
        predicted_noise = self.model(x, label, t)

        mean = (1 / torch.sqrt(alphas_t)) * (
                x - ((1 - alphas_t) / torch.sqrt(1 - alpha_bar_t)) * predicted_noise)

        if t[0] == 0:
            return mean
        noise = StudentT(df=self.nu, loc=0, scale=1).sample(x.shape).to(self.device)

        return torch.sqrt(alphas_t) * mean + torch.sqrt(betas_t) * noise

    def sample(self, label, shape):
        x = torch.randn(shape).to(self.device)
        for t in reversed(range(self.timesteps)):
            t_batch = torch.tensor([t] * shape[0]).to(self.device)
            x = self.p_sample(x, label, t_batch)
        return x.clamp(-1, 1)


def main(rank, world_size):


    # === DDP Init ===
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    # === Dataset ===
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276))
    ])
    dataset = CIFAR100LongTail(root='./data', imbalance_factor=0.01, transform=transform)
    num_classes = len(dataset.classes)

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=64, sampler=sampler, num_workers=4)

    # === Model ===
    model = UNet(in_channels=3, base_channels=128,num_classes=num_classes).to(device)
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])

    # === Training ===
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    scheduler = GeneralizedDDPMScheduler(...)  # Your custom scheduler

    for epoch in range(200):
        sampler.set_epoch(epoch)
        for x, y in dataloader:
            x = x.to(rank)
            t = torch.randint(0, scheduler.num_train_timesteps, (x.size(0),), device=x.device)
            noise = scheduler.sample_noise(x.shape, x.device)
            x_noisy = scheduler.add_noise(x, noise, t)

            # Forward pass and loss
            out = model(x_noisy, t, y)
            loss = ((out - noise)**2).mean()  # or your NLL

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size,), nprocs=world_size)
