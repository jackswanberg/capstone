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
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=64, sampler=sampler, num_workers=4)

    # === Model ===
    model = UNet(...)  # Replace with your DDPM UNet
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
