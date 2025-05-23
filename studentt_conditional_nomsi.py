import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR100
from torchvision import datasets, transforms
from torch.distributions.studentT import StudentT
import argparse
import numpy as np
import os

from GammaDDPM import GammaDDPM, FrechetDDPM
from models import UNet2
from DhariwalUNet import DhariwalUNet
from diffusers import UNet2DConditionModel
import diffusers

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
#     Basic UNet-like Model
# ================================
class ConvNet(nn.Module):
    def __init__(self, channels=3):
        super().__init__()
        self.conv1 = nn.Conv2d(channels + 1, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, channels, 3, padding=1)

    def forward(self, x, t):
        # Broadcast time embedding
        t_embed = t[:, None, None, None].float() / 1000
        t_embed = t_embed.expand(-1, 1, x.shape[2], x.shape[3])
        x = torch.cat([x, t_embed], dim=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return self.conv4(x)


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
    def __init__(self, in_channels=3, base_channels=64, time_emb_dim=256, num_classes = 10):
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
#      Visualization Functions
# ================================
def visualize_denoising(ddpm, label, steps_to_show=[0, 25, 50, 75, 99]):
    x = torch.randn((10, 3, 32, 32)).to(ddpm.device)
    label = torch.tensor([label], device=ddpm.device)
    # label = torch.arange(0,10,device=ddpm.device,dtype=torch.long)
    images = []

    for t in reversed(range(ddpm.timesteps)):
        t_batch = torch.tensor([t]).to(ddpm.device)
        x = ddpm.p_sample(x, label, t_batch)
        if t in steps_to_show:
            img = torch.clamp((x[0] + 1) / 2, 0, 1).cpu()
            images.append(img)

    grid = torch.stack(images)
    grid = torchvision.utils.make_grid(grid, nrow=len(images))
    plt.imshow(grid.permute(1, 2, 0))
    plt.title("Denoising Progression")
    plt.axis("off")
    plt.show()

def visualize_one_sample_per_class(ddpm):
    ddpm.model.eval()
    with torch.no_grad():
        class_labels = torch.arange(10).to(ddpm.device)
        samples = ddpm.sample(class_labels, (10, 3, 32, 32)).cpu()
        samples = (samples + 1) / 2  # De-normalize to [0, 1]

        class_names = [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"
        ]

        fig, axes = plt.subplots(1, 10, figsize=(15, 2))
        for i, ax in enumerate(axes):
            img = samples[i]
            ax.imshow(img.permute(1, 2, 0))
            ax.axis("off")
            ax.set_title(class_names[i], fontsize=8)
        plt.tight_layout()
        plt.show()


# ================================
#     Gaussian Based DDPM
# ================================
class DDPM:
    def __init__(self, model, betas):
        self.model = model
        self.device = next(model.parameters()).device

        self.timesteps = len(betas)
        self.betas = betas.to(self.device)
        self.alphas = 1. - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x_start, t):
        alpha_bar = extract(self.alpha_bars, t, x_start.shape)
        noise = torch.randn_like(x_start).to(self.device)
        x_noisy = torch.sqrt(alpha_bar) * x_start + torch.sqrt(1 - alpha_bar) * noise
        return x_noisy, noise  # Return both!

    def p_losses(self, x_start, label, t):
        x_noisy, noise = self.q_sample(x_start, t)
        encoder_hidden_states = torch.zeros((x_start.shape[0], 1, 1280), device=self.device)
        predicted = self.model(x_noisy, timestep=t, class_labels=label,encoder_hidden_states=encoder_hidden_states)  # <- fixed
        predicted = predicted.sample
        loss = nn.functional.smooth_l1_loss(predicted, noise)
        return loss


    def p_sample(self, x, label, t):
        betas_t = extract(self.betas, t, x.shape)
        alphas_t = extract(self.alphas,t,x.shape)
        alpha_bar_t = extract(self.alpha_bars, t, x.shape)
        encoder_hidden_states = torch.zeros((x.shape[0], 1, 1280), device=self.device)
        predicted_noise = self.model(x, timestep=t, class_labels=label,encoder_hidden_states=encoder_hidden_states)
        predicted_noise = predicted_noise.sample
        mean = (1 / torch.sqrt(alphas_t)) * (
                x - ((1 - alphas_t) / torch.sqrt(1 - alpha_bar_t)) * predicted_noise)

        if t[0] == 0:
            return mean
        noise = torch.randn_like(x).to(self.device)
        return torch.sqrt(alphas_t) * mean + torch.sqrt(betas_t) * noise

    def sample(self, label, shape):
        x = torch.randn(shape).to(self.device)
        for t in reversed(range(self.timesteps)):
            t_batch = torch.tensor([t] * shape[0]).to(self.device)
            x = self.p_sample(x, label, t_batch)
        return x.clamp(-1, 1)


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

# ================================
#        Training Loop
# ================================
def train_ddpm(model, ddpm, dataloader, val_dataloader, epochs=50):
    print("Entered training")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                               mode='min',
                                               factor=0.3,
                                               patience=10)
    print(scheduler.get_last_lr())
    model.train()
    epoch_loss = 0
    for epoch in tqdm(range(epochs)):
        train_loss = 0
        val_loss = 0
        model.train()
        for batch in dataloader:
            x = batch[0].to(ddpm.device)
            label = batch[1].to(ddpm.device)
            t = torch.randint(0, ddpm.timesteps, (x.size(0),), device=ddpm.device)
            assert label.dtype == torch.long, f"Bad label dtype: {label.dtype}"
            assert (label >= 0).all() and (label < 100).all(), f"Label out of range: {label}"
            loss = ddpm.p_losses(x, label, t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        for batch in val_dataloader:
            x = batch[0].to(ddpm.device)
            label = batch[1].to(ddpm.device)
            t = torch.randint(0, ddpm.timesteps, (x.size(0),), device=ddpm.device)
            assert label.dtype == torch.long, f"Bad label dtype: {label.dtype}"
            assert (label >= 0).all() and (label < 100).all(), f"Label out of range: {label}"
            loss = ddpm.p_losses(x, label, t)
            val_loss += loss.item()

        if epoch%10==0:
            model_save_file = f"model_saves/studenttddpm__conditional_epoch{epoch}.pth"
            torch.save(model.state_dict(),model_save_file)
        print(f"Epoch {epoch+1}: Loss = {train_loss:.4f}")
        print(f"Epoch {epoch+1}: Val Loss = {val_loss:.4f}")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu')
    # device = torch.device("cpu")
    print(f"Device: {device}")
    timesteps = 400
    batch_size = 64
    num_workers = 2

     # === Dataset ===
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = CIFAR100LongTail(root='./data', imbalance_factor=0.01, transform=transform)
    val_dataset = CIFAR100LongTail(root='./data', phase = "Validation", imbalance_factor=0.01, transform=transform)
    num_classes = dataset.num_classes


    train_loader = DataLoader(dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset,batch_size=batch_size)

    betas = linear_beta_schedule(timesteps)
    # model = UNet(num_classes=num_classes,base_channels=256).to(device)
    model = UNet2DConditionModel(
        sample_size=32,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 128, 256),
        down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
        class_embed_type="timestep",  # Important for conditional generation
        num_class_embeds=100,  # CIFAR-100
    )
    model.to(device)
    ddpm = DDPM(model, betas)

    train_ddpm(model, ddpm, train_loader,val_loader, epochs=200)
    torch.save(model.state_dict(),"model_saves/student_t_UNET_conditional.pth")

    # model.load_state_dict(torch.load("model_saves/student_t_UNET_conditional.pth"))

    class_label = 1
    visualize_denoising(ddpm, class_label)
    visualize_one_sample_per_class(ddpm)
    # sample_labels = torch.full((16,), 0, dtype=torch.long).to(device)
    sample_labels = torch.randint(0,10,(16,1)).to(device).squeeze()
    print(sample_labels.shape)
    samples = ddpm.sample(sample_labels,(16, 3, 32, 32)).cpu()
    grid = torch.clamp((samples + 1) / 2, 0, 1)  # Convert back to [0, 1]
    grid = torchvision.utils.make_grid(grid, nrow=4)
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    plt.show()



# ================================
#         Run Training
# ================================
if __name__=="__main__":
    main()