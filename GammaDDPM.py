import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.distributions.studentT import StudentT
from torch.distributions.gamma import Gamma
import numpy as np

from models import UNet2

# ================================
#         CIFAR10 Dataset
# ================================
class ToMinusOneToOne:
    def __call__(self, x):
        return x * 2. - 1.
    
transform = transforms.Compose([
    transforms.ToTensor(),
    ToMinusOneToOne()  # Normalize to [-1, 1]
])
train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
num_classes = len(train_data.classes)
train_loader = DataLoader(train_data, batch_size=128, shuffle=True,num_workers=4)

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
    def __init__(self, in_channels=3, base_channels=64, time_emb_dim=256):
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

# Call the function

# ================================
#      Helper Function
# ================================
def extract(a, t, x_shape):
    return a.gather(-1, t.to(torch.int64)).reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))

def sample_gamma_noise(shape, alpha=2.0, beta=1.0, device='cpu'):
    gamma = Gamma(torch.full(shape, alpha).to(device), torch.full(shape, beta).to(device))
    noise = gamma.sample()
    noise = noise - noise.mean(dim=(1,2,3), keepdim=True)
    return noise

def sample_frechet_noise(shape, alpha=2.0, scale=1.0, loc=0.0, device='cpu'):
    # Uniform samples
    u = torch.rand(shape, device=device)
    noise = loc + scale * (-torch.log(u)) ** (-1.0 / alpha)
    noise = noise - noise.mean(dim=(1,2,3), keepdim=True)
    return noise

def gamma_nll(x, alpha=2.0, beta=1.0):
    return -Gamma(alpha, beta).log_prob(x).mean()

def frechet_nll(x, alpha=2.0, scale=1.0, loc=0.0):
    eps = 1e-6
    z = (x - loc) / scale
    z = torch.clamp(z, min=eps)
    log_prob = torch.log(alpha / scale) - (1 + alpha) * torch.log(z) - z.pow(-alpha)
    return -log_prob.mean()

class GammaDDPM:
    def __init__(self, model, betas, gamma_alpha=2.0, gamma_beta=1.0):
        self.model = model
        self.gamma_alpha = gamma_alpha
        self.gamma_beta = gamma_beta
        self.device = next(model.parameters()).device

        self.timesteps = len(betas)
        self.betas = betas.to(self.device)
        self.alphas = 1. - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x_start, t):
        alpha_bar = extract(self.alpha_bars, t, x_start.shape)
        noise = sample_gamma_noise(x_start.shape, device=self.device)
        x_noisy = torch.sqrt(alpha_bar) * x_start + torch.sqrt(1 - alpha_bar) * noise
        return x_noisy, noise  # Return both!

    def p_losses(self, x_start, label, t):
        x_noisy, noise = self.q_sample(x_start, t)
        predicted = self.model(x_noisy, label, t)
        # noise = torch.clamp(noise, min=1e-5)
        # predicted = torch.clamp(predicted, min=1e-5)
        loss = nn.functional.smooth_l1_loss(predicted,noise)
        return loss
        # return gamma_nll(torch.clamp(noise - predicted,min=1e-5), self.gamma_alpha,self.gamma_beta)


    def p_sample(self, x, label, t):
        betas_t = extract(self.betas, t, x.shape)
        alphas_t = extract(self.alphas,t,x.shape)
        alpha_bar_t = extract(self.alpha_bars, t, x.shape)
        predicted_noise = self.model(x, label, t)

        mean = (1 / torch.sqrt(alphas_t)) * (
                x - ((1 - alphas_t) / torch.sqrt(1 - alpha_bar_t)) * predicted_noise)
 
        if t[0] == 0:
            return mean
        noise = sample_gamma_noise(x.shape, alpha=2.0, beta=1.0, device=self.device)

        return torch.sqrt(alphas_t) * mean + torch.sqrt(betas_t) * noise

    def sample(self, shape):
        x = torch.randn(shape).to(self.device)
        for t in reversed(range(self.timesteps)):
            t_batch = torch.tensor([t] * shape[0]).to(self.device)
            x = self.p_sample(x, t_batch)
        return x.clamp(-1, 1)
    
class FrechetDDPM:
    def __init__(self, model, betas, f_alpha=2.0, scale=1.0):
        self.model = model
        self.f_alpha = f_alpha
        self.scale = scale
        self.device = next(model.parameters()).device

        self.timesteps = len(betas)
        self.betas = betas.to(self.device)
        self.alphas = 1. - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x_start, t):
        alpha_bar = extract(self.alpha_bars, t, x_start.shape)
        noise = sample_frechet_noise(x_start.shape, alpha=self.f_alpha, scale=self.scale, device=self.device)
        x_noisy = torch.sqrt(alpha_bar) * x_start + torch.sqrt(1 - alpha_bar) * noise
        return x_noisy, noise  # Return both!

    def p_losses(self, x_start, label, t):
        x_noisy, noise = self.q_sample(x_start, t)
        predicted = self.model(x_noisy, label, t)
        # noise = torch.clamp(noise, min=1e-5)
        # predicted = torch.clamp(predicted, min=1e-5)
        loss = nn.functional.smooth_l1_loss(predicted,noise)
        return loss
        # return gamma_nll(torch.clamp(noise - predicted,min=1e-5), self.gamma_alpha,self.gamma_beta)


    def p_sample(self, x, label, t):
        betas_t = extract(self.betas, t, x.shape)
        alphas_t = extract(self.alphas,t,x.shape)
        alpha_bar_t = extract(self.alpha_bars, t, x.shape)
        predicted_noise = self.model(x, label, t)

        mean = (1 / torch.sqrt(alphas_t)) * (
                x - ((1 - alphas_t) / torch.sqrt(1 - alpha_bar_t)) * predicted_noise)
 
        if t[0] == 0:
            return mean
        noise = sample_frechet_noise(x.shape, alpha=self.f_alpha, scale=self.scale, device=self.device)

        return torch.sqrt(alphas_t) * mean + torch.sqrt(betas_t) * noise

    def sample(self, shape):
        x = torch.randn(shape).to(self.device)
        for t in reversed(range(self.timesteps)):
            t_batch = torch.tensor([t] * shape[0]).to(self.device)
            x = self.p_sample(x, t_batch)
        return x.clamp(-1, 1)

# ================================
#        Training Loop
# ================================
def train_ddpm(model, ddpm, dataloader, epochs=50):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()
    for epoch in tqdm(range(epochs)):
        for batch in dataloader:
            x = batch[0].to(ddpm.device)
            label = batch[1].to(ddpm.device)
            t = torch.randint(0, ddpm.timesteps, (x.size(0),), device=ddpm.device)
            loss = ddpm.p_losses(x, label, t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch%10==0:
            model_save_file = f"model_saves/gammaddpm__conditional_epoch{epoch}.pth"
            torch.save(model.state_dict(),model_save_file)
        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    timesteps = 400
    betas = linear_beta_schedule(timesteps)
    model = UNet2(base_channels=128).to(device)
    ddpm = FrechetDDPM(model, betas, f_alpha=2.0, scale=1.0)



    train_ddpm(model, ddpm, train_loader, epochs=500)
# ================================
#         Run Training
# ================================
if __name__=="__main__":
    main()