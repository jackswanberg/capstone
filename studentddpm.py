import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.distributions.studentT import StudentT
import numpy as np

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
train_loader = DataLoader(train_data, batch_size=64, shuffle=True,num_workers=4)

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
#      Helper Function
# ================================
def extract(a, t, x_shape):
    return a.gather(-1, t).reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))

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
        return torch.sqrt(alpha_bar) * x_start + torch.sqrt(1 - alpha_bar) * noise

    def p_losses(self, x_start, t):
        x_noisy = self.q_sample(x_start, t)
        noise = StudentT(df=self.nu, loc=0, scale=1).sample(x_start.shape).to(self.device)
        predicted = self.model(x_noisy, t)
        return student_t_nll(noise,predicted,self.nu)
        # return F.mse_loss(predicted, noise)

    def p_sample(self, x, t):
        betas_t = extract(self.betas, t, x.shape)
        alphas_t = extract(self.alphas,t,x.shape)
        alpha_bar_t = extract(self.alpha_bars, t, x.shape)
        predicted_noise = self.model(x, t)

        mean = (1 / torch.sqrt(alphas_t)) * (
                x - ((1 - alphas_t) / torch.sqrt(1 - alpha_bar_t)) * predicted_noise)

        if t[0] == 0:
            return mean
        noise = StudentT(df=self.nu, loc=0, scale=1).sample(x.shape).to(self.device)

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
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.train()
    for epoch in tqdm(range(epochs)):
        for batch in dataloader:
            x = batch[0].to(ddpm.device)
            t = torch.randint(0, ddpm.timesteps, (x.size(0),), device=ddpm.device)
            loss = ddpm.p_losses(x, t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

# ================================
#         Run Training
# ================================
if __name__=="__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu')
    timesteps = 100
    betas = linear_beta_schedule(timesteps)
    model = ConvNet().to(device)
    ddpm = StudentTDDPM(model, betas, nu=4.0)

    train_ddpm(model, ddpm, train_loader, epochs=30)

    torch.save(model.state_dict(),"model_saves/student_t.pth")

    samples = ddpm.sample((16, 3, 32, 32)).cpu()
    grid = torch.clamp((samples + 1) / 2, 0, 1)  # Convert back to [0, 1]
    grid = torchvision.utils.make_grid(grid, nrow=4)
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    plt.show()