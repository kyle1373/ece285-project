import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

class ResNetEncoder(nn.Module):
    def __init__(self, latent_dim, condition_dim):
        super(ResNetEncoder, self).__init__()
        self.resnet = resnet18(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 512)
        self.condition_resnet = resnet18(pretrained=True)
        self.condition_resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.condition_resnet.fc = nn.Linear(self.condition_resnet.fc.in_features, 512)
        self.fc_mu = nn.Linear(1024, latent_dim)
        self.fc_logvar = nn.Linear(1024, latent_dim)

    def forward(self, x, condition):
        x = self.resnet(x)
        condition = self.condition_resnet(condition)
        h = torch.cat([x, condition], dim=1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

class ResNetDecoder(nn.Module):
    def __init__(self, latent_dim, condition_dim):
        super(ResNetDecoder, self).__init__()
        self.condition_resnet = resnet18(pretrained=True)
        self.condition_resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.condition_resnet.fc = nn.Linear(self.condition_resnet.fc.in_features, 512)
        self.fc = nn.Linear(latent_dim + 512, 512)
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z, condition):
        condition = self.condition_resnet(condition)
        h = torch.cat([z, condition], dim=1)
        h = F.relu(self.fc(h))
        h = h.view(h.size(0), 512, 1, 1)  # Reshape to be compatible with ConvTranspose2d
        x_recon = self.deconv_layers(h)
        return x_recon

class Res_CVAE(nn.Module):
    def __init__(self, latent_dim, condition_dim):
        super(Res_CVAE, self).__init__()
        self.encoder = ResNetEncoder(latent_dim, condition_dim)
        self.decoder = ResNetDecoder(latent_dim, condition_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, condition):
        mu, logvar = self.encoder(x, condition)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z, condition)
        return x_recon, mu, logvar
