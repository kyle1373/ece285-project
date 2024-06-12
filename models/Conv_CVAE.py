import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv_CVAE(nn.Module):
    def __init__(self, z_dim, condition_dim):
        super(Conv_CVAE, self).__init__()
        self.z_dim = z_dim
        self.condition_dim = condition_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 32, 4, stride=2, padding=1),  # Change input channels to 2 for concatenated input
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc1 = nn.Linear(128 * 8 * 8, z_dim)
        self.fc2 = nn.Linear(128 * 8 * 8, z_dim)

        # Condition encoder
        self.condition_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, condition_dim) 
        )

        # Decoder
        self.fc3 = nn.Linear(z_dim + condition_dim, 128 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128, 8, 8)),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x, condition):
        # Concatenate the condition with the input image
        x_cond = torch.cat([x, condition], dim=1)
        h1 = self.encoder(x_cond)
        return self.fc1(h1), self.fc2(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, condition):
        # Encode the condition
        cond_encoded = self.condition_encoder(condition)
        # Concatenate z with the encoded condition
        z_cond = torch.cat([z, cond_encoded], dim=1)
        h3 = F.relu(self.fc3(z_cond))
        return self.decoder(h3)

    def forward(self, x, condition):
        mu, logvar = self.encode(x, condition)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, condition), mu, logvar