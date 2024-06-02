import torch
import torch.nn as nn
import torch.nn.functional as F

class CVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim, condition_dim):
        super(CVAE, self).__init__()
        self.z_dim = z_dim
        self.condition_dim = condition_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim + condition_dim, hidden_dim),
            nn.ReLU(),
        )
        
        self.fc_mu = nn.Linear(hidden_dim, z_dim)
        self.fc_logvar = nn.Linear(hidden_dim, z_dim)
        
        # Condition encoder
        self.condition_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(),
        )
        
        # Fully connected layer for condition encoding
        self.fc_condition = None
                                      
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(z_dim + condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
    def initialize_fc_condition(self, condition):
        with torch.no_grad():
            sample_output = self.condition_encoder(condition)
            output_size = (min(sample_output.size(2), 4), min(sample_output.size(3), 4))
            fc_input_dim = 128 * output_size[0] * output_size[1]
            self.fc_condition = nn.Linear(fc_input_dim, self.condition_dim).to(condition.device)
            nn.init.xavier_normal_(self.fc_condition.weight)
            nn.init.zeros_(self.fc_condition.bias)
        
    def forward_condition_encoder(self, condition):
        if self.fc_condition is None:
            self.initialize_fc_condition(condition)
        
        _, _, h, w = condition.shape
        if h < 8 or w < 8:
            pad_h = max(0, 8 - h)
            pad_w = max(0, 8 - w)
            condition = F.pad(condition, (0, pad_w, 0, pad_h))            
            
        # Forward pass through the condition encoder
        x = self.condition_encoder(condition)
        batch_size = x.size(0)
        
        # Dynamically determine the size of the adaptive pooling
        output_size = (min(x.size(2), 4), min(x.size(3), 4))
        
        x = nn.AdaptiveAvgPool2d(output_size)(x)  # Adaptive pooling to a fixed size or smaller
        x = x.view(batch_size, -1)  # Flatten
        x = self.fc_condition(x)  # Fully connected layer to transform to condition_dim

        return x
        
    def encode(self, x, cond_encoded):
        x_flat = x.view(x.size(0), -1)
        # Concatenate the condition with the input image
        x_cond = torch.cat([x_flat, cond_encoded], dim=1)
        h1 = self.encoder(x_cond)
        return self.fc_mu(h1), self.fc_logvar(h1)   

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, cond_encoded):
        # Concatenate z with the encoded condition
        z_cond = torch.cat([z, cond_encoded], dim=1)
        return self.decoder(z_cond)

    def forward(self, x, condition):
        cond_encoded = self.forward_condition_encoder(condition)
        mu, logvar = self.encode(x, cond_encoded)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, cond_encoded), mu, logvar
