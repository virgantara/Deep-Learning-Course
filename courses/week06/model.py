import torch
import torch.nn as nn

class EncoderVAE(nn.Module):
    def __init__(self):
        super(EncoderVAE, self).__init__()

        # inputan 1 x 32 x 32 > 32 x 16 x 16
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1) 
        
        # inputan 32 x 16 x 16 > 64 x 8 x 8
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)

        # inputan 64 x 8 x 8 > 128 x 4 x 4
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)

        self.relu = nn.ReLU()

        self.flatten = nn.Flatten()

        self.fc_mu = nn.Linear(128 * 4 * 4, 2)
        self.fc_logvar = nn.Linear(128 * 4 * 4, 2)

    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))

        x = self.flatten(x)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = self._reparameterize(mu, logvar)

        return z, mu, logvar


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # 16x16
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 8x8
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),# 4x4
            nn.ReLU(),
            nn.Flatten(),                                          # 2048
            nn.Linear(2048, 2)                                     # bottleneck
        )

    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(2, 2048),
            nn.Unflatten(1, (128, 4, 4)),
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(x)

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = EncoderVAE()
        self.decoder = Decoder()

    def forward(self, x):
        z, mu, logvar = self.encoder(x)
        out = self.decoder(z)
        return out, mu, logvar

if __name__ == '__main__':
    x = torch.randn(16, 1, 32, 32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE().to(device)
    print(model.encoder)
    x = x.to(device)
    y = model(x)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params / 1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")
