import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, in_channels=3, num_features=128, z_dim=200):
        super(Encoder, self).__init__()

        # inputan 3 x 32 x 32 > 128 x 16 x 16
        self.conv1 = nn.Conv2d(in_channels, num_features, kernel_size=3, stride=2, padding=1) 
        self.bn1 = nn.BatchNorm2d(num_features)
        
        # inputan 128 x 16 x 16 > 128 x 8 x 8
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features)
        
        # inputan 128 x 8 x 8 > 128 x 4 x 4
        self.conv3 = nn.Conv2d(num_features, num_features, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features)
        
        self.conv4 = nn.Conv2d(num_features, num_features, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(num_features)

        self.leaky_relu = nn.LeakyReLU()

        self.flatten = nn.Flatten()

        self.fc_mu = nn.Linear(num_features * 2 * 2, z_dim)
        self.fc_logvar = nn.Linear(num_features * 2 * 2, z_dim)

    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x):
        x = self.leaky_relu(self.bn1(self.conv1(x)))
        x = self.leaky_relu(self.bn2(self.conv2(x)))
        x = self.leaky_relu(self.bn3(self.conv3(x)))
        x = self.leaky_relu(self.bn4(self.conv4(x)))

        shape_before_flattening = x.shape[1:]
        
        x = self.flatten(x)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = self._reparameterize(mu, logvar)

        return z, mu, logvar, shape_before_flattening



class Decoder(nn.Module):
    def __init__(self, z_dim=200, num_features=128, out_channels=3):
        super(Decoder, self).__init__()

        self.fc = nn.Linear(z_dim, num_features * 2 * 2)

        self.deconv1 = nn.ConvTranspose2d(num_features, num_features, kernel_size=4, stride=2, padding=1)  # 2→4
        self.bn1 = nn.BatchNorm2d(num_features)

        self.deconv2 = nn.ConvTranspose2d(num_features, num_features, kernel_size=4, stride=2, padding=1)  # 4→8
        self.bn2 = nn.BatchNorm2d(num_features)

        self.deconv3 = nn.ConvTranspose2d(num_features, num_features, kernel_size=4, stride=2, padding=1)  # 8→16
        self.bn3 = nn.BatchNorm2d(num_features)

        self.deconv4 = nn.ConvTranspose2d(num_features, out_channels, kernel_size=4, stride=2, padding=1)  # 16→32

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()  # Output image range [0, 1]

    def forward(self, z, shape_before_flattening):
        x = self.fc(z)
        x = x.view(-1, *shape_before_flattening)  # reshape into conv feature shape

        x = self.leaky_relu(self.bn1(self.deconv1(x)))
        x = self.leaky_relu(self.bn2(self.deconv2(x)))
        x = self.leaky_relu(self.bn3(self.deconv3(x)))
        x = self.sigmoid(self.deconv4(x))  # Final layer

        return x

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        z, mu, logvar, shape = self.encoder(x)
        out = self.decoder(z, shape)
        return out, mu, logvar

if __name__ == '__main__':
    x = torch.randn(16, 3, 32, 32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE().to(device)
    print(model.decoder)
    x = x.to(device)
    y, mu, logvar = model(x)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params / 1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")
