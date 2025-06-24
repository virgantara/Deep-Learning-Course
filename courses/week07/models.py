import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False), # 1, 64, 64 -> 64, 32, 32
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.1),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False), # 64, 32, 32 -> 128, 16, 16
            nn.BatchNorm2d(128, momentum=0.1),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.1),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False), # 128, 16, 16 -> 256, 8, 8
            nn.BatchNorm2d(256, momentum=0.1),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.1),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False), # 256, 8, 8 -> 512, 4, 4
            nn.BatchNorm2d(512, momentum=0.1),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.1),

            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False), # 512, 4, 4 -> 1, 1, 1
            nn.Sigmoid(),

            nn.Flatten()
            )
    
    def forward(self, x):
        x = self.backbone(x)

        return x


class Generator(nn.Module):
    def __init__(self, embedding_dim):

        super(Generator, self).__init__()

        self.embedding_dim = embedding_dim

        self.backbone = nn.Sequential(
            nn.ConvTranspose2d(in_channels=embedding_dim, out_channels=512, kernel_size=4, stride=1, padding=0, output_padding=0, bias=False), # 100, 1, 1 -> 512, 4, 4
            nn.BatchNorm2d(512, momentum=0.1),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1, output_padding=0, bias=False), # 512, 4, 4 -> 256, 8, 8
            nn.BatchNorm2d(256, momentum=0.1),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1, output_padding=0, bias=False), # 256, 8, 8 -> 128, 16, 16
            nn.BatchNorm2d(128, momentum=0.1),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1, output_padding=0, bias=False), # 128, 16, 16 -> 64, 32, 32
            nn.BatchNorm2d(64, momentum=0.1),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=4, stride=2, padding=1, output_padding=0, bias=False), # 64, 32, 32 -> 1, 64, 64

            nn.Tanh()
        )
    
    def forward(self, x):
        x = x.view(-1, self.embedding_dim, 1, 1)
        x = self.backbone(x)

        return x