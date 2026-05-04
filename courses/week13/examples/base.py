import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights

class SSD(nn.Module):
    def __init__(self, num_classes):
        super(SSD, self).__init__()
        self.num_classes = num_classes

        # Load pre-trained ResNet50 backbone
        resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        # Use ResNet50 up to the last convolutional block (exclude avgpool and fc)
        self.features = nn.Sequential(*list(resnet.children())[:-2])  # output: [B, 2048, H, W]

        self.extras = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2048, 1024, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(1024, 512, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
        ])

        # Localization layers
        self.loc = nn.ModuleList([
            nn.Conv2d(2048, 4 * 4, kernel_size=3, padding=1),   # backbone feature map
            nn.Conv2d(1024, 6 * 4, kernel_size=3, padding=1),
            nn.Conv2d(512, 6 * 4, kernel_size=3, padding=1),
            nn.Conv2d(256, 6 * 4, kernel_size=3, padding=1),
            nn.Conv2d(256, 4 * 4, kernel_size=3, padding=1),
            nn.Conv2d(256, 4 * 4, kernel_size=3, padding=1)
        ])

        # Confidence layers
        self.conf = nn.ModuleList([
            nn.Conv2d(2048, 4 * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(1024, 6 * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(512, 6 * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(256, 6 * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(256, 4 * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(256, 4 * num_classes, kernel_size=3, padding=1)
        ])

    def forward(self, x):
        locs = []
        confs = []

        x = self.features(x)
        print("Backbone output:", x.shape)

        locs.append(self.loc[0](x).permute(0, 2, 3, 1).contiguous())
        confs.append(self.conf[0](x).permute(0, 2, 3, 1).contiguous())

        for i, layer in enumerate(self.extras):
            x = layer(x)
            print(f"Extra layer {i} output:", x.shape)
            locs.append(self.loc[i + 1](x).permute(0, 2, 3, 1).contiguous())
            confs.append(self.conf[i + 1](x).permute(0, 2, 3, 1).contiguous())

        locs = torch.cat([o.view(o.size(0), -1) for o in locs], dim=1)
        confs = torch.cat([o.view(o.size(0), -1) for o in confs], dim=1)

        locs = locs.view(locs.size(0), -1, 4)
        confs = confs.view(confs.size(0), -1, self.num_classes)

        return locs, confs


# Example usage
if __name__ == "__main__":
    num_classes = 21  # 20 classes + background
    ssd = SSD(num_classes)
    x = torch.randn(1, 3, 300, 300)
    locs, confs = ssd(x)
    print("Localization predictions:", locs.size())
    print("Confidence predictions:", confs.size())