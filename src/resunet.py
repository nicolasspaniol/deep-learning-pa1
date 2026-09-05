from torch import nn

# 1. Treinem uma das arquiteturas da aula para segmentação binária (fundo vs. objeto).
# Reportem IoU e Dice.


class ResUNetEncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int=2):
        super().__init__()

        self.seq = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        return self.seq(x) + self.shortcut(x)


class ResUNetDecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, skip_channels: int):
        super().__init__()

        self.seq = nn.Sequential(
            nn.BatchNorm2d(in_channels + skip_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

        # up sampling
        self.ups = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        )

        self.shortcut = nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=1)

    # 'skip' aqui é o tensor de skip connection vindo
    # das saídas intermediárias do encoder
    def forward(self, x, skip):
        x = self.ups(x)
        x = torch.cat([x, skip], dim=1) # concat na dimensão 'C'
        return self.seq(x) + self.shortcut(x)


class ResUNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.enc0 = ResUNetEncoderBlock(in_channels, 64, stride=1)
        self.enc1 = ResUNetEncoderBlock(64, 128)
        self.enc2 = ResUNetEncoderBlock(128, 256)

        self.bridge = ResUNetEncoderBlock(256, 512)

        self.dec0 = ResUNetDecoderBlock(512, 256, 256)
        self.dec1 = ResUNetDecoderBlock(256, 128, 128)
        self.dec2 = ResUNetDecoderBlock(128, 64, 64)

        self.ending = nn.Sequential(
            nn.LazyConv2d(out_channels, kernel_size=1),
            # nn.Sigmoid() -- vamos usar a saída em logits
        )

    def forward(self, x):
        # encoder
        out0 = self.enc0(x)
        out1 = self.enc1(out0)
        out2 = self.enc2(out1)
        # ponte
        y = self.bridge(out2)
        # decoder
        y = self.dec0(y, out2)
        y = self.dec1(y, out1)
        y = self.dec2(y, out0)

        return self.ending(y)


if __name__ == '__main__':
    from synthetic_dataset import SyntheticEllipseDataset
    import torch
    import matplotlib.pyplot as plt
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    model = ResUNet(3, 1)
    model.eval()

    dataset = SyntheticEllipseDataset(num_samples=1)
    for img, mask in DataLoader(dataset):
        y = F.sigmoid(model(img))

        # mesmo número de datapoints
        assert y.shape[0] == img.shape[0]
        # shape[1] é o número de canais, que pode variar
        # mesma resolução
        assert y.shape[2] == img.shape[2]
        assert y.shape[3] == img.shape[3]

        fig, ax = plt.subplots()
        ax.imshow(y[0,0].detach().cpu().numpy(), cmap='gray')
        ax.set_title('Saída do modelo')
        ax.axis('off')
        plt.show()

        break
