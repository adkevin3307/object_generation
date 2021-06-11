import torch
import torch.nn as nn


class ACGAN_Generator(nn.Module):
    def __init__(self, latent_dim: int, image_size: int, channels: int = 3) -> None:
        super(ACGAN_Generator, self).__init__()

        self.init_size = image_size // 4  # Initial size before upsampling

        self.linear = nn.Linear(latent_dim + 24, 128 * self.init_size ** 2)

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x, condition], dim=-1)

        x = self.linear(x)
        x = x.view(x.shape[0], 128, self.init_size, self.init_size)
        x = self.conv_blocks(x)

        return x


class ACGAN_Discriminator(nn.Module):
    def __init__(self, image_size: int, channels: int = 3) -> None:
        super(ACGAN_Discriminator, self).__init__()

        def discriminator_block(in_channels: int, out_channels: int, batch_normalize: bool = True):
            """Returns layers of each discriminator block"""

            block = [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25)
            ]

            if batch_normalize:
                block.append(nn.BatchNorm2d(out_channels, 0.8))

            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(channels, 16, batch_normalize=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        downsample_size = image_size // 2 ** 4

        # Output layers
        self.adv_layer = nn.Sequential(
            nn.Linear(128 * downsample_size ** 2, 1),
            nn.Sigmoid()
        )
        self.aux_layer = nn.Sequential(
            nn.Linear(128 * downsample_size ** 2, 24),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> tuple:
        x = self.conv_blocks(x)
        x = x.view(x.shape[0], -1)

        validity = self.adv_layer(x)
        label = self.aux_layer(x)

        return (validity, label)


class WGAN_Generator(nn.Module):
    def __init__(self, latent_dim: int, image_size: int, channels: int = 3) -> None:
        super(WGAN_Generator, self).__init__()

        self.linear = nn.Sequential(
            nn.Linear(latent_dim + 24, 128)
        )

        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose2d(128, image_size * 8, 4, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(image_size * 8, image_size * 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(image_size * 4, momentum=0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(image_size * 4, image_size * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(image_size * 2, momentum=0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(image_size * 2, image_size, 4, stride=2, padding=1),
            nn.BatchNorm2d(image_size, momentum=0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(image_size, channels, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x, condition], dim=-1)

        x = self.linear(x)
        x = x.view(*x.shape, 1, 1)
        x = self.conv_blocks(x)

        return x


class WGAN_Discriminator(nn.Module):
    def __init__(self, image_size: int, channels: int = 3) -> None:
        super(WGAN_Discriminator, self).__init__()

        self.conv_blocks = nn.Sequential(
            nn.Conv2d(channels + 24, image_size, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(image_size, image_size * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(image_size * 2, momentum=0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(image_size * 2, image_size * 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(image_size * 4, momentum=0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(image_size * 4, image_size * 8, 4, stride=2, padding=1),
            nn.BatchNorm2d(image_size * 8, momentum=0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(image_size * 8, 1, 4)
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        condition = condition.view(*condition.shape, 1, 1).repeat(1, 1, 64, 64)
        x = torch.cat([x, condition], dim=1)

        x = self.conv_blocks(x)

        x = x.squeeze()

        return x
