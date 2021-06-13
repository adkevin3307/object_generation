import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D


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
            nn.Linear(latent_dim + 24, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128)
        )

        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose2d(128, image_size * 8, 4, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(image_size * 8, image_size * 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(image_size * 4, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(image_size * 4, image_size * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(image_size * 2, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(image_size * 2, image_size, 4, stride=2, padding=1),
            nn.BatchNorm2d(image_size, 0.8),
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

        self.transform = nn.Sequential(
            nn.Linear(image_size * image_size * channels + 24, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, image_size * image_size * channels)
        )

        self.conv_blocks = nn.Sequential(
            nn.Conv2d(channels, image_size, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(image_size, image_size * 2, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(image_size * 2, image_size * 4, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(image_size * 4, image_size * 8, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(image_size * 8, 1, 4)
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        origin_shape = x.shape

        x = torch.cat([x.view(x.shape[0], -1), condition], dim=-1)

        x = self.transform(x).view(origin_shape)
        x = self.conv_blocks(x)

        return x


class ActNorm(nn.Module):
    def __init__(self, param_dim: tuple = (1, 3, 1, 1)) -> None:
        super(ActNorm, self).__init__()

        self.scale = nn.Parameter(torch.ones(param_dim))
        self.bias = nn.Parameter(torch.zeros(param_dim))

        self.register_buffer('initialized', torch.tensor(0).to(torch.uint8))

    def forward(self, x: torch.Tensor) -> tuple:
        if not self.initialized:
            self.bias.squeeze().data.copy_(x.transpose(0, 1).flatten(1).mean(1)).view_as(self.bias)
            self.scale.squeeze().data.copy_(x.transpose(0, 1).flatten(1).std(1, False) + 1e-6).view_as(self.scale)

            self.initialized += 1

        z = (x - self.bias) / self.scale

        logdet = -1.0 * torch.sum(self.scale.abs().log()) * x.shape[2] * x.shape[3]

        return (z, logdet)

    def inverse(self, z: torch.Tensor) -> tuple:
        x = z * self.scale + self.bias

        logdet = torch.sum(self.scale.abs().log()) * z.shape[2] * z.shape[3]

        return (x, logdet)


class Invertible1x1Conv(nn.Module):
    def __init__(self, n_channels: int = 3, lu_factorize: bool = False) -> None:
        super(Invertible1x1Conv, self).__init__()

        self.lu_factorize = lu_factorize

        q, _ = torch.qr(torch.randn(n_channels, n_channels))

        if lu_factorize:
            p, l, u = torch.lu_unpack(*q.lu())

            self.p = nn.Parameter(p.squeeze())
            self.l = nn.Parameter(l.squeeze())
            self.u = nn.Parameter(u.squeeze())

            s = self.u.diag()

            self.log_s = nn.Parameter(s.abs().log())

            self.register_buffer('sign_s', s.sign())
            self.register_buffer('l_mask', torch.tril(torch.ones_like(self.l), -1))
        else:
            self.w = nn.Parameter(q)

    def forward(self, x: torch.Tensor) -> tuple:
        _, C, H, W = x.shape

        if self.lu_factorize:
            l = self.l * self.l_mask + torch.eye(C).to(self.l.device)
            u = self.u * self.l_mask.T + torch.diag(self.sign_s * self.log_s.exp())

            self.w = self.p @ l @ u

            logdet = torch.sum(self.log_s) * H * W
        else:
            logdet = torch.slogdet(self.w)[-1] * H * W

        z = F.conv2d(x, self.w.view(C, C, 1, 1))

        return (z, logdet)

    def inverse(self, z: torch.Tensor) -> tuple:
        _, C, H, W = z.shape

        if self.lu_factorize:
            l = torch.inverse(self.l * self.l_mask + torch.eye(C).to(self.l.device))
            u = torch.inverse(self.u * self.l_mask.T + torch.diag(self.sign_s * self.log_s.exp()))

            w_inv = u @ l @ torch.inverse(self.p)

            logdet = -1.0 * torch.sum(self.log_s) * H * W
        else:
            w_inv = torch.inverse(self.w)

            logdet = -1.0 * torch.slogdet(self.w)[-1] * H * W

        x = F.conv2d(z, w_inv.view(C, C, 1, 1))

        return (x, logdet)


class AffineCoupling(nn.Module):
    def __init__(self, n_channels: int, hidden_channels: int) -> None:
        super(AffineCoupling, self).__init__()

        self.conv_1 = nn.Sequential(
            nn.Conv2d(n_channels // 2, hidden_channels, kernel_size=3, padding=1, bias=False),
            ActNorm(param_dim=(1, hidden_channels, 1, 1))
        )
        self.relu_1 = nn.ReLU()

        self.conv_2 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, padding=1, bias=False),
            ActNorm(param_dim=(1, hidden_channels, 1, 1))
        )
        self.relu_2 = nn.ReLU()

        self.conv_3 = nn.Conv2d(hidden_channels, n_channels, kernel_size=3)

        self.log_scale_factor = nn.Parameter(torch.zeros(n_channels, 1, 1))

        self.conv_3.weight.data.zero_()
        self.conv_3.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> tuple:
        x1, x2 = x.chunk(2, dim=1)

        output = self.conv_1(x2)[0]
        output = self.relu_1(output)
        output = self.conv_2(output)[0]
        output = self.relu_2(output)
        output = self.conv_3(output) * self.log_scale_factor.exp()

        t = output[:, 0::2]
        s = output[:, 1::2]

        s = torch.sigmoid(s + 2.0)

        z1 = s * x1 + t
        z2 = x2

        z = torch.cat([z1, z2], dim=1)

        logdet = torch.sum(s.log(), dim=[1, 2, 3])

        return (z, logdet)

    def inverse(self, z: torch.Tensor) -> tuple:
        z1, z2 = z.chunk(2, dim=1)

        output = self.conv_1(z2)[0]
        output = nn.ReLU(output)
        output = self.conv_2(output)[0]
        output = nn.ReLU(output)
        output = self.conv_3(output) * self.log_scale_factor.exp()

        t = output[:, 0::2]
        s = output[:, 1::2]

        s = torch.sigmoid(s + 2.0)

        x1 = (z1 - t) / s
        x2 = z2

        x = torch.cat([x1, x2], dim=1)

        logdet = -1.0 * torch.sum(s.log(), dim=[1, 2, 3])

        return (x, logdet)


class Squeeze(nn.Module):
    def __init__(self) -> None:
        super(Squeeze, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        x = x.reshape(B, C, H // 2, 2, W // 2, 2)
        x = x.permute(0, 1, 3, 5, 2, 4)
        x = x.reshape(B, C * 4, H // 2, W // 2)

        return x

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        B, C, H, W = z.shape

        z = z.reshape(B, C // 4, 2, 2, H, W)
        z = z.permute(0, 1, 4, 2, 5, 3)
        z = z.reshape(B, C // 4, H * 2, W * 2)

        return z


class Gaussianize(nn.Module):
    def __init__(self, n_channels: int) -> None:
        super(Gaussianize, self).__init__()

        self.net = nn.Conv2d(n_channels, n_channels * 2, kernel_size=3, padding=1)
        self.log_scale_factor = nn.Parameter(torch.zeros(n_channels * 2, 1, 1))

        self.net.weight.data.zero_()
        self.net.bias.data.zero_()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> tuple:
        x1 = self.net(x1) * self.log_scale_factor.exp()

        m = x1[:, 0::2]
        logs = x1[:, 1::2]

        z = (x2 - m) * torch.exp(-1.0 * logs)
        logdet = -1.0 * torch.sum(logs, dim=[1, 2, 3])

        return (z, logdet)

    def inverse(self, z1: torch.Tensor, z2: torch.Tensor) -> tuple:
        z1 = self.net(z1) * self.log_scale_factor.exp()

        m = z1[:, 0::2]
        logs = z1[:, 1::2]

        x = m + z2 * torch.exp(logs)
        logdet = torch.sum(logs, dim=[1, 2, 3])

        return (x, logdet)


class Split(nn.Module):
    def __init__(self, n_channels: int) -> None:
        super(Split, self).__init__()

        self.gaussianize = Gaussianize(n_channels // 2)

    def forward(self, x: torch.Tensor) -> tuple:
        x1, x2 = x.chunk(2, dim=1)

        z1 = x1
        z2, logdet = self.gaussianize(x1, x2)

        return (z1, z2, logdet)

    def inverse(self, z1: torch.Tensor, z2: torch.Tensor) -> tuple:
        x1 = z1
        x2, logdet = self.gaussianize.inverse(z1, z2)

        x = torch.cat([x1, x2], dim=1)

        return (x, logdet)


class FlowSequential(nn.Sequential):
    def __init__(self, *args, **kwargs) -> None:
        super(FlowSequential, self).__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> tuple:
        total_logdet = 0.0

        for module in self:
            x, logdet = module(x)
            total_logdet = total_logdet + logdet

        return (x, total_logdet)

    def inverse(self, z: torch.Tensor) -> tuple:
        total_logdet = 0.0

        for module in reversed(self):
            z, logdet = module.inverse(z)
            total_logdet = total_logdet + logdet

        return (z, total_logdet)


class FlowStep(FlowSequential):
    def __init__(self, n_channels: int, hidden_channels: int, lu_factorize: bool = False) -> None:
        super(FlowStep, self).__init__(
            ActNorm(param_dim=(1, n_channels, 1, 1)),
            Invertible1x1Conv(n_channels, lu_factorize),
            AffineCoupling(n_channels, hidden_channels)
        )


class FlowLevel(nn.Module):
    def __init__(self, n_channels: int, hidden_channels: int, depth: int, lu_factorize: bool = False) -> None:
        super(FlowLevel, self).__init__()

        self.squeeze = Squeeze()
        self.flows = FlowSequential(*[FlowStep(n_channels * 4, hidden_channels, lu_factorize) for _ in range(depth)])
        self.split = Split(n_channels * 4)

    def forward(self, x: torch.Tensor) -> tuple:
        x = self.squeeze(x)
        x, logdet_flows = self.flows(x)
        z1, z2, logdet_split = self.split(x)

        logdet = logdet_flows + logdet_split

        return (z1, z2, logdet)

    def inverse(self, z1: torch.Tensor, z2: torch.Tensor) -> tuple:
        x, logdet_split = self.split.inverse(z1, z2)
        x, logdet_flows = self.flows.inverse(x)
        x = self.squeeze.inverse(x)

        logdet = logdet_split + logdet_flows

        return (x, logdet)


class Glow(nn.Module):
    def __init__(self, hidden_channels: int, depth: int, n_levels: int, input_dims: tuple = (3, 64, 64), lu_factorize: bool = False) -> None:
        super(Glow, self).__init__()

        in_channels, H, W = input_dims
        out_channels = int(in_channels * (4 ** (n_levels + 1)) / (2 ** n_levels))
        out_HW = int(H / (2 ** (n_levels + 1)))

        self.output_dims = (out_channels, out_HW, out_HW)

        self.flowlevels = nn.ModuleList([FlowLevel(in_channels * (2 ** i), hidden_channels, depth, lu_factorize) for i in range(n_levels)])
        self.squeeze = Squeeze()
        self.flowsteps = FlowSequential(*[FlowStep(out_channels, hidden_channels, lu_factorize) for _ in range(depth)])
        self.gaussianize = Gaussianize(out_channels)

        self.register_buffer('base_dist_mean', torch.zeros(1))
        self.register_buffer('base_dist_var', torch.ones(1))

    @property
    def base_distribution(self):
        return D.Normal(self.base_dist_mean, self.base_dist_var)

    def forward(self, x: torch.Tensor) -> tuple:
        total_z = []
        total_logdet = 0.0

        for module in self.flowlevels:
            x, z, logdet = module(x)

            total_z.append(z)
            total_logdet += logdet

        x = self.squeeze(x)
        z, logdet = self.flowsteps(x)

        total_logdet += logdet

        z, logdet = self.gaussianize(torch.zeros_like(z), z)

        total_z.append(z)
        total_logdet += logdet

        return (total_z, total_logdet)

    def inverse(self, total_z: list = None, batch_size: int = None, z_std: float = 1.0) -> tuple:
        if total_z == None:
            assert batch_size is not None

            total_z = [z_std * self.base_distribution.sample((batch_size, *self.output_dims)).squeeze()]

        z, total_logdet = self.gaussianize.inverse(torch.zeros_like(total_z[-1]), total_z[-1])

        x, logdet = self.flowsteps.inverse(z)
        total_logdet += logdet

        x = self.squeeze.inverse(x)

        for i, module in reversed(self.flowlevels):
            z = z_std * (self.base_distribution.sample(x.shape).squeeze() if len(total_z) == 1 else total_z[-i - 2])

            x, logdet = module.inverse(x, z)
            total_logdet += logdet

        return (x, total_logdet)

    def log_prob(self, x, bits_per_pixel=False):
        zs, logdet = self.forward(x)
        prob = sum(self.base_distribution.log_prob(z).sum([1, 2, 3]) for z in zs) + logdet

        if bits_per_pixel:
            prob /= (math.log(2) * x[0].numel())

        return prob
