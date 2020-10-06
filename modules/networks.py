# The MIT License
#
# Copyright (c) 2020 Vincent Liu
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SPADE(nn.Module):
    ''' Implements BatchNorm2d with Spatially Adaptive Denormalization '''

    hid_channels = 128

    def __init__(self, channels: int, cond_channels: int):
        super().__init__()

        self.batchnorm = nn.BatchNorm2d(channels)
        self.spade = nn.Sequential(
            nn.Conv2d(cond_channels, self.hid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hid_channels, 2 * channels, kernel_size=3, padding=1),
        )

    def forward(self, x, seg):
        # apply normalization
        x = self.batchnorm(x)

        # compute denormalization
        seg = F.interpolate(seg, size=x.shape[-2:], mode='nearest')
        gamma, beta = torch.chunk(self.spade(seg), 2, dim=1)

        # apply denormalization
        x = x * (1 + gamma) + beta
        return x


class ResidualBlock(nn.Module):
    ''' Implements a residual block with BatchNorm2d + SPADE '''

    def __init__(self, in_channels: int, out_channels: int, cond_channels: int):
        super().__init__()

        hid_channels = min(in_channels, out_channels)

        self.proj = in_channels != out_channels
        if self.proj:
            self.norm0 = SPADE(in_channels, cond_channels)
            self.conv0 = nn.utils.spectral_norm(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            )

        self.activation = nn.LeakyReLU(0.2)
        self.norm1 = SPADE(in_channels, cond_channels)
        self.norm2 = SPADE(hid_channels, cond_channels)
        self.conv1 = nn.utils.spectral_norm(
            nn.Conv2d(in_channels, hid_channels, kernel_size=3, padding=1)
        )
        self.conv2 = nn.utils.spectral_norm(
            nn.Conv2d(hid_channels, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x, seg):
        dx = self.norm1(x, seg)
        dx = self.activation(dx)
        dx = self.conv1(dx)
        dx = self.norm2(dx, seg)
        dx = self.activation(dx)
        dx = self.conv2(dx)

        # Learn skip connection if in_channels != out_channels
        if self.proj:
            x = self.norm0(x, seg)
            x = self.conv0(x)

        return x + dx


class Encoder(nn.Module):
    ''' Implements a GauGAN encoder '''

    max_channels = 512

    def __init__(
        self,
        spatial_size: Tuple[int, int],
        z_dim: int = 256,
        n_downsample: int = 6,
        base_channels: int = 64,
    ):
        super().__init__()

        layers = []
        channels = base_channels
        for i in range(n_downsample):
            in_channels = 3 if i == 0 else channels
            out_channels = 2 * z_dim if i < n_downsample else max(self.max_channels, channels * 2)
            layers += [
                nn.utils.spectral_norm(
                    nn.Conv2d(in_channels, out_channels, stride=2, kernel_size=3, padding=1)
                ),
                nn.InstanceNorm2d(out_channels),
                nn.LeakyReLU(0.2),
            ]
            channels = out_channels

        h, w = spatial_size[0] // 2 ** n_downsample, spatial_size[1] // 2 ** n_downsample
        layers += [
            nn.Flatten(1),
            nn.Linear(channels * h * w, 2 * z_dim),
        ]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return torch.chunk(self.layers(x), 2, dim=1)


class Generator(nn.Module):
    ''' Implements a GauGAN generator '''

    max_channels = 1024

    def __init__(
        self,
        n_classes: int,
        spatial_size: int,
        z_dim: int = 256,
        base_channels: int = 64,
        n_upsample: int = 6,
    ):
        super().__init__()

        h, w = spatial_size[0] // 2 ** n_upsample, spatial_size[1] // 2 ** n_upsample
        self.proj_z = nn.Linear(z_dim, self.max_channels * h * w)
        self.reshape = lambda x: torch.reshape(x, (-1, self.max_channels, h, w))

        self.upsample = nn.Upsample(scale_factor=2)
        self.res_blocks = nn.ModuleList()
        for i in reversed(range(n_upsample)):
            in_channels = self.max_channels if i == n_upsample - 1 else min(self.max_channels, base_channels * 2 ** (i+1))
            out_channels = min(self.max_channels, base_channels * 2 ** i)
            self.res_blocks.append(ResidualBlock(in_channels, out_channels, n_classes))

        self.proj_o = nn.Sequential(
            nn.Conv2d(base_channels, 3, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, z, seg):
        h = self.proj_z(z)
        h = self.reshape(h)
        for res_block in self.res_blocks:
            h = res_block(h, seg)
            h = self.upsample(h)
        h = self.proj_o(h)
        return h


class PatchGANDiscriminator(nn.Module):
    ''' Implements an N-layer PatchGAN discriminator '''

    def __init__(
        self,
        in_channels: int,
        base_channels: int = 64,
        n_layers: int = 3,
    ):
        super().__init__()

        self.layers = nn.ModuleList()

        # initial convolutional layer
        self.layers.append(
            nn.Sequential(
                nn.utils.spectral_norm(
                    nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=2)
                ),
                nn.LeakyReLU(0.2, inplace=True),
            )
        )

        # downsampling convolutional layers
        channels = base_channels
        for _ in range(1, n_layers):
            prev_channels = channels
            channels = min(2 * channels, 512)
            self.layers.append(
                nn.Sequential(
                    nn.utils.spectral_norm(
                        nn.Conv2d(prev_channels, channels, kernel_size=4, stride=2, padding=2)
                    ),
                    nn.InstanceNorm2d(channels, affine=False),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )

        # output convolutional layer
        prev_channels = channels
        channels = min(2 * channels, 512)
        self.layers.append(
            nn.Sequential(
                nn.utils.spectral_norm(
                    nn.Conv2d(prev_channels, channels, kernel_size=4, stride=1, padding=2))
                ,
                nn.InstanceNorm2d(channels, affine=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.utils.spectral_norm(
                    nn.Conv2d(channels, 1, kernel_size=4, stride=1, padding=2)
                ),
            )
        )

    def forward(self, x):
        outputs = [] # for feature matching loss
        for layer in self.layers:
            x = layer(x)
            outputs.append(x)

        return outputs


class Discriminator(nn.Module):
    ''' Implements a GauGAN discriminator '''

    def __init__(
        self,
        in_channels: int,
        base_channels: int = 64,
        n_layers: int = 3,
        n_discriminators: int = 3,
    ):
        super().__init__()

        # initialize all discriminators
        self.discriminators = nn.ModuleList()
        for _ in range(n_discriminators):
            self.discriminators.append(
                PatchGANDiscriminator(in_channels, base_channels=base_channels, n_layers=n_layers)
            )

        # downsampling layer to pass inputs between discriminators at different scales
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)

    def forward(self, x):
        outputs = []

        for i, discriminator in enumerate(self.discriminators):
            # downsample input for subsequent discriminators
            if i != 0:
                x = self.downsample(x)

            outputs.append(discriminator(x))

        # return list of multiscale discriminator outputs
        return outputs

    @property
    def n_discriminators(self):
        return len(self.discriminators)


class GauGAN(nn.Module):
    ''' Implements GauGAN '''

    def __init__(
        self,
        n_classes: int,
        spatial_size: Tuple[int, int],
        base_channels: int = 64,
        z_dim: int = 256,
        n_upsample: int = 6,
        n_disc_layers: int = 3,
        n_disc: int = 3,
    ):
        super().__init__()

        self.z_dim = z_dim
        self.encoder = Encoder(
            spatial_size, z_dim=z_dim, n_downsample=n_upsample, base_channels=base_channels,
        )
        self.generator = Generator(
            n_classes, spatial_size, z_dim=z_dim, base_channels=base_channels, n_upsample=n_upsample,
        )
        self.discriminator = Discriminator(
            n_classes + 3, base_channels=base_channels, n_layers=n_disc_layers, n_discriminators=n_disc,
        )

    def forward(self, x, seg):
        ''' Performs a full forward pass for training. '''
        mu, logvar = self.encode(x)
        z = self.sample_z(mu, logvar)
        x_fake = self.generate(z, seg)
        pred = self.discriminate(x_fake, seg)
        return x_fake, pred

    def encode(self, x):
        return self.encoder(x)

    def generate(self, z, seg):
        ''' Generates fake image from noise vector and segmentation. '''
        return self.generator(z, seg)

    def discriminate(self, x, seg):
        ''' Predicts whether input image is real. '''
        return self.discriminator(torch.cat((x, seg), dim=1))

    def sample_z(self, mu=None, logvar=None, n_samples=1):
        ''' Samples noise vector with reparameterization trick. '''
        if mu is None or logvar is None:
            mu = torch.zeros((n_samples, self.z_dim))
            logvar = torch.zeros((n_samples, self.z_dim))

        eps = torch.randn(mu.size(), device=mu.device).to(mu.dtype)
        return (logvar / 2).exp() * eps + mu

    @property
    def n_disc(self):
        return self.discriminator.n_discriminators
