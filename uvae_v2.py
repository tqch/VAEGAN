import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import PartialBatchNorm2d, BatchDropout


def frechet_distance(mean_1, mean_2, logvar_1, logvar_2):
    # frechet distance for two diagnoal Gaussians
    dif_mean_sq = (mean_1 - mean_2).pow(2)
    dif_var = torch.exp(logvar_1) + torch.exp(logvar_2) - torch.exp(0.5 * (logvar_1 + logvar_2))
    return (dif_mean_sq + dif_var).sum()


def kullback_leibler(mean_1, mean_2, logvar_1, logvar_2):
    # kl divergence for two diagnoal Gaussians
    dif_mean_sq = (mean_1 - mean_2).pow(2)
    dif_logvar = logvar_1 - logvar_2
    kld = torch.exp(dif_logvar) + dif_mean_sq * torch.exp(-logvar_2)
    kld += - dif_logvar + 1
    kld *= 0.5
    return kld.sum()


class ConvBlock(nn.Module):
    def __init__(self, in_chans, out_chans, final=False):
        super(ConvBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_chans)
        self.conv1 = nn.Conv2d(in_chans, out_chans, 3, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_chans)
        self.conv2 = nn.Conv2d(out_chans, out_chans, 3, 1, 1, bias=final)

    def forward(self, x):
        x = F.relu(self.bn1(x))
        x = F.relu(self.bn2(self.conv1(x)))
        x = self.conv2(x)
        return x


class DeconvBlock(nn.Module):
    def __init__(self, in_chans, out_chans, odd=False, splits=None, final=False):
        super(DeconvBlock, self).__init__()
        if splits is None:
            self.bn1 = nn.BatchNorm2d(in_chans)
        else:
            self.bn1 = PartialBatchNorm2d(splits)
        self.odd = odd  # whether the input tensor has an odd spatial size
        self.deconv1 = nn.ConvTranspose2d(in_chans, in_chans, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_chans)
        self.deconv2 = nn.ConvTranspose2d(in_chans, out_chans, 2 + odd, 2, 0 + odd, bias=final)

    def forward(self, x):
        x = F.relu(self.bn1(x))
        x = F.relu(self.bn2(self.deconv1(x)))
        x = self.deconv2(x)
        return x


class SkipConnection(nn.Module):
    def __init__(self, in_chans, out_chans):
        super(SkipConnection, self).__init__()
        self.bn = nn.BatchNorm2d(in_chans)
        # depth-wise convolution
        self.conv = nn.Conv2d(in_chans, out_chans, 1, 1, 0, bias=True)

    def forward(self, x):
        x = F.relu(self.bn(x))
        x = self.conv(x)
        return x


class SelfLoop(nn.Module):
    def __init__(self, in_chans, out_chans):
        super(SelfLoop, self).__init__()
        self.in_bn = nn.BatchNorm2d(in_chans)
        self.in_conv = nn.Conv2d(in_chans, out_chans, 1, 1, 0, bias=False)
        self.mid_bn = nn.BatchNorm2d(out_chans)
        self.mid_conv = nn.Conv2d(out_chans, out_chans, 3, 1, 1, bias=False)
        self.out_bn = nn.BatchNorm2d(out_chans)
        self.out_conv = nn.Conv2d(out_chans, out_chans, 1, 1, 0)

    def forward(self, x):
        x = self.in_conv(F.relu(self.in_bn(x)))
        x = self.mid_conv(F.relu(self.mid_bn(x)))
        x = self.out_conv(F.relu(self.out_bn(x)))
        return x


class Encoder(nn.Module):
    def __init__(self, in_chans, latent_dims, skip_dims):
        super(Encoder, self).__init__()
        self.in_chans = in_chans
        self.latent_dims = latent_dims
        self.skip_dims = skip_dims
        self.in_conv = nn.Conv2d(in_chans, latent_dims[0], 3, 1, 1, bias=False)
        self.down_blocks = nn.ModuleList([ConvBlock(latent_dims[0], latent_dims[0])])
        self.skip_connections = nn.ModuleList([SkipConnection(latent_dims[0], 2 * skip_dims[0])])
        assert len(latent_dims) == (len(skip_dims) + 1)
        for i in range(len(skip_dims) - 1):
            self.down_blocks.append(ConvBlock(latent_dims[i], latent_dims[i + 1]))
            self.skip_connections.append(SkipConnection(latent_dims[i + 1], 2 * skip_dims[i + 1]))
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.final_skip = SkipConnection(latent_dims[-2], 2 * latent_dims[-1])

    def forward(self, x):
        x = self.in_conv(x)
        moms = []
        for block, sc in zip(self.down_blocks, self.skip_connections):
            x = block(x)
            skip = sc(x)
            moms.append(skip)
        x = self.final_skip(x)
        moms.append(self.avg_pool(x))
        return moms


class Decoder(nn.Module):
    def __init__(self, out_chans, latent_dims, skip_dims, out_size, upsampling_sizes):
        super(Decoder, self).__init__()
        self.out_chans = out_chans
        self.latent_dims = latent_dims
        self.skip_dims = skip_dims
        self.out_size = out_size
        self.upsampling_sizes = upsampling_sizes
        self.upsample = nn.ConvTranspose2d(latent_dims[-1], latent_dims[-2], upsampling_sizes[-1], 1)
        self.self_loops = nn.ModuleList([SelfLoop(latent_dims[-2], 2 * skip_dims[-1])])
        self.up_blocks = nn.ModuleList()
        assert (len(latent_dims) - 1) == len(skip_dims) == len(upsampling_sizes)
        for i in range(len(skip_dims) - 1, 0, -1):
            self.up_blocks.append(DeconvBlock(
                latent_dims[i] + skip_dims[i], latent_dims[i - 1], odd=upsampling_sizes[i - 1] % 2 != 0))
            self.self_loops.append(SelfLoop(latent_dims[i - 1], 2 * skip_dims[i - 1]))
        self.out_conv = DeconvBlock(
            latent_dims[0] + skip_dims[0], out_chans, odd=out_size % 2 != 0, final=True)

    def loop(self, i, x, skip=None):
        mom = self.self_loops[i](x)
        if skip is None:
            z = self.sample_z(mom)
        else:
            z = skip
        x = torch.cat([x, z], dim=1)
        return x, mom

    def sample_z(self, mom):
        mean, logvar = mom.chunk(2, dim=1)
        e = torch.randn_like(mean)
        return 0.5 * torch.exp(logvar) * e + mean

    def forward(self, z, skips):
        if skips is None:
            skips = [None for _ in range(len(self.skip_dims))]
        x = self.upsample(z)
        x, mom = self.loop(0, x, skips[-1])
        moms = [mom]
        for i in range(0, len(self.skip_dims) - 1):
            x = self.up_blocks[i](x)
            x, mom = self.loop(i + 1, x, skips[-(i + 2)])
            moms.append(mom)
        x = self.out_conv(x)
        return x, moms


class UVAE(nn.Module):

    def __init__(
            self,
            in_chans,
            latent_dims,
            skip_dims,
            res
    ):
        super(UVAE, self).__init__()
        self.latent_dims = latent_dims
        self.skip_dims = skip_dims
        self.res = res
        downsampling_sizes = []
        for i in range(len(skip_dims)):
            if i == 0:
                sz = res
            sz = math.floor((sz + 1) / 2)
            downsampling_sizes.append(sz)
        self.downsampling_sizes = downsampling_sizes
        self.encoder = Encoder(in_chans, latent_dims, skip_dims)
        self.decoder = Decoder(
            in_chans, latent_dims, skip_dims, out_size=res, upsampling_sizes=downsampling_sizes)
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="sum")

    def forward(self, x):
        moms = self.encode(x)
        z, skips = self.sample_z(moms)
        x_, loops = self.decode(z, skips)
        kld = 0
        for i, mom in enumerate(moms):
            mean, logvar = mom.chunk(2, dim=1)
            if i < len(moms) - 1:
                mean_, logvar_ = loops[-(i + 1)].chunk(2, dim=1)
                kld += kullback_leibler(mean, mean_, logvar, logvar_)
            else:
                kld += 0.5 * (torch.exp(logvar) + mean.pow(2) - logvar - 1).sum()
        reconst_loss = self.bce_loss(x_, x)
        return kld + reconst_loss

    def sample_z(self, moms):
        z = []
        for i, mom in enumerate(moms):
            mean, logvar = mom.chunk(2, dim=1)
            e = torch.randn_like(mean)
            z.append(0.5 * torch.exp(logvar) * e + mean)
        *skips, z = z
        return z, skips

    def encode(self, x):
        moms = self.encoder(x)
        return moms

    def decode(self, z, skips=None):
        x, moms = self.decoder(z, skips)
        return x, moms

    def reconst(self, x):
        moms = self.encode(x)
        z, skips = self.sample_z(moms)
        x_ = self.decode(z, skips)
        return x_

    def sample_x(self, n):
        device = next(iter(self.parameters())).device
        z = torch.randn(n, self.latent_dims[-1], 1, 1).to(device)
        x, _ = self.decode(z)
        return torch.sigmoid(x)


if __name__ == "__main__":
    res = 28
    in_chans = 1
    latent_dims = [32, 32, 64, 64]
    skip_dims = [5, 10, 16]
    model = UVAE(in_chans, latent_dims, skip_dims, res)
    print(model)
