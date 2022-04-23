import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import PartialBatchNorm2d, BatchDropout


def frechet_distance(mean_1, mean_2, logvar_1, logvar_2):
    # frechet distance for two diagnoal Gaussians
    dif_mean_sq = (mean_1 - mean_2).pow(2)
    dif_var = torch.exp(logvar_1) + torch.exp(logvar_2) - torch.exp(0.5 * (logvar_1 + logvar_2))
    return (dif_mean_sq + dif_var).sum()


class ConvBlock(nn.Module):
    def __init__(self, in_chans, out_chans):
        super(ConvBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_chans)
        self.conv1 = nn.Conv2d(in_chans, out_chans, 3, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_chans)
        self.conv2 = nn.Conv2d(out_chans, out_chans, 3, 1, 1, bias=False)

    def forward(self, x):
        x = F.relu(self.bn1(x))
        x = F.relu(self.bn2(self.conv1(x)))
        x = self.conv2(x)
        return x


class DeconvBlock(nn.Module):
    def __init__(self, in_chans, out_chans, splits=None):
        super(DeconvBlock, self).__init__()
        if splits is None:
            self.bn1 = nn.BatchNorm2d(in_chans)
        else:
            self.bn1 = PartialBatchNorm2d(splits)
        self.deconv1 = nn.ConvTranspose2d(in_chans, in_chans, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_chans)
        self.deconv2 = nn.ConvTranspose2d(in_chans, out_chans, 2, 2, 0, bias=False)

    def forward(self, x):
        x = F.relu(self.bn1(x))
        x = F.relu(self.bn2(self.deconv1(x)))
        x = self.deconv2(x)
        return x


class SkipConnection(nn.Module):
    def __init__(self, in_chans, out_chans, drop_prob=0):
        super(SkipConnection, self).__init__()
        self.in_bn = nn.BatchNorm2d(in_chans)
        self.conv = nn.Conv2d(in_chans, out_chans, 1, 1, 0)
        self.out_bn = nn.BatchNorm2d(out_chans)
        assert drop_prob >= 0
        if drop_prob == 0:
            self.drop_out = nn.Identity()
        else:
            self.drop_out = BatchDropout(drop_prob=drop_prob)

    def forward(self, x):
        x = F.relu(self.in_bn(x))
        x = self.drop_out(self.out_bn(self.conv(x)))
        return x


class Encoder(nn.Module):
    def __init__(self, latent_dim, skip_dims, drop_prob):
        super(Encoder, self).__init__()
        self.drop_prob = drop_prob
        self.block1 = ConvBlock(1, latent_dim // 2)  # 28 -> 14
        self.skip1 = SkipConnection(latent_dim // 2, skip_dims[0], drop_prob)
        self.block2 = ConvBlock(latent_dim // 2, latent_dim)  # 14 -> 7
        self.skip2 = SkipConnection(latent_dim, skip_dims[1], drop_prob)
        self.block3 = ConvBlock(latent_dim, latent_dim * 2)  # 7 -> 4
        self.skip3 = SkipConnection(latent_dim * 2, skip_dims[2], drop_prob)
        self.max_pool = nn.MaxPool2d(4)

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.max_pool(x3)
        skip1 = self.skip1(x1)
        skip2 = self.skip2(x2)
        skip3 = self.skip3(x3)
        return x4, skip1, skip2, skip3


class Decoder(nn.Module):
    def __init__(self, latent_dim, skip_dims):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.upsample = nn.ConvTranspose2d(latent_dim, latent_dim, 4, 1)  # 1 -> 4
        self.block1 = nn.Sequential(
            nn.BatchNorm2d(latent_dim + skip_dims[0]),
            nn.ReLU(),
            nn.ConvTranspose2d(latent_dim + skip_dims[0], latent_dim, 3, 1, 1),
            nn.BatchNorm2d(latent_dim),
            nn.ReLU(),
            nn.ConvTranspose2d(latent_dim, latent_dim, 3, 2, 1)
        )  # 4 -> 7
        self.block2 = DeconvBlock(latent_dim + skip_dims[1], latent_dim // 2)  # 7 -> 14
        self.block3 = DeconvBlock(latent_dim // 2 + skip_dims[2], 1)  # 14 -> 28
        self.skip3 = SkipConnection(latent_dim, skip_dims[0])
        self.skip2 = SkipConnection(latent_dim, skip_dims[1])
        self.skip1 = SkipConnection(latent_dim//2, skip_dims[2])

    def in_skip(self, block, skip_connection, x, skip=None):
        x = block(x)
        if skip is None:
            skip = skip_connection(x)
        x = torch.cat([x, skip], dim=1)
        return x

    def forward(self, z, skips):
        if skips is None:
            skips = (None, None, None)
        skip1, skip2, skip3 = skips
        x4 = self.in_skip(self.upsample, self.skip3, z, skip3)
        x3 = self.in_skip(self.block1, self.skip2, x4, skip2)
        x2 = self.in_skip(self.block2, self.skip1, x3, skip1)
        x1 = self.block3(x2)
        return x1


class UVAE(nn.Module):
    SIZES = [14, 7, 4, 1]

    def __init__(self, latent_dim, skip_dims, drop_prob=0.5):
        super(UVAE, self).__init__()
        self.latent_dim = latent_dim
        self.skip_dims = skip_dims
        self.drop_prob = drop_prob
        self.encoder = Encoder(latent_dim, skip_dims, drop_prob)
        self.decoder = Decoder(latent_dim, skip_dims)
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="sum")

    def forward(self, x):
        mom, skips = self.encode(x)
        z = self.sample_z(x.shape[0], mom)
        mean, logvar = mom.chunk(2, dim=1)
        kld = 0.5 * (torch.exp(logvar) + mean.pow(2) - logvar - 1).sum()
        x_ = self.decoder(z, skips)
        reconst_loss = self.bce_loss(x_, x)
        return kld + reconst_loss

    def sample_z(self, n, mom):
        device = next(iter(self.parameters())).device
        mean, logvar = mom.chunk(2, dim=1)
        e = torch.randn(n, self.latent_dim, 1, 1).to(device)
        z = 0.5 * torch.exp(logvar) * e + mean
        return z

    def encode(self, x):
        mom, *skips = self.encoder(x)
        return mom, skips

    def decode(self, z, skips=None):
        # if skips is None:
        #     skips = [torch.zeros(z.shape[0], d, sz, sz).to(z.device)
        #              for d, sz in zip(self.skip_dims, self.SIZES)]
        x = self.decoder(z, skips)
        return torch.sigmoid(x)

    def reconst(self, x):
        mom, *skips = self.encode(x)
        z = self.sample_z(x.shape[0], mom)
        x_ = self.decode(z)
        return x_


class DeblurUVAE(UVAE):
    def __init__(self, latent_dim, skip_dims, transform, alpha=1):
        super(DeblurUVAE, self).__init__(latent_dim, skip_dims)
        self.transform = transform
        self.alpha = alpha

    def forward(self, x):
        mom, skips = self.encode(x)
        z = self.sample_z(x.shape[0], mom)
        mean, logvar = mom.chunk(2, dim=1)
        # kl(q(z|x)||p(z))
        kld = 0.5 * (torch.exp(logvar) + mean.pow(2) - logvar - 1).sum()
        # frechet distance penalty on conditional distribution shift in latent space
        mom_blur, skips_blur = self.encode(self.transform(x))
        mean_blur, logvar_blur = mom_blur.chunk(2, dim=1)
        fd = frechet_distance(mean, mean_blur, logvar, logvar_blur)
        # reconstruction loss
        x_ = self.decoder(z, skips_blur)
        reconst_loss = self.bce_loss(x_, x)
        return kld + reconst_loss + self.alpha * fd
