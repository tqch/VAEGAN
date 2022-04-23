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
    kld += - dif_logvar - 1
    kld *= 0.5
    return kld.sum()


class ConvBlock(nn.Module):
    def __init__(self, in_chans, out_chans, stride=1, final=False):
        super(ConvBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_chans)
        self.conv1 = nn.Conv2d(in_chans, out_chans, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_chans)
        self.conv2 = nn.Conv2d(out_chans, out_chans, 3, 1, 1, bias=final)
        if in_chans != out_chans or stride != 1:
            self.skip = nn.Conv2d(in_chans, out_chans, 1, stride, 0, bias=final)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        skip = self.skip(x)
        x = F.relu(self.bn1(x))
        x = F.relu(self.bn2(self.conv1(x)))
        x = self.conv2(x)
        return x + skip


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


class Passage(nn.Module):
    def __init__(self, in_chans, out_chans):
        super(Passage, self).__init__()
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
    def __init__(self, in_chans, latent_dims, layer_configs):
        super(Encoder, self).__init__()
        self.in_chans = in_chans
        self.latent_dims = latent_dims
        self.layer_configs = layer_configs
        self.in_conv = nn.Conv2d(in_chans, latent_dims[0], 3, 1, 1, bias=False)
        self.downs = nn.ModuleList([self.make_layer(latent_dims[0], latent_dims[0], layer_configs[0])])
        self.receptions = nn.ModuleList([Passage(2 * latent_dims[0], 2 * latent_dims[0])])
        assert (len(latent_dims) - 1) == len(layer_configs)
        for i in range(len(latent_dims) - 2):
            self.downs.append(self.make_layer(
                latent_dims[i], latent_dims[i + 1], layer_configs[i + 1]))
            self.receptions.append(Passage(2 * latent_dims[i + 1], 2 * latent_dims[i + 1]))
        self.final_layer = nn.Sequential(
            nn.BatchNorm2d(latent_dims[-2]),
            nn.ReLU(),
            nn.Conv2d(latent_dims[-2], 2 * latent_dims[-1], 3, 1, 1),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    @staticmethod
    def make_layer(in_chans, out_chans, num_blocks):
        return nn.Sequential(
            ConvBlock(in_chans, out_chans, stride=2),
            *[ConvBlock(out_chans, out_chans) for _ in range(num_blocks-1)]
        )

    def respond(self, i, site, feedback):
        rcpt = self.receptions[i]
        mom = rcpt(torch.cat([site, feedback], dim=1))
        response = self.sample_z(mom)
        return response, mom

    def sample_z(self, mom):
        mean, logvar = mom.chunk(2, dim=1)
        e = torch.randn_like(mean)
        return 0.5 * torch.exp(logvar) * e + mean

    def forward(self, x):
        x = self.in_conv(x)
        sites = []
        for layer in self.downs:
            x = layer(x)
            sites.append(x)
        mom = self.final_layer(x)
        return mom, sites


class Decoder(nn.Module):
    def __init__(self, out_chans, latent_dims, layer_configs, out_size, upsampling_sizes, drop_rate):
        super(Decoder, self).__init__()
        self.out_chans = out_chans
        self.latent_dims = latent_dims
        self.layer_configs = layer_configs
        self.out_size = out_size
        self.upsampling_sizes = upsampling_sizes
        self.drop_rate = drop_rate
        self.upsample = nn.ConvTranspose2d(latent_dims[-1], latent_dims[-2], upsampling_sizes[-1], 1)
        self.bootstraps = nn.ModuleList([Passage(latent_dims[-2], 2 * latent_dims[-2])])
        self.tanh_scales = nn.ParameterList([nn.Parameter(torch.zeros(1, latent_dims[-2], 1, 1))])
        self.ups = nn.ModuleList()
        assert (len(latent_dims) - 1) == len(upsampling_sizes) == len(layer_configs)
        for i in range(len(latent_dims) - 2, 0, -1):
            self.ups.append(self.make_layer(
                latent_dims[i], latent_dims[i - 1],
                odd=upsampling_sizes[i - 1] % 2 != 0,
                num_blocks=layer_configs[i]
            ))
            self.bootstraps.append(Passage(latent_dims[i - 1], 2 * latent_dims[i - 1]))
            self.tanh_scales.append(nn.Parameter(torch.zeros(1, latent_dims[i - 1], 1, 1)))
        num_blocks = layer_configs[0]
        if num_blocks == 1:
            self.out_conv = DeconvBlock(
                latent_dims[0], out_chans, odd=out_size % 2 != 0, final=True)
        else:
            self.out_conv = nn.Sequential(
                DeconvBlock(latent_dims[0], out_chans, odd=out_size % 2 != 0),
                *[ConvBlock(out_chans, out_chans) for _ in range(num_blocks-2)],
                ConvBlock(out_chans, out_chans, final=True)
            )
        self.drop_out = BatchDropout(drop_rate=drop_rate)

    @staticmethod
    def make_layer(in_chans, out_chans, odd, num_blocks):
        return nn.Sequential(
            DeconvBlock(in_chans, out_chans, odd),
            *[ConvBlock(out_chans, out_chans) for _ in range(num_blocks - 1)]
        )

    def bootstrap(self, i, x, sample=True):
        mom = self.bootstraps[i](x)
        response = None
        if sample:
            response = self.sample_z(mom)
        return response, mom

    def feedback(self, i, x, response):
        if i != len(self.latent_dims) - 2:
            moveup = self.ups[i]
        else:
            moveup = self.out_conv
        response = self.tanh_scales[i] * torch.tanh(response)
        response = self.drop_out(response)
        x = moveup(x + response)
        return x

    def sample_z(self, mom):
        mean, logvar = mom.chunk(2, dim=1)
        e = torch.randn_like(mean)
        return 0.5 * torch.exp(logvar) * e + mean

    def forward(self, z):
        x = self.upsample(z)
        moms = []
        for i in range(len(self.latent_dims)-1):
            response, mom = self.bootstrap(i, x)
            moms.append(mom)
            x = self.feedback(i, x, response)
        return x


class UVAE(nn.Module):

    def __init__(
            self,
            in_chans,
            latent_dims,
            layer_configs,
            image_res,
            response_drop_rate=0.5
    ):
        super(UVAE, self).__init__()
        self.latent_dims = latent_dims
        self.layer_configs = layer_configs
        self.image_res = image_res
        downsampling_sizes = []
        for i in range(len(latent_dims)-1):
            if i == 0:
                sz = image_res
            sz = math.floor((sz + 1) / 2)
            downsampling_sizes.append(sz)
        self.downsampling_sizes = downsampling_sizes
        self.encoder = Encoder(in_chans, latent_dims, layer_configs)
        self.decoder = Decoder(
            in_chans, latent_dims, layer_configs, out_size=image_res,
            upsampling_sizes=downsampling_sizes, drop_rate=response_drop_rate)
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="sum")

    def forward(self, x):
        mom, sites = self.encode(x)
        mean, logvar = mom.chunk(2, dim=1)
        kld = 0.5 * (torch.exp(logvar) + mean.pow(2) - logvar - 1).sum()
        z = self.sample_z(mom)
        x_, moms, moms_ = self.decode(z, sites)
        for mom, mom_ in zip(moms, moms_):
            mean, logvar = mom.chunk(2, dim=1)
            mean_, logvar_ = mom_.chunk(2, dim=1)
            kld += kullback_leibler(mean, mean_, logvar, logvar_)
        reconst_loss = self.bce_loss(x_, x)
        return kld + reconst_loss

    def sample_z(self, mom):
        mean, logvar = mom.chunk(2, dim=1)
        e = torch.randn_like(mean)
        return 0.5 * torch.exp(logvar) * e + mean

    def encode(self, x):
        mom, sites = self.encoder(x)
        return mom, sites

    def decode(self, z, sites=None):
        if sites is not None:
            moms = []
            moms_ = []
            x = self.decoder.upsample(z)
            for i in range(len(self.latent_dims)-1):
                site = sites[-(i+1)]
                response, mom = self.encoder.respond(-(i+1), site, x)
                moms.append(mom)
                _, mom_ = self.decoder.bootstrap(i, x, sample=False)
                moms_.append(mom_)
                x = self.decoder.feedback(i, x, response)
            return x, moms, moms_
        else:
            x = self.decoder.upsample(z)
            for i in range(len(self.latent_dims)-1):
                response, _ = self.decoder.bootstrap(i, x)
                x = self.decoder.feedback(i, x, response)
            return x

    def reconst(self, x):
        mom, sites = self.encode(x)
        z = self.sample_z(mom)
        x_, *_ = self.decode(z, sites)
        return torch.sigmoid(x_)

    def sample_x(self, n):
        device = next(iter(self.parameters())).device
        z = torch.randn(n, self.latent_dims[-1], 1, 1).to(device)
        x = self.decoder(z)
        return torch.sigmoid(x)


if __name__ == "__main__":
    image_res = 28
    in_chans = 1
    latent_dims = [32, 32, 64, 64]
    layer_configs = [2, 2, 2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UVAE(in_chans, latent_dims, layer_configs, image_res)
    print(model)
    model.to(device)
    x = torch.randn(16, 1, 28, 28).to(device)
    # model(x)
    z = torch.randn(16, 64, 1, 1).to(device)
    model.decode(z)
