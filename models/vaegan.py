import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import Reshape


class Encoder(nn.Module):
    def __init__(self, in_chan, latent_dim, hidden_dims, image_res):
        super(Encoder, self).__init__()
        self.in_chan = in_chan
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.image_res = image_res

        self.in_layer = nn.Sequential(
            nn.Conv2d(in_chan, hidden_dims[0], 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.mid_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.mid_layers.append(
                self.make_mid_layer(hidden_dims[i], hidden_dims[i + 1]))
        self.out_layer = self.make_out_layer()

    def make_out_layer(self):
        multiplier = self.image_res // 16
        return nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(multiplier ** 2 * self.hidden_dims[-1], 2 * self.latent_dim)
        )

    def make_mid_layer(self, in_chan, out_chan):
        return nn.Sequential(
            nn.Conv2d(in_chan, out_chan, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_chan),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        x = self.in_layer(x)
        for layer in self.mid_layers:
            x = layer(x)
        x = self.out_layer(x)
        return x


class Discriminator(Encoder):
    def __init__(self, in_chan, hidden_dims, image_res):
        super(Discriminator, self).__init__(
            in_chan, latent_dim=1, hidden_dims=hidden_dims, image_res=image_res)

    def make_out_layer(self):
        multiplier = self.image_res // 16
        return nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(multiplier ** 2 * self.hidden_dims[-1], 1)
        )


class ResidualBlock(nn.Module):
    def __init__(self, in_chan, mid_chan, out_chan, stride=1, bias=False):
        super(ResidualBlock, self).__init__()
        self.in_bn = nn.BatchNorm2d(in_chan)
        self.in_conv = nn.Conv2d(in_chan, mid_chan, 3, stride, 1, bias=False)
        self.mid_bn = nn.BatchNorm2d(mid_chan)
        self.mid_conv = nn.Conv2d(mid_chan, mid_chan, 3, 1, 1, bias=False)
        self.out_bn = nn.BatchNorm2d(mid_chan)
        self.out_conv = nn.Conv2d(mid_chan, out_chan, 3, 1, 1, bias=bias)
        if in_chan != out_chan or stride != 1:
            self.skip = nn.Conv2d(in_chan, out_chan, 1, stride, 0, bias=bias)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        skip = self.skip(x)
        x = self.in_conv(F.relu(self.in_bn(x), inplace=True))
        x = self.mid_conv(F.relu(self.mid_bn(x), inplace=True))
        x = self.out_conv(F.relu(self.out_bn(x), inplace=True))
        return x + skip


class Decoder(nn.Module):
    def __init__(
            self,
            out_chan,
            latent_dim,
            hidden_dims,
            image_res,
            refine_mid_dim=64,
            num_refine_blocks=1
    ):
        super(Decoder, self).__init__()
        self.out_chan = out_chan
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        multiplier = image_res // 16
        self.in_layer = nn.Sequential(
            nn.Linear(latent_dim, multiplier ** 2 * hidden_dims[0], bias=False),
            Reshape((-1, hidden_dims[0], multiplier, multiplier)),
            nn.BatchNorm2d(hidden_dims[0]),
            nn.ReLU(inplace=True)
        )
        self.mid_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.mid_layers.append(self.make_mid_layer(
                hidden_dims[i], hidden_dims[i + 1]))
        self.crude = nn.ConvTranspose2d(hidden_dims[-1], out_chan, 4, 2, 1)
        self.refine_mid_dim = refine_mid_dim
        self.num_refine_blocks = num_refine_blocks
        self.refine = self.make_refine_layer()

    @staticmethod
    def make_mid_layer(in_chan, out_chan):
        return nn.Sequential(
            nn.ConvTranspose2d(in_chan, out_chan, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True)
        )

    def make_refine_layer(self):
        if self.num_refine_blocks == 1:
            return ResidualBlock(
                self.out_chan,
                self.refine_mid_dim,
                self.out_chan
            )
        else:
            blocks = [ResidualBlock(
                self.out_chan, self.refine_mid_dim, self.refine_mid_dim)]
            for i in range(self.num_refine_blocks - 1):
                out_chan = self.refine_mid_dim
                if i == self.num_refine_blocks - 2:
                    out_chan = self.out_chan
                blocks.append(ResidualBlock(
                    self.refine_mid_dim,
                    self.refine_mid_dim,
                    out_chan,
                    bias=True
                ))
            return nn.Sequential(*blocks)

    def forward(self, z):
        x = self.in_layer(z)
        for layer in self.mid_layers:
            x = layer(x)
        x = self.crude(x)
        out_crude = torch.sigmoid(x)
        out_refined = torch.sigmoid(self.refine(x))  # stop gradient
        return out_crude, out_refined


class VAEGAN(nn.Module):
    def __init__(
            self,
            in_chan,
            latent_dim,
            configs,
            image_res,
            l2_reg_coef=5e-4,
            reconst_loss=nn.MSELoss(reduction="sum")
    ):
        super(VAEGAN, self).__init__()
        self.in_chan = in_chan
        self.latent_dim = latent_dim
        self.image_res = image_res
        self.encoder = Encoder(
            in_chan, latent_dim, image_res=image_res, **configs["Enc"])
        self.decoder = Decoder(
            in_chan, latent_dim, image_res=image_res, **configs["Dec"])
        self.discriminator = Discriminator(
            in_chan, image_res=image_res, **configs["Dis"])
        self.l2_reg_coef = l2_reg_coef
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="sum")
        self.reconst_loss = reconst_loss

    def forward(self, x=None, component="vae"):
        if component == "vae":
            mom = self.encoder(x)
            mean, logvar = mom.chunk(2, dim=1)
            z = self.sample_z(mean, logvar)
            kld = 0.5 * (logvar.exp() + mean.pow(2) - logvar - 1).sum()
            x_crude, x_refined = self.decode(z)
            reconst_loss = self.reconst_loss(x_crude, x)
            ds = self.discriminator(x_refined)
            gen_loss = self.bce_loss(ds, torch.ones_like(ds))
            l2_reg = self.l2_reg()
            return kld + reconst_loss, gen_loss, l2_reg * self.l2_reg_coef
        elif component == "discriminator":
            mom = self.encoder(x)
            mean, logvar = mom.chunk(2, dim=1)
            z = self.sample_z(mean, logvar)
            _, x_ = self.decode(z)
            # calculate the discriminator scores
            ds = self.discriminator(x)  # real
            ds_ = self.discriminator(x_.detach())  # fake
            dis_loss = self.bce_loss(ds, torch.ones_like(ds))
            dis_loss += self.bce_loss(ds_, torch.zeros_like(ds_))
            return dis_loss * 0.5
        else:
            raise NotImplementedError

    def l2_reg(self):
        l2_reg = 0
        for p in self.decoder.refine.parameters():
            if p.requires_grad:
                l2_reg += p.pow(2).sum()
        return l2_reg

    def sample_noise(self, n):
        device = next(self.parameters()).device
        z = torch.rand(n, self.latent_dim).to(device)
        return z

    def sample_z(self, mean, logvar):
        e = torch.randn_like(mean)
        return e * torch.exp(0.5 * logvar) + mean

    def sample_x(self, n):
        z = self.sample_noise(n)
        _, x = self.decode(z)
        return x

    def encode(self, x):
        mom = self.encoder(x)
        return mom

    def decode(self, z):
        x_crude, x_refined = self.decoder(z)
        return x_crude, x_refined

    def reconst(self, x):
        mean, logvar = self.encode(x).chunk(2, dim=1)
        z = self.sample_z(mean, logvar)
        _, x = self.decode(z)
        return x


if __name__ == "__main__":
    image_res = (32, 32)
    n_channels = 3
    latent_dim = 128
    configs = {
        "Dec": {
            "hidden_dims": [512, 256, 128, 64],
            "refine_mid_dim": 128,
            "num_refine_blocks": 1
        },
        "Enc": {"hidden_dims": [64, 128, 256, 512]},
        "Dis": {"hidden_dims": [64, 128, 256, 512]}
    }
    vaegan = VAEGAN(
        in_chan=n_channels,
        latent_dim=latent_dim,
        image_res=image_res[0],
        configs=configs
    )
    print(vaegan)
    x = torch.randn(64, 3, 32, 32)
    vaegan(x)
    vaegan.sample_x(16)
