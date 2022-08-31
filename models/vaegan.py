import os
import torch
import torch.nn as nn
from . import resnet
from . import dcgan
from functools import partial

backbone = {"resnet": resnet, "dcgan": dcgan}[os.environ.get("BACKBONE", "dcgan")]

if backbone.__name__ == "models.resnet":
    class _Encoder(backbone.Encoder):
        block = partial(backbone.ResidualBlock, resample="downsample", normalize=False)
else:
    _Encoder = backbone.Encoder

from .unet import UNet


class Decoder(backbone.Decoder):
    def __init__(
            self,
            out_ch,
            base_ch,
            latent_dim,
            reconst_ch,
            image_res,
            out_act
    ):
        super(Decoder, self).__init__(out_ch, base_ch, latent_dim, image_res=image_res, out_act=out_act)
        self.reconst = UNet(out_ch, reconst_ch, image_res, out_act=out_act)
        self.out_act = out_act

    def forward(self, x, reconst=False):
        out = super().forward(x)
        if reconst:
            out = self.reconst(out)
        return out


if backbone.__name__.split(".")[-1] == "dcgan" and os.environ.get("ANTI_ARTIFACT", ""):
    class _Decoder(Decoder):
        @staticmethod
        def make_layer(in_ch, out_ch):
            return backbone.AntiArtifactDecoder.make_layer(in_ch, out_ch)
else:
    _Decoder = Decoder


class VAEGAN(nn.Module):
    def __init__(
            self,
            in_ch,
            base_ch,
            latent_dim,
            reconst_ch,
            image_res,
            out_act="tanh",
            reconst_loss=nn.MSELoss(reduction="sum"),
            noise_scale=0.,
            anti_artifact=True,  # only effective when using dcgan backbone
    ):
        super(VAEGAN, self).__init__()
        self.in_ch = in_ch
        self.latent_dim = latent_dim
        self.image_res = image_res
        self.encoder = backbone.Encoder(in_ch, base_ch, out_dim=2 * latent_dim, image_res=image_res)
        out_act = {"tanh": torch.tanh, "sigmoid": torch.sigmoid}[out_act]
        self.decoder = (_Decoder if anti_artifact else Decoder)(
            in_ch, base_ch, latent_dim=latent_dim, reconst_ch=reconst_ch, image_res=image_res, out_act=out_act)
        self.netD = _Encoder(in_ch, base_ch, image_res=image_res)
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="mean")
        self.reconst_loss = reconst_loss
        self.register_buffer("noise_scale", torch.as_tensor(noise_scale))

    def forward(self, x, component="vae"):
        assert component in {"vae", "netD"}
        B = x.shape[0]
        if component == "vae":
            ####################
            # posterior sampling
            ####################
            mom = self.encoder(x)
            mean, logvar = mom.chunk(2, dim=1)
            noise = self.sample_z(mean, logvar)
            kld = 0.5 * (logvar.exp() + mean.pow(2) - logvar - 1).sum()
            x_ = self.decode(noise, reconst=True)
            reconst_loss = self.reconst_loss(x_, x)
            vae_loss = (kld + reconst_loss).div(B * self.latent_dim)
            ################
            # prior sampling
            ################
            x_ = self.sample_x(B)
            # add instance noise
            if self.noise_scale:
                x_ = self.add_noise(x_)
            ds_ = self.netD(x_)
            gen_loss = self.bce_loss(ds_, torch.ones_like(ds_))
            return vae_loss, gen_loss
        elif component == "netD":
            with torch.no_grad():
                x_ = self.sample_x(B, reconst=False)
            # add instance noise
            if self.noise_scale:
                x = self.add_noise(x)
                x_ = self.add_noise(x_)
            # calculate the discriminator scores
            ds = self.netD(x)  # real
            ds_ = self.netD(x_)  # fake
            dis_loss = self.bce_loss(ds, torch.ones_like(ds))
            dis_loss += self.bce_loss(ds_, torch.zeros_like(ds_))
            return dis_loss
        else:
            raise NotImplementedError

    def add_noise(self, x):
        return x + self.noise_scale * torch.randn_like(x)

    def sample_noise(self, n, dim):
        device = next(self.parameters()).device
        return torch.randn((n, dim), device=device)

    def sample_z(self, mean, logvar):
        eps = torch.randn_like(mean)
        return eps * torch.exp(0.5 * logvar) + mean

    def sample_x(self, n, noise=None, reconst=False):
        if noise is None:
            noise = self.sample_noise(n, self.latent_dim)
        return self.decode(noise, reconst=reconst)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, noise, reconst=True):
        return self.decoder(noise, reconst=reconst)

    def reconst(self, x):
        mean, logvar = self.encode(x).chunk(2, dim=1)
        z = self.sample_z(mean, logvar)
        return self.decode(z, reconst=True)


def test():
    image_res = (64, 64)
    in_ch = 3
    base_ch = 64
    latent_dim = 128
    reconst_ch = 64
    vaegan = VAEGAN(
        in_ch=in_ch,
        base_ch=base_ch,
        latent_dim=latent_dim,
        reconst_ch=reconst_ch,
        image_res=image_res[0]
    )
    print(vaegan)
    x = torch.randn(64, 3, 64, 64)
    vaegan(x)
    vaegan.sample_x(64)
