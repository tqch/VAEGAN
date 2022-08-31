import os
import torch
import torch.nn as nn
from . import resnet
from . import dcgan

backbone = {"resnet": resnet, "dcgan": dcgan}[os.environ.get("BACKBONE", "dcgan")]


class VAE(nn.Module):
    def __init__(
            self,
            in_ch,
            base_ch,
            latent_dim,
            image_res,
            out_act="tanh",
            reconst_loss=nn.MSELoss(reduction="sum")
    ):
        super(VAE, self).__init__()
        self.in_ch = in_ch
        self.latent_dim = latent_dim
        self.image_res = image_res
        self.encoder = backbone.Encoder(in_ch, base_ch, out_dim=2 * latent_dim, image_res=image_res)
        out_act = {"tanh": torch.tanh, "sigmoid": torch.sigmoid}[out_act]
        self.decoder = backbone.Decoder(in_ch, base_ch, latent_dim, image_res=image_res, out_act=out_act)
        self.reconst_loss = reconst_loss

    def sample_noise(self, n):
        device = next(self.parameters()).device
        z = torch.randn((n, self.latent_dim), device=device)
        return z

    def sample_z(self, mean, logvar):
        e = torch.randn_like(mean)
        return e * torch.exp(0.5 * logvar) + mean

    def sample_x(self, n, noise=None):
        if noise is None:
            noise = self.sample_noise(n)
        return self.decode(noise)

    def encode(self, x):
        mom = self.encoder(x)
        mean, logvar = mom.chunk(2, dim=1)
        return mean, logvar

    def decode(self, noise):
        return self.decoder(noise)

    def reconst(self, x):
        mean, logvar = self.encode(x)
        z = self.sample_z(mean, logvar)
        return self.decode(z)

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.sample_z(mean, logvar)
        kld = 0.5 * (logvar.exp() + mean.pow(2) - logvar - 1).sum()
        x_ = self.decode(z)
        reconst_loss = self.reconst_loss(x_, x)
        # batch average + division by latent dimension
        return (kld + reconst_loss).div(x.shape[0] * self.latent_dim)


def test():
    image_res = (32, 32)
    in_ch = 3
    base_ch = 64
    latent_dim = 128
    vae = VAE(
        in_ch=in_ch,
        base_ch=base_ch,
        latent_dim=latent_dim,
        image_res=image_res[0],
    )
    print(vae)
    x = torch.randn(64, 3, 32, 32)
    vae(x)
    vae.sample_x(16)
