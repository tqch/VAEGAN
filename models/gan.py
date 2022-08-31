import os
import torch
import torch.nn as nn
from . import resnet
from . import dcgan
from functools import partial

backbone = {"resnet": resnet, "dcgan": dcgan}[os.environ.get("BACKBONE", "dcgan")]

if backbone.__name__ == "models.resnet":
    class Encoder(backbone.Encoder):
        block = partial(backbone.ResidualBlock, resample="downsample", normalize=False)
else:
    Encoder = backbone.Encoder


class GAN(nn.Module):
    def __init__(
            self,
            in_ch,
            base_ch,
            latent_dim,
            image_res,
            out_act="tanh",
            noise_scale=0.
    ):
        super(GAN, self).__init__()
        self.latent_dim = latent_dim
        self.image_res = image_res
        self.netD = Encoder(in_ch, base_ch, image_res=image_res)
        out_act = {"tanh": torch.tanh, "sigmoid": torch.sigmoid}[out_act]
        self.netG = backbone.Decoder(in_ch, base_ch, latent_dim=latent_dim, image_res=image_res, out_act=out_act)
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="mean")
        self.register_buffer("noise_scale", torch.as_tensor(noise_scale))

    def forward(self, x=None, n=None, dis=True):
        if dis:
            x_ = self.sample_x(x.shape[0]).detach()
            # add instance noise
            if self.noise_scale:
                x = self.add_noise(x)
                x_ = self.add_noise(x_)
            # discriminator scores
            ds = self.netD(x)
            ds_ = self.netD(x_)
            # artificial labels
            y = torch.ones_like(ds)
            y_ = torch.zeros_like(ds_)
            dis_loss = self.bce_loss(ds, y)
            dis_loss += self.bce_loss(ds_, y_)
            return dis_loss
        else:
            x_ = self.sample_x(n)
            # add instance noise
            if self.noise_scale:
                x_ = self.add_noise(x_)
            ds_ = self.netD(x_)
            y_ = torch.ones_like(ds_)
            gen_loss = self.bce_loss(ds_, y_)
            return gen_loss

    def add_noise(self, x):
        return x + self.noise_scale * torch.randn_like(x)

    def sample_x(self, n, noise=None):
        if noise is None:
            noise = self.sample_noise(n)
        return self.netG(noise)

    def sample_noise(self, n):
        device = next(iter(self.parameters())).device
        z = torch.randn((n, self.latent_dim), device=device)
        return z


def test():
    image_res = (64, 64)
    in_ch = 3
    base_ch = 64
    latent_dim = 128
    model = GAN(
        in_ch=in_ch,
        base_ch=base_ch,
        latent_dim=latent_dim,
        image_res=image_res[0]
    )
    print(model)
    x = torch.randn(64, 3, 64, 64)
    model(x)
    model.sample_x(16)
    model(x).backward()
