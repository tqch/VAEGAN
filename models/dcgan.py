import torch
import torch.nn as nn
from modules import Reshape


class Discriminator(nn.Module):
    def __init__(self, in_chan, hidden_dims, image_res):
        super(Discriminator, self).__init__()
        self.in_chan = in_chan
        self.hidden_dims = hidden_dims
        self.in_layer = nn.Sequential(
            nn.Conv2d(in_chan, hidden_dims[0], 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.conv_layers.append(self.make_layer(hidden_dims[i], hidden_dims[i + 1]))
        multiplier = image_res // 16
        self.out_layer = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(hidden_dims[-1] * multiplier ** 2, 1),
            nn.Sigmoid()
        )

    def make_layer(self, in_chan, out_chan):
        return nn.Sequential(
            nn.Conv2d(in_chan, out_chan, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_chan),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        x = self.in_layer(x)
        for layer in self.conv_layers:
            x = layer(x)
        x = self.out_layer(x)
        return x


class Generator(nn.Module):
    def __init__(self, out_chan, latent_dim, hidden_dims, image_res):
        super(Generator, self).__init__()
        self.out_chan = out_chan
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        multiplier = image_res // 16
        self.in_layer = nn.Sequential(
            nn.Linear(latent_dim, multiplier ** 2 * hidden_dims[0], bias=False),
            Reshape((-1, hidden_dims[0], 2, 2)),
            nn.BatchNorm2d(hidden_dims[0]),
            nn.ReLU(inplace=True)
        )
        self.deconv_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.deconv_layers.append(self.make_layer(
                hidden_dims[i], hidden_dims[i + 1]))
        self.out_layer = nn.ConvTranspose2d(hidden_dims[-1], out_chan, 4, 2, 1)

    @staticmethod
    def make_layer(in_chan, out_chan):
        return nn.Sequential(
            nn.ConvTranspose2d(in_chan, out_chan, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True)
        )

    def forward(self, z):
        x = self.in_layer(z)
        for layer in self.deconv_layers:
            x = layer(x)
        x = torch.sigmoid(self.out_layer(x))
        return x


class DCGAN(nn.Module):
    def __init__(
            self,
            in_chan,
            latent_dim,
            image_res,
            configs
    ):
        super(DCGAN, self).__init__()
        self.latent_dim = latent_dim
        self.configs = configs
        self.image_res = image_res
        self.discriminator = Discriminator(in_chan, image_res=image_res, **configs["D"])
        self.generator = Generator(
            in_chan, latent_dim=latent_dim, image_res=image_res, **configs["G"])
        self.bce_loss = nn.BCELoss(reduction="sum")

    def forward(self, x=None, n=None, dis=True):
        if dis:
            x_ = self.sample_x(x.shape[0]).detach()
            # discriminator scores
            ds = self.discriminator(x)
            ds_ = self.discriminator(x_)
            # artificial labels
            y = torch.ones_like(ds)
            y_ = torch.zeros_like(ds_)
            gan_loss = self.bce_loss(ds, y)
            gan_loss += self.bce_loss(ds_, y_)
            return gan_loss / 2
        else:
            x_ = self.sample_x(n)
            ds_ = self.discriminator(x_)
            y_ = torch.ones_like(ds_)
            # reduce gan loss w.r.t. the generator
            # the effect is to increase discriminator scores of fake examples
            gan_loss = self.bce_loss(ds_, y_)
            return gan_loss

    def sample_x(self, n):
        z = self.sample_noise(n)
        x = self.generator(z)
        return x

    def sample_noise(self, n):
        device = next(iter(self.parameters())).device
        z = torch.randn(n, self.latent_dim).to(device)
        return z


if __name__ == "__main__":
    image_res = (32, 32)
    n_channels = 3
    latent_dim = 128
    configs = {
        "D": {"hidden_dims": [64, 128, 256, 512]},
        "G": {"hidden_dims": [512, 256, 128, 64]}
    }
    dcgan = DCGAN(
        in_chan=n_channels,
        latent_dim=latent_dim,
        image_res=image_res[0],
        configs=configs
    )
    print(dcgan)
    x = torch.randn(64, 3, 32, 32)
    dcgan(x)
    dcgan.sample_x(16)
