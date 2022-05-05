import torch
import torch.nn as nn
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


class Decoder(nn.Module):
    def __init__(
            self,
            out_chan,
            latent_dim,
            hidden_dims,
            image_res
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
        self.out_layer = nn.ConvTranspose2d(hidden_dims[-1], out_chan, 4, 2, 1)

    @staticmethod
    def make_mid_layer(in_chan, out_chan):
        return nn.Sequential(
            nn.ConvTranspose2d(in_chan, out_chan, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True)
        )

    def forward(self, z):
        x = self.in_layer(z)
        for layer in self.mid_layers:
            x = layer(x)
        x = self.out_layer(x)
        x = torch.sigmoid(x)
        return x


class VAE(nn.Module):
    def __init__(
            self,
            in_chan,
            latent_dim,
            image_res,
            configs,
            reconst_loss=nn.MSELoss(reduction="sum")
    ):
        super(VAE, self).__init__()
        self.in_chan = in_chan
        self.latent_dim = latent_dim
        self.image_res = image_res
        self.encoder = Encoder(
            in_chan, latent_dim, image_res=image_res, **configs["E"])
        self.decoder = Decoder(
            in_chan, latent_dim, image_res=image_res, **configs["D"])
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="sum")
        self.reconst_loss = reconst_loss

    def sample_noise(self, n):
        device = next(self.parameters()).device
        z = torch.randn(n, self.latent_dim).to(device)
        return z

    def sample_z(self, mean, logvar):
        e = torch.randn_like(mean)
        return e * torch.exp(0.5 * logvar) + mean

    def sample_x(self, n):
        z = self.sample_noise(n)
        x = self.decode(z)
        return x

    def encode(self, x):
        mom = self.encoder(x)
        mean, logvar = mom.chunk(2, dim=1)
        return mean, logvar

    def decode(self, z):
        x = self.decoder(z)
        return x

    def reconst(self, x):
        mean, logvar = self.encode(x)
        z = self.sample_z(mean, logvar)
        x = self.decode(z)
        return x

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.sample_z(mean, logvar)
        kld = 0.5 * (logvar.exp() + mean.pow(2) - logvar - 1).sum()
        x_ = self.decode(z)
        reconst_loss = self.reconst_loss(x_, x)
        return kld + reconst_loss


if __name__ == "__main__":
    image_res = (32, 32)
    n_channels = 3
    latent_dim = 128
    configs = {
        "E": {"hidden_dims": [64, 128, 256, 512]},
        "D": {"hidden_dims": [512, 256, 128, 64]}
    }
    vae = VAE(
        in_chan=n_channels,
        latent_dim=latent_dim,
        image_res=image_res[0],
        configs=configs
    )
    print(vae)
    x = torch.randn(64, 3, 32, 32)
    vae(x)
    vae.sample_x(16)
