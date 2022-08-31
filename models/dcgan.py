import math
import torch
import torch.nn as nn
from modules import Reshape


class Encoder(nn.Module):
    def __init__(self, in_ch, base_ch, out_dim=1, image_res=64):
        super(Encoder, self).__init__()
        self.in_ch = in_ch
        self.base_ch = base_ch
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layers = nn.ModuleList()
        num_layers = max(math.ceil(math.log(image_res, 2)) - 3, 3)
        spatial_size = image_res // 2 ** (num_layers + 1)
        for i in range(num_layers):
            self.layers.append(self.make_layer(base_ch * 2 ** i, base_ch * 2 ** (i + 1)))
        self.out_fc = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(base_ch * 2 ** num_layers * spatial_size ** 2, out_dim),
        )

    @staticmethod
    def make_layer(in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        out = self.in_conv(x)
        for layer in self.layers:
            out = layer(out)
        out = self.out_fc(out)
        return out


class Decoder(nn.Module):
    def __init__(self, out_ch, base_ch, latent_dim, image_res=64, out_act=torch.tanh):
        super(Decoder, self).__init__()
        self.out_ch = out_ch
        self.latent_dim = latent_dim
        self.base_ch = base_ch
        num_layers = max(math.ceil(math.log(image_res, 2)) - 3, 3)
        spatial_size = image_res // 2 ** (num_layers + 1)
        self.in_fc = nn.Sequential(
            nn.Linear(latent_dim, base_ch * 2 ** num_layers * spatial_size ** 2, bias=False),
            Reshape((-1, base_ch * 2 ** num_layers, spatial_size, spatial_size)),
            nn.BatchNorm2d(base_ch * 2 ** num_layers),
            nn.ReLU(inplace=True)
        )
        self.layers = nn.ModuleList()
        for i in range(num_layers, 0, -1):
            self.layers.append(self.make_layer(base_ch * 2 ** i, base_ch * 2 ** (i - 1)))
        self.out_conv = nn.ConvTranspose2d(base_ch, out_ch, 4, 2, 1)
        self.out_act = out_act

    @staticmethod
    def make_layer(in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.in_fc(x)
        for layer in self.layers:
            out = layer(out)
        out = self.out_conv(out)
        return self.out_act(out)


class AntiArtifactDecoder(Decoder):
    """
    Using Resize + Conv to replace Transposed Conv
    which mitigates the checkerboard artifacts produced by the latter
    as suggested by Odena, et al., "Deconvolution and Checkerboard Artifacts", Distill, 2016.
    http://doi.org/10.23915/distill.00003
    """
    @staticmethod
    def make_layer(in_ch, out_ch):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )