import math
import torch
import torch.nn as nn
from modules import Reshape
from functools import partial


class AttentionBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(AttentionBlock, self).__init__()
        hid_ch = in_ch // 8
        self.in_conv = nn.Conv2d(in_ch, 3 * hid_ch, 1, 1, 0)
        self.out_conv = nn.Conv2d(hid_ch, out_ch, 1, 1, 0)
        self.hid_ch = hid_ch
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1, 1, 0)

    def forward(self, x):
        q, k, v = self.in_conv(x).chunk(3, dim=1)
        weight = torch.einsum("bchw,bcHW->bhwHW", q, k).div(math.sqrt(self.hid_ch))
        out = torch.einsum("bhwc,bCc->bChw",
            torch.softmax(weight.flatten(start_dim=3), dim=3), v.flatten(start_dim=2))
        return self.out_conv(out) + self.skip(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, resample="none", normalize=True, use_bias=False):
        super(ResidualBlock, self).__init__()
        assert resample in {"upsample", "downsample", "none"}

        stride = 1
        if resample == "upsample":
            self.sampler = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        elif resample == "downsample":
            stride = 2
            self.sampler = nn.AvgPool2d(2)
        self.resample = resample

        self.in_bn = nn.BatchNorm2d(in_ch) if normalize else nn.Identity()
        self.in_act = nn.ReLU(inplace=normalize)
        self.in_conv = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=not normalize)
        self.out_bn = nn.BatchNorm2d(out_ch) if normalize else nn.Identity()
        self.out_act = nn.ReLU(inplace=normalize)
        self.out_conv = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=use_bias)

        self.skip = nn.Conv2d(
            in_ch, out_ch, 1, 1, 0, bias=use_bias) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        if self.resample == "upsample":
            x = self.sampler(x)
        skip = self.skip(self.sampler(x) if self.resample == "downsample" else x)
        out = self.in_conv(self.in_act(self.in_bn(x)))
        out = self.out_conv(self.out_act(self.out_bn(out)))
        return out + skip


class Encoder(nn.Module):
    block = partial(ResidualBlock, resample="downsample", normalize=True)

    def __init__(self, in_ch, base_ch, out_dim=1, apply_attn=True, image_res=64):
        super(Encoder, self).__init__()
        self.in_ch = in_ch
        self.base_ch = base_ch
        self.in_conv = self.block(in_ch, base_ch)
        self.in_conv.in_bn = self.in_conv.in_act = nn.Identity()
        self.layers = nn.ModuleList()
        if apply_attn and math.ceil(math.log(image_res, 2)) < 6:  # e.g. 32 x 32
            self.layers.append(AttentionBlock(base_ch, base_ch))
        num_layers = max(math.ceil(math.log(image_res, 2)) - 3, 3)
        spatial_size = image_res // 2 ** (num_layers + 1)
        for i in range(num_layers):
            self.layers.append(self.block(
                base_ch * 2 ** i, base_ch * 2 ** (i + 1), use_bias=(i == num_layers)))
            if apply_attn and i == math.ceil(math.log(image_res, 2)) - 6:
                self.layers.append(AttentionBlock(base_ch * 2 ** (i + 1), base_ch * 2 ** (i + 1)))
        self.out_fc = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.ReLU(),
            nn.Linear(base_ch * 2 ** (i + 1) * spatial_size ** 2, out_dim),
        )

    def forward(self, x):
        out = self.in_conv(x)
        for layer in self.layers:
            out = layer(out)
        out = self.out_fc(out)
        return out


class Decoder(nn.Module):
    block = partial(ResidualBlock, resample="upsample")

    def __init__(self, out_ch, base_ch, latent_dim, apply_attn=True, image_res=64, out_act=torch.tanh):
        super(Decoder, self).__init__()
        self.out_ch = out_ch
        self.latent_dim = latent_dim
        self.base_ch = base_ch
        num_layers = max(math.ceil(math.log(image_res, 2)) - 2, 4)
        spatial_size = image_res // 2 ** num_layers
        self.in_fc = nn.Sequential(
            nn.Linear(latent_dim, base_ch * 2 ** num_layers * spatial_size ** 2, bias=False),
            Reshape((-1, base_ch * 2 ** num_layers, spatial_size, spatial_size)))
        self.layers = nn.ModuleList()
        for i in range(num_layers, 0, -1):
            self.layers.append(self.block(base_ch * 2 ** i, base_ch * 2 ** (i - 1)))
            if apply_attn and i == num_layers + math.floor(math.log(spatial_size, 2)) - 3:
                self.layers.append(AttentionBlock(base_ch * 2 ** (i - 1), base_ch * 2 ** (i - 1)))
        self.out_conv = nn.Sequential(
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, out_ch, 3, 1, 1))
        self.out_act = out_act

    def forward(self, x):
        out = self.in_fc(x)
        for layer in self.layers:
            out = layer(out)
        out = self.out_conv(out)
        return self.out_act(out)
