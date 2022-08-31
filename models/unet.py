import os
import math
import torch
import torch.nn as nn
from . import resnet
from . import dcgan
from functools import partial

backbone = {"resnet": resnet, "dcgan": dcgan}[os.environ.get("BACKBONE", "dcgan")]

if backbone.__name__ == "models.dcgan":
    def _block(in_ch, out_ch, resample):
        if resample == "upsample":
            if os.environ.get("ANTI_ARTIFACT", ""):
                return backbone.AntiArtifactDecoder.make_layer(in_ch, out_ch)
            else:
                return backbone.Decoder.make_layer(in_ch, out_ch)
        elif resample == "downsample":
            return backbone.Encoder.make_layer(in_ch, out_ch)
        else:
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True))
else:
    _block = backbone.ResidualBlock


class Encoder(nn.Module):
    block = partial(_block, resample="downsample")

    def __init__(self, in_ch, base_ch, image_res=64):
        super(Encoder, self).__init__()
        self.in_ch = in_ch
        self.base_ch = base_ch
        self.in_conv = _block(in_ch, base_ch, resample="none")
        try:
            self.in_conv.in_bn = self.in_conv.in_act = nn.Identity()
        except AttributeError:
            pass
        self.layers = nn.ModuleList()
        num_layers = math.ceil(math.log(image_res, 2) - 3)
        for i in range(num_layers):
            layer = [self.block(base_ch * 2 ** i, base_ch * 2 ** (i + 1)), ]
            if i == num_layers - 1:
                layer.append(resnet.AttentionBlock(base_ch * 2 ** (i + 1), base_ch * 2 ** (i + 1)))
            self.layers.append(nn.Sequential(*layer))

    def forward(self, x):
        outs = [self.in_conv(x)]
        for layer in self.layers:
            outs.append(layer(outs[-1]))
        return outs


class Decoder(nn.Module):
    block = partial(_block, resample="upsample")

    def __init__(self, out_ch, base_ch, image_res=64, out_act=torch.tanh):
        super(Decoder, self).__init__()
        self.out_ch = out_ch
        self.base_ch = base_ch
        num_layers = math.ceil(math.log(image_res, 2) - 3)
        self.layers = nn.ModuleList()
        for i in range(num_layers, 0, -1):
            self.layers.append(self.block(
                base_ch * 2 ** min(num_layers, i + 1), base_ch * 2 ** (i - 1)))
        if isinstance(self.block, resnet.ResidualBlock):
            self.out_conv = nn.Sequential(
                nn.BatchNorm2d(base_ch * 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(2 * base_ch, out_ch, 3, 1, 1))
        else:
            self.out_conv = nn.Conv2d(2 * base_ch, out_ch, 3, 1, 1)
        self.out_act = out_act

    def forward(self, outs):
        out = None
        for layer in self.layers:
            out = layer(outs.pop() if out is None else torch.cat([out, outs.pop()], dim=1))
        out = self.out_conv(torch.cat([out, outs.pop()], dim=1))
        return self.out_act(out)


class UNet(nn.Module):
    def __init__(self, in_ch, base_ch, image_res, out_act):
        super(UNet, self).__init__()
        self.down = Encoder(in_ch, base_ch, image_res)
        self.up = Decoder(in_ch, base_ch, image_res, out_act)

    def forward(self, x):
        outs = self.down(x)
        return self.up(outs)


def test():
    model = UNet(3, 64, 64)
    print(model)
    model(torch.randn(64, 3, 64, 64))
