import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PartialBatchNorm2d(nn.Module):
    def __init__(self, splits, flip=False):
        super(PartialBatchNorm2d, self).__init__()
        self.splits = splits
        self.bn_list = nn.ModuleList([
            nn.BatchNorm2d(spl)
            for spl in splits[int(flip)::2]
        ])
        self.flip = flip

    def forward(self, x):
        x_splits = list(torch.split(x, self.splits, dim=1))
        for i, bn in enumerate(self.bn_list):
            ii = int(self.flip) + 2 * (i-1)
            x_splits[ii] = bn(x_splits[ii])
        return torch.cat(x_splits, dim=1)


class BatchDropout(nn.Module):
    def __init__(self, drop_rate, return_none=False):
        super(BatchDropout, self).__init__()
        self.drop_rate = drop_rate
        self.return_none = return_none

    def forward(self, x):
        if self.training:
            u = random.random()
            if u < self.drop_rate:
                if self.return_none:
                    x = None
                else:
                    x = x.zero_()
        return x


class BinLoss(nn.Module):
    def __init__(self, total_trials, reduction="mean"):
        super(BinLoss, self).__init__()
        self.total_trials = total_trials
        self.log_factorial = lambda n: torch.lgamma(n + 1)
        self.lfn = self.log_factorial(torch.tensor(total_trials))
        self.bce_loss = nn.BCELoss(reduction="none")
        self.reduction = reduction

    def forward(self, probs, obs):
        mles = obs / self.total_trials
        # calculate log factorials
        lfobs = self.log_factorial(obs)
        lfobsc = self.log_factorial(self.total_trials-obs)
        bce_loss = self.bce_loss(probs, mles)
        bin_loss = lfobs + lfobsc - self.lfn + self.total_trials * bce_loss
        if self.reduction == "mean":
            bin_loss = bin_loss.mean()
        if self.reduction == "sum":
            bin_loss = bin_loss.sum()
        return bin_loss


class BinWithMLELoss(nn.Module):
    def __init__(self, total_trials, reduction="mean"):
        super(BinLoss, self).__init__()
        self.total_trials = total_trials
        self.log_factorial = lambda n: torch.lgamma(n + 1)
        self.lfn = self.log_factorial(torch.tensor(total_trials))
        self.bce_loss = nn.BCELoss(reduction="none")
        self.reduction = reduction

    def forward(self, probs, mles):
        obs = (mles * self.total_trials).int()
        # calculate log factorials
        lfobs = self.log_factorial(obs)
        lfobsc = self.log_factorial(self.total_trials-obs)
        bce_loss = self.bce_loss(probs, mles)
        bin_loss = lfobs + lfobsc - self.lfn + self.total_trials * bce_loss
        if self.reduction == "mean":
            bin_loss = bin_loss.mean()
        if self.reduction == "sum":
            bin_loss = bin_loss.sum()
        return bin_loss


class Reshape(nn.Module):
    # Reshape module
    def __init__(self, dims):
        super(Reshape, self).__init__()
        self.dims = dims

    def forward(self, x):
        return x.reshape(*self.dims)

    def extra_repr(self):
        s = "{dims}"
        return s.format(**self.__dict__)


class ResidualBlock(nn.Module):
    def __init__(self, in_chan, out_chan, stride=1, bias=False):
        super(ResidualBlock, self).__init__()
        self.in_bn = nn.BatchNorm2d(in_chan)
        self.in_conv = nn.Conv2d(in_chan, out_chan, 3, stride, 1, bias=False)
        self.out_bn = nn.BatchNorm2d(out_chan)
        self.out_conv = nn.Conv2d(out_chan, out_chan, 3, 1, 1, bias=bias)
        if in_chan != out_chan or stride != 1:
            self.skip = nn.Conv2d(in_chan, out_chan, 1, stride, 0, bias=bias)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        skip = self.skip(x)
        x = self.in_conv(F.relu(self.in_bn(x), inplace=True))
        x = self.out_conv(F.relu(self.out_bn(x), inplace=True))
        return x + skip
