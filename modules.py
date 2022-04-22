import random
import torch
import torch.nn as nn


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
    def __init__(self, drop_prob, return_none=False):
        super(BatchDropout, self).__init__()
        self.drop_prob = drop_prob
        self.return_none = return_none

    def forward(self, x):
        if self.training:
            u = random.random()
            if u < self.drop_prob:
                if self.return_none:
                    x = None
                else:
                    x = x.zero_()
        return x
