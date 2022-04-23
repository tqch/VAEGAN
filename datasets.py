import os
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, SubsetRandomSampler, Sampler

sampler = SubsetRandomSampler
DATA_INFO = {
    "mnist": {
        "data": datasets.MNIST,
        "resolution": (28, 28),
        "channels": 1,
        "transform": transforms.ToTensor(),
        "train_size": 60000,
        "test_size": 10000
    }
}

ROOT = os.path.expanduser("~/datasets")


def train_val_split(dataset, val_size, random_seed=None):
    train_size = DATA_INFO[dataset]["train_size"]
    if random_seed is not None:
        np.random.seed(random_seed)
    train_inds = np.arange(train_size)
    np.random.shuffle(train_inds)
    val_size = int(train_size * val_size)
    val_inds, train_inds = train_inds[:val_size], train_inds[val_size:]
    return train_inds, val_inds


class SubsetSequentialSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def get_dataloader(dataset, batch_size, split, val_size=0.1, random_seed=None, root=ROOT):
    if dataset == "mnist":
        transform = DATA_INFO[dataset]["transform"]
        if split == "test":
            data = DATA_INFO[dataset]["data"](
                root=root, train=False, download=False, transform=transform)
            dataloader = DataLoader(data, batch_size=batch_size)
        else:
            data = DATA_INFO[dataset]["data"](
                root=root, train=True, download=False, transform=transform)
            train_inds, val_inds = train_val_split(dataset, val_size, random_seed)
            if split == "train":
                sampler = SubsetRandomSampler(train_inds)
            elif split == "val":
                sampler = SubsetSequentialSampler(val_inds)  # trivial sequential sampler
            # no need of shuffling when using customized sampler
            dataloader = DataLoader(data, batch_size=batch_size, sampler=sampler)
    return dataloader
