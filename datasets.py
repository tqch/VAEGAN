import os
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, SubsetRandomSampler, Sampler

sampler = SubsetRandomSampler


def crop_celeba(img):
    return transforms.functional.crop(img, top=40, left=15, height=148, width=148)


DATA_INFO = {
    "mnist": {
        "data": datasets.MNIST,
        "resolution": (28, 28),
        "channels": 1,
        "transform": transforms.ToTensor(),
        "train_size": 60000,
        "test_size": 10000
    },
    "cifar10": {
        "data": datasets.CIFAR10,
        "resolution": (32, 32),
        "channels": 3,
        "transform": transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]),
        "train_size": 50000,
        "test_size": 10000
    },
    "celeba": {
        "data": datasets.CelebA,
        "resolution": (64, 64),
        "channels": 3,
        "transform": transforms.Compose([
            crop_celeba,
            transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]),
        "train": 162770,
        "test": 19962,
        "validation": 19867
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


def get_dataloader(
        dataset,
        batch_size,
        split,
        val_size=0.1,
        random_seed=None,
        root=ROOT,
        pin_memory=False,
        num_workers=os.cpu_count()
):
    transform = DATA_INFO[dataset]["transform"]
    dataloader_configs = {
        "batch_size": batch_size,
        "pin_memory": pin_memory,
        "num_workers": num_workers
    }
    if dataset == "celeba":
        data = DATA_INFO[dataset]["data"](
                root=root, split=split, download=False, transform=transform)
        dataloader = DataLoader(data, **dataloader_configs)
    else:
        if split == "test":
            data = DATA_INFO[dataset]["data"](
                root=root, train=False, download=False, transform=transform)
            dataloader = DataLoader(data, **dataloader_configs)
        else:
            data = DATA_INFO[dataset]["data"](
                root=root, train=True, download=False, transform=transform)
            train_inds, val_inds = train_val_split(dataset, val_size, random_seed)
            if split == "train":
                sampler = SubsetRandomSampler(train_inds)
            elif split == "valid":
                sampler = SubsetSequentialSampler(val_inds)  # trivial sequential sampler
            # no need of shuffling when using customized sampler
            dataloader = DataLoader(data, sampler=sampler, **dataloader_configs)
    return dataloader
