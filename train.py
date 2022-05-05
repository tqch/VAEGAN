import torch

import os, json
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torchvision.utils import make_grid
from torchvision.transforms import GaussianBlur

import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["figure.dpi"] = 144

from datasets import get_dataloader, DATA_INFO

from datetime import datetime
from utils import dict2str

from metrics.fid_score import InceptionStatistics, get_precomputed, fid
from metrics.snr import PSNR

from models.vae import VAE
from models.dcgan import DCGAN
from models.vaegan import VAEGAN

MODEL_LIST = {
    "vae": VAE,
    "dcgan": DCGAN,
    "vaegan": VAEGAN
}

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--model", choices=["vae", "dcgan", "vaegan"], default="vae")
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--lr", default=0.0002, type=float)
    parser.add_argument("--beta1", default=0.5, type=float)
    parser.add_argument("--beta2", default=0.999, type=float)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--dataset", choices=["mnist", "cifar10", "celeba"], default="cifar10")
    parser.add_argument("--root", default="~/datasets", type=str)
    parser.add_argument("--task", choices=["reconstruction", "generation", "deblur"], default="generation")
    parser.add_argument("--gpus", default=0, type=int)
    parser.add_argument("--latent-dim", default=128, type=int)
    parser.add_argument("--fig-dir", default="./figs", type=str)
    parser.add_argument("--config-dir", default="./default_configs", type=str)
    parser.add_argument("--chkpt-dir", default="./chkpts", type=str)
    parser.add_argument("--log-dir", default="./logs", type=str)
    parser.add_argument("--seed", default=1234, type=int)

    args = parser.parse_args()

    root = os.path.expanduser(args.root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = args.dataset

    in_chan = DATA_INFO[dataset]["channels"]
    image_res = DATA_INFO[dataset]["resolution"][0]

    seed = args.seed
    batch_size = args.batch_size
    trainloader = get_dataloader(
        dataset, batch_size=batch_size, split="train", val_size=0.1, random_seed=seed, root=root)
    valloader = get_dataloader(
        dataset, batch_size=batch_size, split="val", val_size=0.1, random_seed=seed, root=root)

    config_path = os.path.join(args.config_dir, args.model+".json")
    with open(config_path, "r") as f:
        configs = json.load(f)

    latent_dim = args.latent_dim

    model = MODEL_LIST[args.model](
        in_chan=in_chan,
        latent_dim=latent_dim,
        image_res=image_res,
        configs=configs
    )
    model.to(device)

    model_name = model.__class__.__name__.lower()

    eval_device = torch.device("cuda")
    istats = InceptionStatistics(device=eval_device)
    psnr = PSNR()
    target_mean, target_var = get_precomputed(dataset)

    lr = args.lr
    beta1 = args.beta1
    beta2 = args.beta2

    optimizer = Adam(model.parameters(), lr=lr, betas=(beta1, beta2))

    n_display = 16
    x_val, _ = next(iter(valloader))
    x_val = x_val[:n_display]

    transform_blur = GaussianBlur(kernel_size=7, sigma=3.)

    task = args.task
    image_tag = "random_samples"
    if task == "deblur":
        x_corrupted = transform_blur(x_val)
        image_tag = "deblurred_examples"
    elif task == "reconstruction":
        image_tag = "reconstructed_examples"

    hps = {
        "task": task,
        "lr": lr,
        "batch_size": batch_size,
        "latent_dim": latent_dim,
        "configs": configs
    }
    hps_info = dict2str(hps)

    fig_dir = os.path.join(args.fig_dir, f"{dataset}_{model_name}_{hps_info}")
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    timestamp = datetime.now().strftime("%Y-%m-%dT%H%M%S%f")
    log_dir = args.log_dir
    log_path = os.path.join(
        log_dir, f"{model_name}_{timestamp}" + "_" + hps_info)
    chkpt_dir = "./chkpts"
    if not os.path.exists(chkpt_dir):
        os.makedirs(chkpt_dir)
    writer = SummaryWriter(log_path)

    n_epochs = args.epochs
    best_fid = np.inf
    for e in range(n_epochs):
        with tqdm(trainloader, desc=f"{e + 1}/{n_epochs} epochs") as t:
            total_train_count = 0
            total_train_loss = 0
            for i, (x, _) in enumerate(t):
                model.train()
                loss = model(x.to(device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()
                total_train_count += x.shape[0]

                t.set_postfix({
                    "train_vae_loss": total_train_loss / total_train_count})

                if i == len(trainloader) - 1:
                    model.eval()
                    eval_batch_size = 64
                    max_eval_count = 10000
                    istats.reset()
                    psnr.reset()
                    for _ in range(0, max_eval_count + eval_batch_size, eval_batch_size):
                        with torch.no_grad():
                            x = model.sample_x(eval_batch_size)
                        istats(x.to(eval_device))

                    gen_mean, gen_var = istats.get_statistics()
                    train_vae_loss = total_train_loss / total_train_count
                    train_fid = fid(gen_mean, target_mean, gen_var, target_var)

                    eval_count = 0
                    for x, _ in trainloader:
                        if eval_count >= max_eval_count:
                            break
                        with torch.no_grad():
                            x_ = model.reconst(x.to(device)).detach().cpu()
                        psnr((x_ - x).numpy())
                        eval_count += x.shape[0]

                    train_psnr = psnr.get_metrics()
                    t.set_postfix({
                        "train_vae_loss": train_vae_loss,
                        "train_fid": train_fid,
                        "train_psnr": train_psnr
                    })
                    writer.add_scalar("elbo", -train_vae_loss, e + 1)
                    writer.add_scalar("fid", train_fid, e + 1)
                    writer.add_scalar("psnr", train_psnr, e + 1)
                    if train_fid < best_fid:
                        best_fid = train_fid
                        chkpt_path = os.path.join(
                            chkpt_dir,
                            f"{dataset}_{model_name}_{hps_info}.pt"
                        )
                        torch.save(
                            {
                                "model": model.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "epoch": e + 1,
                                "fid": best_fid
                            }, chkpt_path)

                    with torch.no_grad():
                        if task == "generation":
                            xx_ = model.sample_x(n_display).detach().cpu()
                        else:
                            if task == "deblur":
                                # recover from corrupted validation data
                                x_ = model.reconst(x_corrupted.to(device)).detach().cpu()
                                xx_ = torch.cat([x_corrupted, x_], dim=0)
                            elif task == "reconstruction":
                                x_ = model.reconst(x_val.to(device)).detach().cpu()
                                xx_ = torch.cat([x_val, x_], dim=0)
                        # convert to 2d numpy array in the order (H, W, C)
                    npimg = make_grid(xx_, nrow=8).permute(1, 2, 0).numpy()
                    img_format = "HWC"
                    if npimg.shape[2] == 1:
                        img_format = "HW"
                        npimg = npimg.squeeze(axis=2)
                    writer.add_image(image_tag, npimg, e + 1, dataformats=img_format)
                    plt.imsave(os.path.join(
                        fig_dir,
                        f"{e + 1}.jpg"
                    ), npimg)

    writer.flush()  # make sure that all pending events have been written to disk
    writer.close()
