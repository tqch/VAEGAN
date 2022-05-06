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
from utils import dict2str, RunningStatistics

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


class Trainer:
    def __init__(self, model, **optimizers):
        self.model = model
        self.model_name = model.__class__.__name__.lower()
        self.optimizer_names = []
        for k, v in optimizers.items():
            setattr(self, k, v)
            self.optimizer_names.append(k)
        self.running_stats = RunningStatistics()

    def named_state_dicts(self):
        for k in ["model", ] + self.optimizer_names:
            yield k, getattr(self, k).state_dict()

    def vae_train_step(self, x):
        loss = self.model(x)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"train_vae_loss": loss.item()}

    def gan_train_step(self, x):
        # update discriminator
        dis_loss = self.model(x)
        self.model.zero_grad()
        dis_loss.backward()
        self.optimizer_d.step()
        # update generator step
        gen_loss = self.model(n=x.shape[0] * 2, dis=False)
        self.model.zero_grad()
        gen_loss.backward()
        self.optimizer_g.step()

        return {
            "train_dis_loss": dis_loss.item(),
            "train_gen_loss": gen_loss.item()
        }

    def vaegan_train_step(self, x):
        # update vae (encoder + crude + refine)
        vae_loss, gen_loss, l2_reg = self.model(x)
        loss = vae_loss + gen_loss + l2_reg
        self.model.zero_grad()
        loss.backward()
        self.optimizer_vae.step()
        # update discriminator
        dis_loss = self.model(x=x, component="discriminator")
        self.model.zero_grad()
        dis_loss.backward()
        self.optimizer_d.step()

        return {
            "train_vae_loss": vae_loss.item(),
            "train_gen_loss": gen_loss.item(),
            "train_dis_loss": dis_loss.item()
        }

    def step(self, x):
        count = x.shape[0]
        if self.model_name == "vae":
            self.running_stats.update(count, **self.vae_train_step(x))
        elif self.model_name == "dcgan":
            self.running_stats.update(count, **self.gan_train_step(x))
        elif self.model_name == "vaegan":
            self.running_stats.update(count, **self.vaegan_train_step(x))
        else:
            raise NotImplementedError

    def current_stats(self):
        return self.running_stats.extract()

    def reset_stats(self):
        self.running_stats.reset()

    def restart_from_chkpt(self, chkpt_path, device=torch.device("cpu")):
        chkpt = torch.load(chkpt_path, map_location=device)
        self.model.load_state_dict(chkpt["model"])
        for k in self.optimizer_names:
            getattr(self, k).load_state_dict(chkpt[k])
        return chkpt["epoch"], chkpt["fid"]

    def checkpoint(self, chkpt_path, **extra_info):
        chkpt = []
        for k, v in self.named_state_dicts():
            chkpt.append((k, v))
        for k, v in extra_info.items():
            chkpt.append((k, v))
        torch.save(dict(chkpt), chkpt_path)


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
    parser.add_argument("--restart", action="store_true")

    args = parser.parse_args()

    root = os.path.expanduser(args.root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = args.dataset

    in_chan = DATA_INFO[dataset]["channels"]
    image_res = DATA_INFO[dataset]["resolution"][0]

    seed = args.seed
    batch_size = args.batch_size

    trainloader = get_dataloader(
        dataset, batch_size=batch_size, split="train", val_size=0.1, random_seed=seed, root=root, pin_memory=True)
    valloader = get_dataloader(
        dataset, batch_size=batch_size, split="valid", val_size=0.1, random_seed=seed, root=root, pin_memory=True)

    config_path = os.path.join(args.config_dir, args.model + ".json")
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

    if model_name == "vae":
        optimizer = Adam(model.parameters(), lr=lr, betas=(beta1, beta2))
        trainer = Trainer(model=model, optimizer=optimizer)
    elif model_name == "dcgan":
        optimizer_d = Adam(model.discriminator.parameters(), lr=lr, betas=(beta1, beta2))
        optimizer_g = Adam(model.generator.parameters(), lr=lr, betas=(beta1, beta2))
        trainer = Trainer(model=model, optimizer_d=optimizer_d, optimizer_g=optimizer_g)
    elif model_name == "vaegan":
        optimizer_vae = Adam([
            {"params": model.encoder.parameters()},
            {"params": model.decoder.parameters()}
        ], lr=lr, betas=(beta1, beta2))
        optimizer_d = Adam(model.discriminator.parameters(), lr=lr, betas=(beta1, beta2))
        trainer = Trainer(model=model, optimizer_vae=optimizer_vae, optimizer_d=optimizer_d)


    n_display = 16
    x_val, _ = next(iter(valloader))
    x_val = x_val[:n_display]

    transform_blur = GaussianBlur(kernel_size=7, sigma=3.)

    task = args.task
    if task == "generation":
        image_tag = "random_samples"
    elif task == "deblur":
        x_corrupted = transform_blur(x_val)
        image_tag = "deblurred_examples"
    elif task == "reconstruction":
        image_tag = "reconstructed_examples"
    else:
        raise NotImplementedError

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
    chkpt_path = os.path.join(
        chkpt_dir,
        f"{dataset}_{model_name}_{hps_info}.pt"
    )

    from_epoch = 0
    best_fid = np.inf
    if args.restart and os.path.exists(chkpt_path):
        from_epoch, best_fid = trainer.restart_from_chkpt(chkpt_path, device=device)

    writer = SummaryWriter(log_path)

    n_epochs = args.epochs

    for e in range(from_epoch, n_epochs):
        with tqdm(trainloader, desc=f"{e + 1}/{n_epochs} epochs") as t:
            trainer.reset_stats()
            for i, (x, _) in enumerate(t):

                model.train()
                trainer.step(x.to(device))
                t.set_postfix(trainer.current_stats())

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
                    train_fid = fid(gen_mean, target_mean, gen_var, target_var)
                    writer.add_scalar("fid", train_fid, e + 1)
                    extra_stats = [("train_fid", train_fid)]

                    if "vae" in model_name:
                        eval_count = 0
                        for x, _ in trainloader:
                            if eval_count >= max_eval_count:
                                break
                            with torch.no_grad():
                                x_ = model.reconst(x.to(device)).detach().cpu()
                            psnr((x_ - x).numpy())
                            eval_count += x.shape[0]
                        train_psnr = psnr.get_metrics()
                        writer.add_scalar("psnr", train_psnr, e + 1)
                        extra_stats.append(("train_psnr", train_psnr))

                    training_stats = trainer.current_stats()
                    training_stats.update(dict(extra_stats))
                    t.set_postfix(training_stats)

                    if "vae" in model_name:
                        writer.add_scalar("elbo", -training_stats["train_vae_loss"], e + 1)
                    if "gan" in model_name:
                        writer.add_scalar("dis_loss", training_stats["train_dis_loss"], e + 1)
                        writer.add_scalar("gen_loss", training_stats["train_gen_loss"], e + 1)

                    if train_fid < best_fid:
                        best_fid = train_fid
                        trainer.checkpoint(chkpt_path, fid=best_fid, epoch=e+1)

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
