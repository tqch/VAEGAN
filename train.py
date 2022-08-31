import os
import math
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
from metrics.fid_score import InceptionStatistics, get_precomputed, calc_fd
from metrics.snr import PSNR


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
        self.model.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return {"train_vae_loss": loss.item()}

    def gan_train_step(self, x):
        # update discriminator
        dis_loss = self.model(x)
        self.model.zero_grad(set_to_none=True)
        dis_loss.backward()
        self.optimizer_d.step()
        # update generator step
        gen_loss = self.model(n=x.shape[0] * 2, dis=False)
        self.model.zero_grad(set_to_none=True)
        gen_loss.backward()
        self.optimizer_g.step()

        return {
            "train_dis_loss": dis_loss.item(),
            "train_gen_loss": gen_loss.item()
        }

    def vaegan_train_step(self, x):
        # update vae (encoder + generator + reconstructor)
        vae_loss, gen_loss = self.model(x)
        loss = vae_loss + gen_loss
        loss.backward()
        self.optimizer_vae.step()
        self.model.zero_grad(set_to_none=True)
        # update discriminator
        dis_loss = self.model(x=x, component="netD")
        dis_loss.backward()
        self.optimizer_d.step()
        self.model.zero_grad(set_to_none=True)

        return {
            "train_vae_loss": vae_loss.item(),
            "train_gen_loss": gen_loss.item(),
            "train_dis_loss": dis_loss.item()
        }

    def step(self, x):
        count = x.shape[0]
        if self.model_name == "vae":
            self.running_stats.update(count, **self.vae_train_step(x))
        elif self.model_name == "gan":
            self.running_stats.update(count, **self.gan_train_step(x))
        elif self.model_name == "vaegan":
            self.running_stats.update(count, **self.vaegan_train_step(x))
        else:
            raise NotImplementedError

    def current_stats(self):
        return self.running_stats.extract()

    def reset_stats(self):
        self.running_stats.reset()

    def resume_from_chkpt(self, chkpt_path, device=torch.device("cpu")):
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
    parser.add_argument("--model", choices={"gan", "vae", "vaegan"}, default="vae")
    parser.add_argument("--backbone", choices={"dcgan", "resnet"}, default="dcgan")
    parser.add_argument("--out-act", choices={"tanh", "sigmoid"}, default="tanh")
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--lr", default=5e-4, type=float)
    parser.add_argument("--d-factor", default=0.5, type=float)
    parser.add_argument("--g-factor", default=0.5, type=float)
    parser.add_argument("--beta1", default=0.5, type=float)
    parser.add_argument("--beta2", default=0.999, type=float)
    parser.add_argument("--dataset", choices=["mnist", "cifar10", "celeba"], default="cifar10")
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--num-workers", default=os.cpu_count(), type=int)
    parser.add_argument("--root", default="~/datasets", type=str)
    parser.add_argument("--task", choices=["reconstruction", "generation", "deblur"], default="generation")
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--eval-device", default=0, type=int)
    parser.add_argument("--base-ch", default=64, type=int)
    parser.add_argument("--latent-dim", default=128, type=int)
    parser.add_argument("--reconst_ch", default=64, type=int)
    parser.add_argument("--instance-noise", action="store_true")
    parser.add_argument("--fig-dir", default="./figs", type=str)
    parser.add_argument("--chkpt-dir", default="./chkpts", type=str)
    parser.add_argument("--log-dir", default="./logs", type=str)
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--calc-metrics", action="store_true")
    parser.add_argument("--chkpt-intv", default=10, type=int)
    parser.add_argument("--comment", default="", type=str)
    parser.add_argument("--anti-artifact", action="store_true")

    args = parser.parse_args()
    os.environ["BACKBONE"] = backbone = args.backbone
    os.environ["ANTI_ARTIFACT"] = anti_artifact = "true" if args.anti_artifact else ""

    from models import MODEL_LIST

    root = os.path.expanduser(args.root)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    dataset = args.dataset

    in_ch = DATA_INFO[dataset]["channels"]
    image_res = DATA_INFO[dataset]["resolution"][0]

    seed = args.seed
    batch_size = args.batch_size
    num_workers = args.num_workers

    task = args.task
    train_split = "all" if dataset in ["celeba", ] else "train"
    val_size = 0 if task == "generation" else 0.1
    trainloader = get_dataloader(
        dataset, batch_size=batch_size, split=train_split, val_size=val_size,
        random_seed=seed, root=root, pin_memory=True, num_workers=num_workers)
    valloader = get_dataloader(
        dataset, batch_size=batch_size, split="valid", val_size=0.1,
        random_seed=seed, root=root, pin_memory=True, num_workers=num_workers)

    model_name = args.model
    base_ch = args.base_ch
    latent_dim = args.latent_dim
    reconst_ch = args.reconst_ch
    out_act = args.out_act

    model_configs = dict(
        in_ch=in_ch,
        base_ch=base_ch,
        latent_dim=latent_dim,
        image_res=image_res,
        out_act=out_act
    )
    if out_act == "sigmoid":
        # if output activation is sigmoid
        # remove the last transformation layer, i.e. `transforms.Normalize`
        trainloader.dataset.transform.transforms.pop()
        grid_kwargs = dict(normalize=False)
    else:
        grid_kwargs = dict(normalize=True, value_range=(-1, 1))

    instance_noise = args.instance_noise
    noise_scale = 0.1 if instance_noise else 0.
    if "gan" in model_name:
        model_configs.update(noise_scale=noise_scale)
    if model_name == "vaegan":
        model_configs.update(reconst_ch=reconst_ch)

    model = MODEL_LIST[model_name](**model_configs)
    model.to(device)

    calc_metrics = args.calc_metrics
    eval_device = torch.device(f"cuda:{args.eval_device}" if torch.cuda.is_available() else "cpu")
    if calc_metrics:
        istats = InceptionStatistics(device=eval_device)
        psnr = PSNR()
        target_mean, target_var = get_precomputed(dataset)

    lr = args.lr

    # ======= TTUR =======
    # ======= DCGAN ======
    # |Dataset|lr (D)|lr (G)|
    # |:-----:|:----:|:----:|
    # |CelebA|1e-5|5e-4|
    # |CIFAR-10|1e-4|5e-4|

    d_factor = args.d_factor
    g_factor = args.g_factor
    beta1 = args.beta1
    beta2 = args.beta2

    if model_name == "vae":
        optimizer = Adam(model.parameters(), lr=lr, betas=(beta1, beta2))
        trainer = Trainer(model=model, optimizer=optimizer)
    elif model_name == "gan":
        optimizer_d = Adam(model.netD.parameters(), lr=d_factor * lr, betas=(beta1, beta2))
        optimizer_g = Adam(model.netG.parameters(), lr=g_factor * lr, betas=(beta1, beta2))
        trainer = Trainer(model=model, optimizer_d=optimizer_d, optimizer_g=optimizer_g)
    elif model_name == "vaegan":
        optimizer_vae = Adam([
            {"params": model.encoder.parameters()},
            {"params": model.decoder.parameters(), "lr": g_factor * lr}
        ], lr=lr, betas=(beta1, beta2))
        optimizer_d = Adam(model.netD.parameters(), lr=d_factor * lr, betas=(beta1, beta2))
        trainer = Trainer(model=model, optimizer_vae=optimizer_vae, optimizer_d=optimizer_d)
    else:
        NotImplementedError("Unsupported model type!")

    nimgs = 64 if task == "generation" else 32
    x_val, _ = next(iter(valloader))
    x_val = x_val[:nimgs]
    transform_blur = GaussianBlur(kernel_size=7, sigma=3.)

    fixed_noise = None
    if task == "generation":
        image_tag = "random_samples"
        fixed_noise = torch.randn((nimgs, latent_dim), device=device)
    elif task == "deblur":
        x_corrupted = transform_blur(x_val)
        image_tag = "deblurred_examples"
    elif task == "reconstruction":
        image_tag = "reconstructed_examples"
    else:
        raise NotImplementedError

    meta_dict = {
        "dataset": dataset,
        "model": model_name,
        "backbone": backbone,
        "anti_artifact": anti_artifact,
        "task": task,
        "lr": lr,
        "g_factor": g_factor,
        "batch_size": batch_size,
        "instance_noise": instance_noise,
        "model_configs": model_configs
    }
    metadata = dict2str(meta_dict)

    timestamp = datetime.now().strftime("%Y-%m-%dT%H%M%S%f")
    fig_dir = os.path.join(args.fig_dir, dataset, f"{model_name}_{timestamp}")
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    log_dir = args.log_dir
    log_path = os.path.join(
        log_dir, dataset, f"{model_name}_{timestamp}")
    writer = SummaryWriter(log_path)  # tensorboard writer
    with open(os.path.join(log_path, "metadata"), "w") as f:
        f.write(metadata)

    comment = args.comment
    chkpt_dir = os.path.join("./chkpts", f"{dataset}_{model_name}{comment and '_' + comment}")
    if not os.path.exists(chkpt_dir):
        os.makedirs(chkpt_dir)
    chkpt_path = os.path.join(chkpt_dir, f"checkpoint.pt")

    from_epoch = 0
    best_fid = np.inf
    if args.resume and os.path.exists(chkpt_path):
        from_epoch, best_fid = trainer.resume_from_chkpt(chkpt_path, device=device)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = True

    n_epochs = args.epochs
    for e in range(from_epoch, n_epochs):
        model.train()
        with tqdm(trainloader, desc=f"{e + 1}/{n_epochs} epochs") as t:
            trainer.reset_stats()
            for i, (x_real, _) in enumerate(t):

                trainer.step(x_real.to(device))
                t.set_postfix(trainer.current_stats())

                if i == len(trainloader) - 1:
                    model.eval()
                    extra_stats = dict()
                    if args.calc_metrics:
                        eval_batch_size = 128
                        max_eval_count = 10000
                        istats.reset()
                        psnr.reset()
                        for _ in range(0, max_eval_count + eval_batch_size, eval_batch_size):
                            with torch.no_grad():
                                x = model.sample_x(eval_batch_size)
                            istats(x.to(eval_device))

                        gen_mean, gen_var = istats.get_statistics()
                        train_fid = calc_fd(gen_mean, target_mean, gen_var, target_var)
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

                        if train_fid < best_fid:
                            best_fid = train_fid

                    training_stats = trainer.current_stats()
                    training_stats.update(dict(extra_stats))
                    t.set_postfix(training_stats)

                    if "vae" in model_name:
                        writer.add_scalar("elbo", -training_stats["train_vae_loss"], e + 1)
                    if "gan" in model_name:
                        if training_stats["train_dis_loss"] < (1 / 3) * math.log(2):  # ~80% clf accuracy
                            model.noise_scale *= 2
                        elif training_stats["train_dis_loss"] > math.log(2):  # ~50% clf accuracy
                            model.noise_scale *= 0.9
                        writer.add_scalar("dis_loss", training_stats["train_dis_loss"], e + 1)
                        writer.add_scalar("gen_loss", training_stats["train_gen_loss"], e + 1)

                    if instance_noise:
                        writer.add_scalar("noise_scale", model.noise_scale, e + 1)

                    with torch.no_grad():
                        if task == "generation":
                            xx_ = model.sample_x(nimgs, noise=fixed_noise).cpu()
                        else:
                            if task == "deblur":
                                # recover from corrupted validation data
                                x_ = model.reconst(x_corrupted.to(device)).cpu()
                                xx_ = torch.cat([x_corrupted, x_], dim=0)
                            elif task == "reconstruction":
                                x_ = model.reconst(x_val.to(device)).cpu()
                                xx_ = torch.cat([x_val, x_], dim=0)
                        # convert to 2d numpy array in the order (H, W, C)
                    npimg = make_grid(xx_, nrow=8, **grid_kwargs).permute(1, 2, 0).numpy()
                    img_format = "HWC"
                    if npimg.shape[2] == 1:
                        img_format = "HW"
                        npimg = npimg.squeeze(axis=2)
                    writer.add_image(image_tag, npimg, e + 1, dataformats=img_format)
                    plt.imsave(os.path.join(
                        fig_dir,
                        f"{e + 1}.jpg"
                    ), npimg)

                    if (e + 1) % args.chkpt_intv == 0:
                        trainer.checkpoint(chkpt_path, fid=best_fid, epoch=e + 1)

    writer.flush()  # make sure that all pending events have been written to disk
    writer.close()

    # save the decoder/generator for future evaluation
    dst = os.path.join(args.chkpt_dir, f"{dataset}_{model_name}_{timestamp}.pt")
    state_dict = model.state_dict()
    for k in state_dict.keys():
        if not k.split(".")[0] in ["decoder", "netG"]:
            state_dict[k] = None
    torch.save(state_dict, dst)
