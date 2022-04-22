if __name__ == "__main__":
    import os
    import numpy as np
    import torch
    from torch.optim import Adam
    from torch.utils.tensorboard import SummaryWriter
    from tqdm import tqdm
    from torchvision.transforms import GaussianBlur
    from torchvision.utils import make_grid
    import matplotlib.pyplot as plt

    import matplotlib as mpl
    mpl.rcParams["figure.dpi"] = 144

    from uvae_fd import DeblurUVAE
    from datasets import get_dataloader

    from datetime import datetime
    from utils import dict2str

    root = os.path.expanduser("~/datasets")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = "mnist"
    seed = 1234

    trainloader = get_dataloader(
        dataset, batch_size=512, split="train", val_size=0.1, random_seed=seed, root=root)
    valloader = get_dataloader(
        dataset, batch_size=512, split="val", val_size=0.1, random_seed=seed, root=root)

    latent_dim = 64
    skip_dims = [1, 1, 1]
    transform_blur = GaussianBlur(kernel_size=7, sigma=3.)
    alpha = 10

    hps = {
        "latent_dim": latent_dim,
        "alpha": alpha,
        "skip_dims": skip_dims
    }
    hps_info = dict2str(hps)

    chkpt_dir = "./chkpts"
    if not os.path.exists(chkpt_dir):
        os.makedirs(chkpt_dir)

    model = DeblurUVAE(latent_dim=latent_dim, skip_dims=skip_dims,
                       transform=transform_blur, alpha=alpha)
    model.to(device)

    model_name = model.__class__.__name__.lower()

    optimizer = Adam(model.parameters(), lr=0.001)

    n_epochs = 30

    n_display = 16
    x_val, _ = next(iter(valloader))
    # transform_test = GaussianBlur(kernel_size=7, sigma=3.)
    x_val = x_val[:n_display]
    x_corrupted = transform_blur(x_val)

    fig_dir = "./figs"
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    timestamp = datetime.now().strftime("%Y-%m-%dT%H%M%S%f")
    log_dir = "./logs/"
    log_path = os.path.join(
        log_dir, f"{model_name}_{timestamp}" + "_" + hps_info)
    writer = SummaryWriter(log_path)

    best_val_loss = np.inf
    for e in range(n_epochs):
        with tqdm(trainloader, desc=f"{e + 1}/{n_epochs} epochs") as t:
            total_train_count = 0
            total_train_loss = 0
            total_val_count = 0
            total_val_loss = 0
            for i, (x, _) in enumerate(t):
                model.train()
                loss = model(x.to(device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_train_count += x.shape[0]
                total_train_loss += loss.item()
                t.set_postfix({"elbo": - total_train_loss / total_train_count})
                if i == len(trainloader) - 1:
                    train_loss = total_train_loss / total_train_count
                    model.eval()
                    with torch.no_grad():
                        for x, _ in valloader:
                            loss = model.forward(x.to(device))
                            total_val_loss += loss.item()
                            total_val_count += x.shape[0]
                        val_loss = total_val_loss / total_val_count
                        t.set_postfix({
                            "train_elbo": - train_loss, "val_elbo": - val_loss})
                        writer.add_scalar("train_loss", train_loss, e+1)
                        writer.add_scalar("val_loss", val_loss, e+1)
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            try:
                                os.remove(chkpt_path)
                            except NameError:
                                pass
                            except FileNotFoundError:
                                print("File not found!")
                            chkpt_path = os.path.join(
                                chkpt_dir,
                                f"{dataset}_{model_name}_{alpha:.0e}fd_{e+1}.pt"
                            )
                            torch.save(model.state_dict(), chkpt_path)
                        # recover from corrupted validation data
                        x_ = model.reconst(x_corrupted.to(device)).detach().cpu()
                    xx_ = torch.cat([x_corrupted, x_], dim=0)
                    # convert to 2d numpy array in the order (H, W, C)
                    npimg = make_grid(xx_, nrow=8).permute(1, 2, 0).numpy()
                    img_format = "HWC"
                    if npimg.shape[2] == 1:
                        img_format = "HW"
                        npimg = npimg.squeeze(axis=2)
                    writer.add_image("deblurred_examples", npimg, e+1, dataformats=img_format)
                    plt.imsave(os.path.join(
                        fig_dir,
                        f"{dataset}_{model_name}_{alpha:.0e}fd_{e+1}.jpg"
                    ), npimg)
    writer.flush()  # make sure that all pending events have been written to disk
    writer.close()
