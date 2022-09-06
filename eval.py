if __name__ == "__main__":
    import os
    import math
    from tqdm import trange
    import torch
    from datasets import DATA_INFO
    from metrics.fid_score import InceptionStatistics, get_precomputed, calc_fd
    from metrics.precision_recall import ManifoldBuilder, Manifold, calc_pr
    from functools import partial
    from copy import deepcopy
    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument("--root", default="~/datasets", type=str)
    parser.add_argument("--dataset", choices=["mnist", "cifar10", "celeba"], default="cifar10")
    parser.add_argument("--model", choices=["gan", "vae", "vaegan"], default="gan")
    parser.add_argument("--backbone", choices={"dcgan", "resnet"}, default="dcgan")
    parser.add_argument("--base-ch", default=64, type=int)
    parser.add_argument("--latent-dim", default=128, type=int)
    parser.add_argument("--reconst-ch", default=64, type=int)
    parser.add_argument("--out-act", choices={"tanh", "sigmoid"}, default="tanh", type=str)
    parser.add_argument("--model-device", default=0, type=int)
    parser.add_argument("--eval-device", default=0, type=int)
    parser.add_argument("--eval-batch-size", default=512, type=int)
    parser.add_argument("--eval-total-size", default=50000, type=int)
    parser.add_argument("--nhood-size", default=3, type=int)
    parser.add_argument("--row-batch-size", default=10000, type=int)
    parser.add_argument("--col-batch-size", default=10000, type=int)
    parser.add_argument("--chkpt", default="./chkpts/cifar10_dcgan.pt", type=str)
    parser.add_argument("--precomputed-dir", default="./precomputed", type=str)
    parser.add_argument("--metrics", nargs="+", default=["fid", "pr"], type=str)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--anti-artifact", action="store_true")

    args = parser.parse_args()

    root = os.path.expanduser(args.root)
    chkpt_path = args.chkpt
    try:
        dataset, model_name, *extra_info = os.path.basename(chkpt_path).split("_")
        assert dataset in {"mnist", "cifar10", "celeba"}
    except (ValueError, AssertionError):
        dataset = args.dataset
        model_name = args.model
    model_device = torch.device(f"cuda:{args.model_device}" if torch.cuda.is_available() else "cpu")
    eval_device = torch.device(f"cuda:{args.eval_device}" if torch.cuda.is_available() else "cpu")

    print(f"Dataset: {dataset}")

    in_ch = DATA_INFO[dataset]["channels"]
    image_res = DATA_INFO[dataset]["resolution"][0]

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

    if model_name == "vaegan":
        model_configs["reconst_ch"] = reconst_ch

    args = parser.parse_args()
    os.environ["BACKBONE"] = backbone = args.backbone
    os.environ["ANTI_ARTIFACT"] = "true" if args.anti_artifact else ""

    from models import MODEL_LIST

    model = MODEL_LIST[model_name](**model_configs)

    model.to(model_device)
    model.eval()

    print(f"Model: {model_name}")
    print(f"Backbone: {backbone}")
    print(f"Output activation: {out_act}")

    chkpt = torch.load(chkpt_path, map_location=model_device)
    if "model" in chkpt:
        chkpt = chkpt["model"]
    for k in list(chkpt.keys()):
        if chkpt[k] is None:
            del chkpt[k]
    model.load_state_dict(chkpt, strict=False)
    del chkpt

    class OutputTransform:
        def __init__(self, model, transform):
            self.model = model
            self.transform = transform
        def sample_x(self, n):
            return self.transform(self.model.sample_x(n))
    if out_act == "sigmoid":
        model = OutputTransform(model, transform=lambda x: 2 * x - 1)

    precomputed_dir = args.precomputed_dir
    eval_batch_size = args.eval_batch_size
    eval_total_size = args.eval_total_size

    def eval_fid():
        istats = InceptionStatistics(device=eval_device)
        true_mean, true_var = get_precomputed(dataset, download_dir=precomputed_dir)
        istats.reset()

        num_eval_batches = math.ceil(eval_total_size / eval_batch_size)
        with torch.inference_mode():
            for i in trange(num_eval_batches):
                x = model.sample_x(
                    (eval_total_size % eval_batch_size or eval_batch_size)
                    if i == num_eval_batches - 1 else eval_batch_size)
                istats(x.to(eval_device))
        gen_mean, gen_var = istats.get_statistics()
        fid = calc_fd(gen_mean, gen_var, true_mean, true_var)
        return fid

    row_batch_size = args.row_batch_size
    col_batch_size = args.col_batch_size
    nhood_size = args.nhood_size


    def eval_pr():
        decimal_places = math.ceil(math.log(eval_total_size, 10))
        str_fmt = f".{decimal_places}f"
        _ManifoldBuilder = partial(
            ManifoldBuilder, extr_batch_size=eval_batch_size, max_sample_size=eval_total_size,
            row_batch_size=row_batch_size, col_batch_size=col_batch_size, nhood_size=nhood_size, device=eval_device)
        manifold_path = os.path.join(precomputed_dir, f"pr_manifold_{dataset}.pt")
        if not os.path.exists(manifold_path):
            dataset_kwargs = {
                "celeba": {"split": "all"},
            }.get(dataset, {"train": True})
            transform = DATA_INFO[dataset]["_transform"]
            manifold_builder = _ManifoldBuilder(
                data=DATA_INFO[dataset]["data"](root=root, transform=transform, **dataset_kwargs))
            manifold_builder.save(manifold_path)
            true_manifold = deepcopy(manifold_builder.manifold)
            del manifold_builder
        else:
            true_manifold = torch.load(manifold_path)
        gen_manifold = deepcopy(_ManifoldBuilder(model=model).manifold)

        precision, recall = calc_pr(
            gen_manifold, true_manifold,
            row_batch_size=row_batch_size, col_batch_size=col_batch_size, device=eval_device)
        return f"{precision:{str_fmt}}/{recall:{str_fmt}}"

    def warning(msg):
        def print_warning():
            print(msg)
        return print_warning

    for metric in set(args.metrics):
        result = {"fid": eval_fid, "pr": eval_pr}.get(metric, warning("Unsupported metric passed! Ignore."))()
        print(f"{metric.upper()}: {result}")
        if args.save:
            csv_path = os.path.join(os.path.dirname(chkpt_path), f"evaluation.csv")
            header = not os.path.exists(csv_path)
            with open(csv_path, "a+") as f:
                if header:
                    f.write("dataset, model name, extra info, metric, result\n")
                f.write(f"{dataset}, {model_name}, {' '.join(extra_info)}, {metric}, {result}\n")
