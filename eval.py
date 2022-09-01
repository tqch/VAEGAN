if __name__ == "__main__":
    import os
    import math
    from tqdm import trange
    import torch
    from datasets import DATA_INFO
    from metrics.fid_score import InceptionStatistics, get_precomputed, calc_fd
    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument("--dataset", choices=["mnist", "cifar10", "celeba"], default="cifar10")
    parser.add_argument("--model", choices=["gan", "vae", "vaegan"], default="gan")
    parser.add_argument("--backbone", choices={"dcgan", "resnet"}, default="dcgan")
    parser.add_argument("--base-ch", default=64, type=int)
    parser.add_argument("--latent-dim", default=128, type=int)
    parser.add_argument("--reconst-ch", default=64, type=int)
    parser.add_argument("--out-act", choices={"tanh", "sigmoid"}, default="tanh", type=str)
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--eval-device", default=0, type=int)
    parser.add_argument("--eval-batch-size", default=512, type=int)
    parser.add_argument("--eval-total-size", default=50000, type=int)
    parser.add_argument("--chkpt", default="./chkpts/cifar10_dcgan.pt", type=str)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--anti-artifact", action="store_true")

    args = parser.parse_args()

    chkpt_path = args.chkpt
    try:
        dataset, model_name, *extra_info = os.path.basename(chkpt_path).split("_")
        assert dataset in {"mnist", "cifar10", "celeba"}
    except (ValueError, AssertionError):
        dataset = args.dataset
        model_name = args.model
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    eval_device = torch.device(f"cuda:{args.eval_device}" if torch.cuda.is_available() else "cpu")

    out_act = args.out_act
    if out_act == "sigmoid":
        input_transform = lambda x: 2 * x - 1
    else:
        input_transform = lambda x: x
    istats = InceptionStatistics(input_transform=input_transform, device=eval_device)
    target_mean, target_var = get_precomputed(dataset)

    print(f"Dataset: {dataset}")

    in_ch = DATA_INFO[dataset]["channels"]
    image_res = DATA_INFO[dataset]["resolution"][0]

    base_ch = args.base_ch
    latent_dim = args.latent_dim
    reconst_ch = args.reconst_ch

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

    model.to(device)
    model.eval()

    print(f"Model: {model_name}")
    print(f"Backbone: {backbone}")
    print(f"Output activation: {out_act}")

    chkpt = torch.load(chkpt_path, map_location=device)
    if "model" in chkpt:
        chkpt = chkpt["model"]
    for k in list(chkpt.keys()):
        if chkpt[k] is None:
            del chkpt[k]
    model.load_state_dict(chkpt, strict=False)
    del chkpt

    eval_batch_size = args.eval_batch_size
    eval_total_size = args.eval_total_size
    istats.reset()

    num_eval_batches = math.ceil(eval_total_size / eval_batch_size)
    with torch.inference_mode():
        for i in trange(num_eval_batches):
            x = model.sample_x(
                (eval_total_size % eval_batch_size or eval_batch_size)
                if i == num_eval_batches - 1 else eval_batch_size)
            istats(x.to(eval_device))
    gen_mean, gen_var = istats.get_statistics()
    eval_fid = calc_fd(gen_mean, target_mean, gen_var, target_var)
    print(f"FID: {eval_fid}")
    if args.save:
        csv_path = os.path.join(os.path.dirname(chkpt_path), "evaluation.csv")
        header = not os.path.exists(csv_path)
        with open(csv_path, "a+") as f:
            if header:
                f.write("dataset, model name, extra info, fid\n")
            f.write(f"{dataset}, {model_name}, {' '.join(extra_info)}, {eval_fid}\n")
