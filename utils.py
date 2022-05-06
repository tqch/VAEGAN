import torch


def dict2str(d):
    out_str = []
    for k, v in d.items():
        out_str.append(str(k))
        if isinstance(v, (list, tuple)):
            v = "_".join(list(map(str, v)))
        elif isinstance(v, float):
            v = f"{v:.0e}"
        elif isinstance(v, dict):
            v = dict2str(v)
        out_str.append(str(v))
    out_str = "_".join(out_str)
    return out_str


def restart_from_chkpt(chkpt_path, model, optimizers, device=torch.device("cpu")):
    chkpt = torch.load(chkpt_path, map_location=device)
    model.load_state_dict(chkpt["model"])
    for k in optimizers.keys():
        optimizers[k].load_state_dict(chkpt[k])
    fid = chkpt["fid"]
    return fid, model, optimizers


class RunningStatistics:
    def __init__(self, **kwargs):
        self.count = 0
        self.stats = []
        for k, v in kwargs.items():
            self.stats.append((k, v))
        self.stats = dict(self.stats)

    def reset(self):
        self.count = 0
        for k in self.stats:
            self.stats[k] = 0

    def update(self, n, **kwargs):
        self.count += n
        for k, v in kwargs.items():
            self.stats[k] = self.stats.get(k, 0) + v

    def extract(self):
        avg_stats = []
        for k, v in self.stats.items():
            avg_stats.append((k, v/self.count))
        return dict(avg_stats)

    def __repr__(self):
        out_str = "Count(s): {}\n"
        out_str += "Statistics:\n"
        for k in self.stats:
            out_str += f"\t{k} = {{{k}}}\n"  # double curly-bracket to escape
        return out_str.format(self.count, **self.stats)
