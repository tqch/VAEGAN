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
