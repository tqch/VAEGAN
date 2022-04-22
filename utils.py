def dict2str(d):
    out_str = []
    for k, v in d.items():
        out_str.append(str(k))
        if isinstance(v, (list, tuple)):
            vv = "_".join(list(map(str, v)))
            out_str.append(vv)
    out_str = "_".join(out_str)
    return out_str