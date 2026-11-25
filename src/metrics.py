"""Metrics pool integer counts over the full evaluation set."""

import numpy as np
from scipy import ndimage


def cm(y, p, k):
    y, p = np.asarray(y).ravel().astype("int64"), np.asarray(p).ravel()
    v = y != 255
    return np.bincount(k * y[v] + p[v], minlength=k * k).reshape(k, k)


def scores(matrix, names):
    a = np.asarray(matrix)
    t = np.diag(a)
    y = a.sum(1)
    p = a.sum(0)
    u = y + p - t

    def ratio(x, n):
        return np.divide(x, n, out=np.full(len(x), np.nan), where=n > 0)

    iou = ratio(t, u)
    dice = ratio(2 * t, y + p)
    prec = np.where(u > 0, np.nan_to_num(ratio(t, p)), np.nan)
    rec = np.where(u > 0, np.nan_to_num(ratio(t, y)), np.nan)

    def f(v):
        return float(v) if np.isfinite(v) else None

    return dict(
        accuracy=f(t.sum() / a.sum()) if a.sum() else None,
        macro_f1=f(np.nanmean(dice)),
        macro_precision=f(np.nanmean(prec)),
        macro_recall=f(np.nanmean(rec)),
        miou=f(np.nanmean(iou)),
        mean_dice=f(np.nanmean(dice)),
        confusion_matrix=a.tolist(),
        per_class=[
            dict(
                name=n,
                support=int(y[i]),
                predicted=int(p[i]),
                precision=f(prec[i]),
                recall=f(rec[i]),
                iou=f(iou[i]),
                dice=f(dice[i]),
            )
            for i, n in enumerate(names)
        ],
    )


def calibration(p, y):
    pred = p.argmax(1)
    conf = p.max(1)
    bins = []
    for i in range(15):
        v = (conf >= i / 15) & ((conf <= 1) if i == 14 else (conf < (i + 1) / 15))
        if v.any():
            bins.append(
                dict(
                    lower=i / 15,
                    n=int(v.sum()),
                    accuracy=float((pred[v] == y[v]).mean()),
                    confidence=float(conf[v].mean()),
                )
            )
    return dict(
        ece=sum(v["n"] * abs(v["confidence"] - v["accuracy"]) for v in bins) / len(y),
        nll=float(-np.log(p[np.arange(len(y)), y].clip(1e-12)).mean()),
        bins=bins,
    )


def spatial(y, p):
    valid = y != 255
    edge = np.zeros_like(valid)
    small = np.zeros_like(valid)
    for axis in (0, 1):
        left, r = [slice(None)] * 2, [slice(None)] * 2
        left[axis] = slice(1, None)
        r[axis] = slice(None, -1)
        left, r = tuple(left), tuple(r)
        diff = valid[left] & valid[r] & (y[left] != y[r])
        edge[left] |= diff
        edge[r] |= diff
    for c in np.unique(y[valid & (y >= 4)]):
        comp, _ = ndimage.label(y == c)
        sizes = np.bincount(comp.ravel())
        small |= (comp > 0) & (sizes[comp] <= 16)
    surface = valid & (y < 4)
    wrong = p != y
    return np.array(
        [
            edge.sum(),
            (edge & wrong).sum(),
            small.sum(),
            (small & wrong).sum(),
            surface.sum(),
            (surface & (p >= 4)).sum(),
        ]
    )
