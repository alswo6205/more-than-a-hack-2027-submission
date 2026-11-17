"""Shared evaluation: frame F1 (dead), stoppage F1 (Lang 2022: >=2 s, IoU>=0.5), boundary error, recall by length."""
import numpy as np
from scipy.ndimage import median_filter

from config import HZ

def segs(dead, period):
    out = []
    for p in np.unique(period):
        ii = np.where(period == p)[0]
        d = dead[ii].astype(int)
        e = np.diff(np.r_[0, d, 0]); s = np.where(e == 1)[0]; t = np.where(e == -1)[0]
        out += [(ii[0] + a, ii[0] + b) for a, b in zip(s, t)]
    return out

def frame_scores(y_dead, p_dead):
    tp = (y_dead & p_dead).sum(); fp = (~y_dead & p_dead).sum(); fn = (y_dead & ~p_dead).sum()
    pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
    return dict(f_prec=pr, f_rec=rc, f_f1=2 * pr * rc / max(pr + rc, 1e-9), f_acc=(y_dead == p_dead).mean())

def stop_scores(y_dead, p_dead, period, min_s=2.0):
    G = [s for s in segs(y_dead, period) if (s[1] - s[0]) >= min_s * HZ]
    Pr = [s for s in segs(p_dead, period) if (s[1] - s[0]) >= min_s * HZ]
    used, matches = set(), []
    for gi, (a, b) in enumerate(G):
        best, bj = 0, None
        for pj, (c, d) in enumerate(Pr):
            if pj in used or d <= a or c >= b: continue
            iou = (min(b, d) - max(a, c)) / (max(b, d) - min(a, c))
            if iou > best: best, bj = iou, pj
        if bj is not None and best >= 0.5:
            used.add(bj); matches.append((gi, bj))
    tp = len(matches)
    pr = tp / max(len(Pr), 1); rc = tp / max(len(G), 1)
    ds = np.array([Pr[j][0] - G[i][0] for i, j in matches]) / HZ if matches else np.array([np.nan])
    de = np.array([Pr[j][1] - G[i][1] for i, j in matches]) / HZ if matches else np.array([np.nan])
    lens = np.array([(b - a) / HZ for a, b in G]); hit = np.zeros(len(G), bool)
    for i, _ in matches: hit[i] = True
    bins = {}
    for lo, hi in [(2, 5), (5, 10), (10, 30), (30, 1e9)]:
        k = (lens >= lo) & (lens < hi)
        bins[f"rec_{lo}-{hi if hi < 1e9 else 'inf'}s"] = hit[k].mean() if k.any() else np.nan
    return dict(s_prec=pr, s_rec=rc, s_f1=2 * pr * rc / max(pr + rc, 1e-9), n_true=len(G), n_pred=len(Pr),
                start_within2=np.mean(np.abs(ds) <= 2), end_within2=np.mean(np.abs(de) <= 2),
                start_mae=np.nanmean(np.abs(ds)), end_mae=np.nanmean(np.abs(de)), **bins)

def viterbi2(p, period, pen):
    """2-state (alive/dead) smoothing: emission log-prob, switching penalty pen (nats)."""
    out = np.zeros(len(p), bool)
    lp = np.log(np.clip(np.c_[1 - p, p], 1e-6, 1))
    for q in np.unique(period):
        ii = np.where(period == q)[0]; e = lp[ii]; n = len(ii)
        D = np.zeros((n, 2)); B = np.zeros((n, 2), int); D[0] = e[0]
        for t in range(1, n):
            for s in range(2):
                c = D[t - 1] - pen * (np.arange(2) != s)
                B[t, s] = c.argmax(); D[t, s] = c.max() + e[t, s]
        st = np.zeros(n, int); st[-1] = D[-1].argmax()
        for t in range(n - 1, 0, -1): st[t - 1] = B[t, st[t]]
        out[ii] = st == 1
    return out

def post(p, period, kind, param, thr=0.5):
    if kind == "none": return p >= thr
    if kind == "median":
        r = np.zeros(len(p), bool)
        for q in np.unique(period):
            ii = period == q; r[ii] = median_filter((p[ii] >= thr).astype(float), size=param, mode="nearest") > 0.5
        return r
    if kind == "viterbi": return viterbi2(p, period, param)

POST_GRID = [("none", None)] + [("median", w) for w in (5, 9, 19, 29, 45)] + [("viterbi", c) for c in (2, 4, 8, 16)]

def tune_post(p_val, y_dead_val, per_val):
    """choose post-processing on validation by stoppage F1 (tie: frame F1)."""
    best = None
    for kind, prm in POST_GRID:
        thrs = [0.5] if kind == "viterbi" else [0.3, 0.4, 0.5, 0.6]
        for thr in thrs:
            pd = post(p_val, per_val, kind, prm, thr)
            sc = stop_scores(y_dead_val, pd, per_val)["s_f1"] + 1e-3 * frame_scores(y_dead_val, pd)["f_f1"]
            if best is None or sc > best[0]: best = (sc, kind, prm, thr)
    return best[1:]
