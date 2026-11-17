"""Build the YSAL submission (final model) for the competition match (BAR-ATH). No training, CPU is fine.

Final model
  A/D : (V1[mid] D probability + tree p_dead) / 2
  H/A : E4HA[mid] (ball-input team model)
  decoder: one 3-state Viterbi per half (H<->A switch 2.0, H/A<->D switch 2.5), 5 FPS -> 25 FPS by nearest step
  E4HA input: the ball is hidden in frames that an auxiliary Viterbi (same A/D, V1 team probability) labels D
  fixed: 110808~113741 (before the second-half kick-off) = D (FAQ.md), outside the halves = X

  python run_predict.py --repo <competition repo> [--data-dir <data>] [--tree deadball_out/deadball_frames.csv] [--out YSAL.csv]

Extracted from apply_combos.py (only the final-model branch kept; computation unchanged).
"""
import argparse, os, sys, time
import xml.etree.ElementTree as ET
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
N_FRAMES = 199319
HALVES = {1: (11185, 90651), 2: (110808, 190197)}      # DATA.md (video frame, inclusive)
PRE_KO = (110808, 113741)                                # FAQ.md: all D
STEP = 5
V1_PITCH = (105.0, 68.0)                                 # same as V1 prep_atd (configs/base.yaml)
E4_PITCH = (105.3, 68.0)                                 # same as mtah-e4 run.py (DATA.md measured pitch)


def read_meta(path):
    root = ET.parse(path).getroot()
    tid = [t.get("id") for t in root.iter("Team")]
    return np.array([0 if p.get("teamId") == tid[0] else 1 for p in root.iter("Player")], np.int8)   # 0 = home (first Team)


def read_raw(path, n_slots, n_frames):
    """'<frame_id>:<p1>;<p2>;...:<ball>' -> 0~1 coordinates (video frame = frame_id - 1)"""
    X = np.full((n_frames, n_slots, 2), np.nan, np.float32); B = np.full((n_frames, 2), np.nan, np.float32)
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            i = int(parts[0]) - 1
            if not 0 <= i < n_frames:
                continue
            for j, g in enumerate(parts[1].split(";")[:n_slots]):
                if g and g[0] not in "Nn":
                    v = g.split(","); X[i, j] = (float(v[0]), float(v[1]))
            b = parts[2]
            if b and b[0] not in "Nn":
                v = b.split(","); B[i] = (float(v[0]), float(v[1]))
    return X, B


def vit3(pH, pA, pD, cha=2.0, cad=2.5):
    l = np.log(np.clip(np.c_[pH, pA, pD], 1e-7, 1)); n = len(l); dp = l[0].copy(); bk = np.zeros((n, 3), np.int8)
    pen = np.array([[0, cha, cad], [cha, 0, cad], [cad, cad, 0]])
    for t in range(1, n):
        v = dp[:, None] - pen; bk[t] = v.argmax(0); dp = l[t] + v.max(0)
    p = np.zeros(n, np.int8); p[-1] = dp.argmax()
    for t in range(n - 1, 0, -1):
        p[t - 1] = bk[t, p[t]]
    return p                                   # 0 = H, 1 = A, 2 = D


# ---------------- E4HA (same computation as base_feats / assemble / e4_predict in train/train_e4ha.py) ----------------
T, NMAX = 50, 26


def e4_net(models_new):
    import torch.nn as nn
    class Net(nn.Module):
        def __init__(s, d=64):
            super().__init__(); s.enc = models_new.FrameEncoder(10, d, True)
            s.gru = nn.GRU(2 * d, d, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
            s.head = nn.Linear(2 * d, 3); s.bhead = nn.Linear(2 * d, 1)
        def forward(s, f, m, b):
            B_, T_, N, F = f.shape; x, _, _ = s.enc(f.reshape(B_ * T_, N, F), m.reshape(B_ * T_, N), b.reshape(B_ * T_, 3))
            h, _ = s.gru(x.reshape(B_, T_, -1)); return s.head(h), s.bhead(h).squeeze(-1)
    return Net()


def e4_inputs(X, Y, team):
    """X, Y (n, slots) metres. team -1 home / 1 away. left team side=-1"""
    n = len(X); hl = np.nanmean(X[:, team == -1]) < np.nanmean(X[:, team == 1])
    side = np.where(team == -1, -1. if hl else 1., 1. if hl else -1.)
    feat = np.zeros((n, NMAX, 7), np.float32); pxy = np.zeros((n, NMAX, 2), np.float32); mask = np.zeros((n, NMAX), bool); cnt = np.zeros(n, int)
    for j in range(X.shape[1]):
        x, y = X[:, j].astype(float), Y[:, j].astype(float)
        vx = np.r_[0, np.diff(x)] / 0.2; vy = np.r_[0, np.diff(y)] / 0.2; jump = np.hypot(vx, vy) > 12; vx[jump] = np.nan; vy[jump] = np.nan
        vk = ~np.isnan(vx); vx = np.nan_to_num(vx); vy = np.nan_to_num(vy)
        rows = np.flatnonzero(~np.isnan(x)); rows = rows[cnt[rows] < NMAX]; c = cnt[rows]
        feat[rows, c] = np.stack([np.full(len(rows), side[j]), x[rows] / 52.5 - 1, y[rows] / 34 - 1, vx[rows] / 8, vy[rows] / 8, np.hypot(vx[rows], vy[rows]) / 8, vk[rows]], 1)
        pxy[rows, c] = np.stack([x[rows], y[rows]], 1); mask[rows, c] = True; cnt[rows] += 1
    return feat, pxy, mask, hl


def e4_home_prob(model, feat, pxy, mask, hl, ball, vis, dev):
    import torch, torch.nn.functional as Fn
    dx = np.where(vis[:, None], pxy[..., 0] - ball[:, None, 0], 0); dy = np.where(vis[:, None], pxy[..., 1] - ball[:, None, 1], 0)
    F = np.concatenate([feat, np.stack([dx / 20, dy / 20, np.hypot(dx, dy) / 20], -1) * mask[..., None]], -1).astype(np.float32)
    Bt = np.stack([np.where(vis, ball[:, 0] / 52.5 - 1, 0), np.where(vis, ball[:, 1] / 34 - 1, 0), vis.astype(float)], -1).astype(np.float32)
    n = len(F); lp = np.zeros((n, 2)); w = np.zeros(n)
    st = sorted(set(list(range(0, max(1, n - T + 1), 25)) + [max(0, n - T)])); tri = np.minimum(np.arange(1, T + 1), np.arange(T, 0, -1)).astype(float)
    with torch.no_grad():
        for k in range(0, len(st), 64):
            ss = st[k:k + 64]
            o, _ = model(*[torch.from_numpy(np.stack([a[s:s + T] for s in ss])).to(dev) for a in (F, mask, Bt)])
            o = Fn.log_softmax(o[..., :2], -1).cpu().numpy()
            for j, s in enumerate(ss):
                L = min(T, n - s); lp[s:s + L] += o[j, :L] * tri[:L, None]; w[s:s + L] += tri[:L]
    pl = np.exp(lp[:, 0] / np.maximum(w, 1e-9))          # probability that the left team has the ball
    return pl if hl else 1 - pl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="competition repo (games.py)")
    ap.add_argument("--data-dir"); ap.add_argument("--atd-meta"); ap.add_argument("--atd-raw")
    ap.add_argument("--models", default=os.path.join(HERE, "models"))
    ap.add_argument("--tree", default="deadball_out/deadball_frames.csv")
    ap.add_argument("--out", default="YSAL.csv"); ap.add_argument("--n-frames", type=int, default=N_FRAMES)
    a = ap.parse_args(); repo = os.path.abspath(os.path.expanduser(a.repo)); t0 = time.time()
    sys.path.insert(0, repo); sys.path.insert(0, os.path.join(HERE, "third_party"))
    import torch, models_new
    from pa_sm.pipeline import load_checkpoint, predict_arrays
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if a.atd_meta and a.atd_raw:
        meta, raw = a.atd_meta, a.atd_raw
    else:
        from games import asset_paths
        p = asset_paths(data_dir=a.data_dir); meta, raw = p["atd_meta"], p["atd_raw"]
    v1_path = os.path.join(a.models, "v1_mid.zip"); e4_path = os.path.join(a.models, "e4ha_mid.zip")
    for f in [meta, raw, a.tree, v1_path, e4_path]:
        if not os.path.exists(f):
            raise SystemExit(f"missing file: {f}")

    team01 = read_meta(meta); team = np.where(team01 == 0, -1, 1).astype(np.int8)
    X01, B01 = read_raw(raw, len(team), a.n_frames)
    print(f"[1] ATD read: {len(team)} slots, {time.time()-t0:.0f}s", flush=True)
    tree = pd.read_csv(a.tree).set_index("video_frame")["p_dead"]
    model, cfg, _, d_ = load_checkpoint(v1_path, dev)
    e4 = e4_net(models_new); e4.load_state_dict(torch.load(e4_path, map_location=dev)); e4 = e4.to(dev).eval()
    print(f"[2] models loaded ({dev})", flush=True)

    arr = np.full(a.n_frames, "X", dtype=object)
    for h, (s, e) in HALVES.items():
        if e >= a.n_frames:
            continue
        idx = np.arange(s, e + 1, STEP)
        xy_v1 = np.stack([X01[idx, :, 0] * V1_PITCH[0], X01[idx, :, 1] * V1_PITCH[1]], -1).astype(np.float32)
        Xe = X01[idx, :, 0] * E4_PITCH[0]; Ye = X01[idx, :, 1] * E4_PITCH[1]; ball = np.c_[B01[idx, 0] * E4_PITCH[0], B01[idx, 1] * E4_PITCH[1]]
        feat, pxy, mask, hl = e4_inputs(Xe, Ye, team)
        pT = tree.reindex(idx).to_numpy(float)
        if np.isnan(pT).any():
            raise SystemExit("frames missing in deadball_frames.csv (rerun deadball/predict_atd.py)")
        n = cfg["preprocessing"]["max_slots"]; xy_p = np.full((len(idx), n, 2), np.nan, np.float32); xy_p[:, :len(team)] = xy_v1
        tm = np.zeros(n, np.int8); tm[:len(team)] = team
        pr, _ = predict_arrays(model, cfg, xy_p, tm, d_)
        pD1 = pr[:, 2]; h1 = pr[:, 0] / np.clip(pr[:, 0] + pr[:, 1], 1e-9, None); pD = (pD1 + pT) / 2
        aux = vit3((1 - pD) * h1, (1 - pD) * (1 - h1), pD)                  # auxiliary Viterbi: only used to hide the ball
        vis = np.isfinite(ball[:, 0]) & (aux != 2)
        hp = e4_home_prob(e4, feat, pxy, mask, hl, ball, vis, dev)
        lab5 = vit3((1 - pD) * hp, (1 - pD) * (1 - hp), pD)
        pos = np.minimum((np.arange(e + 1 - s) + STEP // 2) // STEP, len(lab5) - 1)
        arr[s:e + 1] = np.array(["H", "A", "D"], dtype=object)[lab5[pos]]
        print(f"[3] half {h} done {time.time()-t0:.0f}s", flush=True)
    arr[PRE_KO[0]:PRE_KO[1] + 1] = "D"
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    pd.DataFrame({"frame": np.arange(a.n_frames), "label": arr}).to_csv(a.out, index=False)
    print(f"saved {a.out}  {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
