"""Merge everything into one training npz per match: work/dataset/<match>.npz
keys: xy, teams, frame, period, gt_orig, ball, nd_mask, tree_pD, gt_loose, gt_mid
labels: 0 = home (BallPossession 1), 1 = away (BallPossession 2), 2 = dead

  python preprocess/build_npz.py

nd_mask (5 Hz ball-missing mask imitating new_data) is produced by another team member.
TODO: put it at ND_MASK_DIR/<match>_nd_mask.npy (bool, one value per 5 Hz frame).
If it is missing an all-False mask is written and a warning is printed
(train_e4ha.py needs a real mask: it samples ball-gap lengths from it).
"""
import os
import numpy as np, pandas as pd
from paths import BASE, TREE, LABELS, ND_MASK_DIR, DATASET, MATCHES

NSLOT = 42


def to_gt(poss):
    return np.where(poss == 0, 2, np.where(poss == 1, 0, 1)).astype(np.int8)


def main():
    os.makedirs(DATASET, exist_ok=True); tree = np.load(TREE)
    for m in MATCHES:
        b = np.load(os.path.join(BASE, f"{m}.npz")); fr = b["frame"]; per = b["period"]
        assert (b["teams"][NSLOT:] == 0).all()
        lab = pd.read_csv(os.path.join(LABELS, f"{m}_labels.csv")).set_index("frame_id").reindex(fr)
        assert lab.notna().all().all(), m
        gt_orig = b["gt"].astype(np.int8); assert (to_gt(lab.poss_orig.to_numpy()) == gt_orig).all(), m
        tp = np.zeros(len(fr), np.float32)
        for p in (1, 2):
            ix = np.flatnonzero(per == p); tp[ix] = tree[f"{m}_{p}"]
        f = os.path.join(ND_MASK_DIR, f"{m}_nd_mask.npy")
        if os.path.exists(f):
            nd = np.load(f).astype(bool); assert len(nd) == len(fr), m
        else:
            nd = np.zeros(len(fr), bool); print(f"WARNING {m}: {f} not found, nd_mask = all False (TODO)")
        np.savez_compressed(os.path.join(DATASET, f"{m}.npz"), xy=b["xy"][:, :NSLOT], teams=b["teams"][:NSLOT], frame=fr, period=per,
                            gt_orig=gt_orig, ball=b["ball"][:, :2].astype(np.float32), nd_mask=nd,
                            tree_pD=tp,
                            gt_loose=to_gt(lab.poss_loose.to_numpy()), gt_mid=to_gt(lab.poss_mid.to_numpy()))
        print(m, len(fr), "frames", flush=True)


if __name__ == "__main__":
    main()
