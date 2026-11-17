"""Shared settings for train_v1.py / train_e4ha.py (taken from data_build/train_and_eval_combo.py)."""
import os, sys, glob, numpy as np, torch
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "third_party"))
DATA = os.environ.get("GK_DATA", os.path.join(REPO, "work", "dataset"))
OUT = os.environ.get("GK_OUT", os.path.join(REPO, "work", "train_out"))
os.makedirs(OUT, exist_ok=True)
hit = glob.glob(DATA + "/**/J03WMX.npz", recursive=True)
assert hit and "gt_mid" in np.load(hit[0]).files, f"dataset not found under {DATA} (run preprocess/build_npz.py)"
DATA = os.path.dirname(hit[0]); dev = "cuda" if torch.cuda.is_available() else "cpu"; print("device", dev, DATA, flush=True)
M = ["J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9"]; VAL = "J03WPY"; TRAIN = [m for m in M if m != VAL]
LAB = "mid"     # middle label (gt_mid)
D = {m: dict(np.load(f"{DATA}/{m}.npz")) for m in M}
