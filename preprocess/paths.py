"""Paths shared by the preprocessing scripts. Override with environment variables."""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
# Sportec (IDSSE) open data: DFL_04_03_positions_raw_observed_*.xml, event/, metadata/
SPORTEC_RAW = os.environ.get("SPORTEC_RAW", os.path.join(REPO, "sportec_raw"))
# 5 FPS player-position tables from the ball-recovery benchmark (recovery_bench/inputs/<match>_input.csv)
BENCH_INPUTS = os.environ.get("BENCH_INPUTS", os.path.join(REPO, "recovery_bench", "inputs"))
# new_data ball-missing masks (TODO, owned by another team member): <ND_MASK_DIR>/<match>_nd_mask.npy
ND_MASK_DIR = os.environ.get("ND_MASK_DIR", os.path.join(REPO, "work", "nd_mask"))
WORK = os.environ.get("WORK_DIR", os.path.join(REPO, "work"))
BALL5 = os.path.join(WORK, "ball5")        # make_ball5.py
BASE = os.path.join(WORK, "base")          # step1
TREE = os.path.join(WORK, "tree_prob.npz") # step2
LABELS = os.path.join(WORK, "labels")      # relabel_controlled_possession.py
DATASET = os.path.join(WORK, "dataset")    # build_npz.py -> <match>.npz used for training
THIRD_PARTY = os.path.join(REPO, "third_party")
MATCHES = ["J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9"]
