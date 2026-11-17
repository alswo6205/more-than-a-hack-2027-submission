"""Paths shared by the preprocessing scripts. Override with environment variables."""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
# Sportec (IDSSE) open data: DFL_04_03_positions_raw_observed_*.xml, event/, metadata/
SPORTEC_RAW = os.environ.get("SPORTEC_RAW", os.path.join(REPO, "sportec_raw"))
# new_data/<match>_with_gt.csv: 5 FPS frames, ball columns, ball_masked (= nd_mask)
NEW_DATA = os.environ.get("NEW_DATA", os.path.join(REPO, "new_data"))
WORK = os.environ.get("WORK_DIR", os.path.join(REPO, "work"))
BALL5 = os.path.join(WORK, "ball5")        # make_ball5.py
BENCH_INPUTS = os.environ.get("BENCH_INPUTS", os.path.join(WORK, "inputs"))  # make_inputs.py
BASE = os.path.join(WORK, "base")          # step1
TREE = os.path.join(WORK, "tree_prob.npz") # step2
LABELS = os.path.join(WORK, "labels")      # relabel_controlled_possession.py
DATASET = os.path.join(WORK, "dataset")    # build_npz.py -> <match>.npz used for training
THIRD_PARTY = os.path.join(REPO, "third_party")
MATCHES = ["J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9"]
