# Possession pipeline: sub_13 (branch `sub13`)

This branch has everything needed to rebuild the `sub_13` submission. It covers preprocessing, training and prediction.

**All models were trained only on the 7 public Sportec (IDSSE) matches**: J03WMX, J03WN1, J03WOH, J03WOY, J03WPY, J03WQQ and J03WR9.
The competition match (BAR-ATH) is used only for prediction. It is never used for training or model selection.
The trained models are included, so you can go straight to steps 4 and 5 without training anything.

## What sub_13 is

| Part | Method |
|---|---|
| A/D (alive / dead) | simple average of the V1 D probability and the tree `p_dead`. V1 is trained on the middle label (`gt_mid`) |
| H/A (home / away) | E4HA, a ball-input team model trained on the middle label |
| Decoder | one 3-state Viterbi per half. H↔A switch cost 2.0, H/A↔D switch cost 2.5. 5 FPS is mapped to 25 FPS by nearest step |
| Ball hiding | an auxiliary Viterbi uses the same A/D and the V1 team probability. Frames it labels D have the ball hidden from E4HA |
| Fixed rule | 110808–113741 (before the second-half kick-off) is D (FAQ.md). Frames outside the halves are X |

## Folder layout

| Path | Contents |
|---|---|
| `preprocess/` | Sportec → one training npz per match (see the table below) |
| `train/train_v1.py` | V1 on `gt_mid` → `v1_mid.pt` |
| `train/train_e4ha.py` | E4HA on `gt_mid` → `e4ha_mid.pt` |
| `train/common_train.py` | shared data loading and the train/validation split (train on 6 matches, select on J03WPY) |
| `deadball/` | tree A/D model: code and trained model (`deadball/model/`). See `deadball/README.md` |
| `models/v1_mid.zip`, `models/e4ha_mid.zip` | trained weights. Despite the `.zip` name these are torch `.pt` files |
| `run_sub13.py` | builds the sub_13 CSV for BAR-ATH |
| `third_party/` | copied external code (see Sources) |
| `work/` | generated intermediates (git-ignored) |

### Preprocessing steps

| Script | Output |
|---|---|
| `make_ball5.py` (new) | `work/ball5/<m>_ball5.csv`: ball x, y, z, BallPossession and BallStatus from the Sportec position XML, every 5th frame of each half |
| `step1_base_npz.py` | `work/base/<m>.npz`: positions, ball, original labels |
| `step2_tree_pD_lomo.py` | `work/tree_prob.npz`: leave-one-match-out tree D probability (`tree_pD`) |
| `relabel_controlled_possession.py` | `work/labels/<m>_labels.csv`: event-based relabelling (`poss_orig`, `poss_loose`, `poss_mid`, `poss_strict`) |
| `build_npz.py` (new) | `work/dataset/<m>.npz` with the keys `xy, teams, frame, period, gt_orig, ball, nd_mask, tree_pD, gt_loose, gt_mid` |

> **TODO – new_data ball-missing mask (`nd_mask`)**: another team member produces this mask, so it is not built here.
> Put it at `work/nd_mask/<match>_nd_mask.npy` (bool, one value per 5 Hz frame), or set `ND_MASK_DIR`, before running `build_npz.py`.
> Without it `build_npz.py` writes an all-False mask and prints a warning, and `train_e4ha.py` stops
> (it samples ball-gap lengths from the mask). V1 training and `run_sub13.py` do not use it.

## Data setup

Neither folder is committed (both are in `.gitignore`). Paths are relative to the repo root; override with env vars `SPORTEC_RAW` / `BENCH_INPUTS`.

### `sportec_raw/` (Sportec / IDSSE open data)
- Source: TODO (download URL / DOI of the Sportec open dataset)
- Expected layout: `sportec_raw/DFL_04_03_positions_raw_observed_*.xml`, `sportec_raw/event/`, `sportec_raw/metadata/`
- Steps: TODO (download → unzip → place files as above)

### `recovery_bench/inputs/` (5 FPS player positions, `<match>_input.csv`)
- Source: TODO (who produces it / which script generates it)

## How to run

All commands run from this repository's root unless noted. `<comp>` is the competition repository (it has `games.py`, `data/` and `evaluation/`).

**1. Install**
```
pip install -r requirements.txt
```

**2. Preprocess (Sportec only)**

Inputs:
- Sportec raw XML folder (`DFL_04_03_positions_raw_observed_*.xml`, `event/`, `metadata/`). Set it with the `SPORTEC_RAW` environment variable.
- `recovery_bench/inputs/<m>_input.csv` (5 FPS player positions). Set it with `BENCH_INPUTS`; the default is `recovery_bench/inputs` in this repo.

```
cd preprocess
python make_ball5.py
python step1_base_npz.py
python step2_tree_pD_lomo.py
python relabel_controlled_possession.py --raw "$SPORTEC_RAW" --out ../work/labels
python build_npz.py
cd ..
```

**3. Train (optional: trained weights are already in `models/`)**
```
python train/train_v1.py        # 30 epochs max, early stopping, about 1 h on GPU
python train/train_e4ha.py      # 20 epochs
```
- Use `--epochs 1` for a quick smoke test.
- The outputs go to `work/train_out/v1_mid.pt` and `work/train_out/e4ha_mid.pt`. To use them, copy them to `models/v1_mid.zip` and `models/e4ha_mid.zip`.
- The included weights were trained on Kaggle (GPU), so retraining on another device can give slightly different weights.

**4. Tree probabilities for BAR-ATH (run in `<comp>`)**
```
python <this repo>/deadball/predict_atd.py --out deadball_out
```
This writes `deadball_out/deadball_frames.csv` (`video_frame, half, graded, p_dead, dead`).

**5. sub_13 (run in `<comp>`)**
```
python <this repo>/run_sub13.py --repo . --tree deadball_out/deadball_frames.csv --out combo_submissions/sub_13.csv
```
- `--data-dir` sets the data folder (the default comes from `games.asset_paths`).
- No training is needed, and it runs on CPU in about 20 s.

**6. Validate**
```
python evaluation/validate_submission.py combo_submissions/sub_13.csv
```

## Checks done on this branch
- `make_ball5.py` gives the same values as the earlier `gk_test/*_ball5.csv` for all 7 matches.
- `run_sub13.py` gives the same output as the earlier `combo_submissions/sub_13.csv`: 0 of 199,319 frames differ.
- `deadball/predict_atd.py` reproduces the earlier `deadball_frames.csv` exactly.
- `train_v1.py` and `train_e4ha.py` each ran for 1 epoch without errors.

## Sources of copied code
| Path | Source |
|---|---|
| `third_party/pa_sm/` | V1 repository `barcelona-possession_ad` (team member's repo), package `pa_sm/`. Unmodified |
| `third_party/models_new.py` | `mtah-e4` repository (`models_new.py`: Set Attention / PMA frame encoder). Unmodified |
| `preprocess/step1_*.py`, `step2_*.py`, `features.py`, `config.py`, `common.py` | earlier data-build code (`recovery_bench_check/data_build/`). Only paths were changed, and step2's evaluation table was dropped |
| `train/train_*.py` | `data_build/train_and_eval_combo.py` (= `kaggle/combo_experiment.ipynb`). The `mid`-label parts only, with the same computation |
| `run_sub13.py` | `recovery_bench_check/apply_combos.py`. Only the sub_13 branch was kept |

## Data policy
Competition data (`data/`, `deadball_out/`, submission CSVs, anything BAR-ATH) is not stored in this repository, and `.gitignore` blocks it.
