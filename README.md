# YSAL — More Than A Hack 2027 possession pipeline

This repository has everything needed to rebuild the final YSAL submission. It covers preprocessing, training and prediction.

**Final model:** A/D = average of V1 and tree, H/A = E4HA, both trained on the controlled-possession (mid) labels.

**All models were trained only on the 7 public Sportec (IDSSE) matches**: J03WMX, J03WN1, J03WOH, J03WOY, J03WPY, J03WQQ and J03WR9.
The competition match (BAR-ATH) is used only for prediction. It is never used for training or model selection.
The trained models are included, so you can go straight to steps 4 and 5 without training anything.

## Final model

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
| `run_predict.py` | builds the final YSAL CSV for BAR-ATH |
| `third_party/` | copied external code (see Sources) |
| `work/` | generated intermediates (git-ignored) |

### Preprocessing steps

| Script | Output |
|---|---|
| `make_new_data.py` | `new_data/<m>_with_gt.csv`: 5 FPS Sportec data with ATD-like synthetic ball-missing masks |
| `make_inputs.py` (new) | `work/inputs/<m>_input.csv`: 5 FPS player positions (0–105 / 0–68 m) from the Sportec position XML, on the `new_data` frames |
| `make_ball5.py` (new) | `work/ball5/<m>_ball5.csv`: ball x, y, z, BallPossession and BallStatus from the Sportec position XML, every 5th frame of each half |
| `step1_base_npz.py` | `work/base/<m>.npz`: positions, ball, original labels |
| `step2_tree_pD_lomo.py` | `work/tree_prob.npz`: leave-one-match-out tree D probability (`tree_pD`) |
| `relabel_controlled_possession.py` | `work/labels/<m>_labels.csv`: event-based relabelling (`poss_orig`, `poss_loose`, `poss_mid`, `poss_strict`) |
| `build_npz.py` (new) | `work/dataset/<m>.npz` with the keys `xy, teams, frame, period, gt_orig, ball, nd_mask, tree_pD, gt_loose, gt_mid` |

`nd_mask` is `ball_masked` from `new_data/<m>_with_gt.csv` (`build_npz.py` reads it directly). `train_e4ha.py` samples ball-gap lengths from it.

## Data setup

Neither folder is committed (both are in `.gitignore`). Paths are relative to the repo root; override with env vars `SPORTEC_RAW` / `NEW_DATA`.

### `sportec_raw/` (Sportec / IDSSE open data)
- Source: Bassek, M., Rein, R., Weber, H. & Memmert, D. "An integrated dataset of spatiotemporal and event data in elite soccer", *Scientific Data* 12, 195 (2025), https://doi.org/10.1038/s41597-025-04505-y (CC-BY 4.0)
- Download: figshare, https://doi.org/10.6084/m9.figshare.28196177 (each match has a match information, an events and a position XML file)
- Expected layout: `sportec_raw/DFL_04_03_positions_raw_observed_*.xml`, `sportec_raw/event/`, `sportec_raw/metadata/`
- Steps:
  1. Download all files of the dataset from the figshare page above and unzip them if needed.
  2. Put the position files (`DFL_04_03_positions_raw_observed_*_DFL-MAT-<match>.xml`) directly in `sportec_raw/`.
  3. Put the event files in `sportec_raw/event/` and the match information files in `sportec_raw/metadata/`.


### `new_data/` (`<match>_with_gt.csv`, 5 FPS, one file per Sportec match)

- Source: Generated from the Sportec XML files using `preprocess/make_new_data.py`.
- Generation: Run `python preprocess/make_new_data.py` from the repository root.
- Method: ATD-like contiguous ball-missing augmentation at 5 FPS, targeting a 10.6% missing rate within Alive frames.
- GT reference: All 7 Sportec matches, including the validation match J03WPY, use GT Alive labels for mask generation.
- Outputs: `new_data/<match>_with_gt.csv`, `work/masks/<match>_nd_mask.npy`, and `work/masks/<match>_report.json`.
- Used for: The 5 FPS frame list and ball columns (`make_inputs.py`) and `ball_masked` = `nd_mask` (`build_npz.py`).


## How to run

All commands run from this repository's root unless noted. `<comp>` is the competition repository (it has `games.py`, `data/` and `evaluation/`).

**1. Install**
```
pip install -r requirements.txt
```


**2. Preprocess (Sportec only)**

Input: `sportec_raw/` (see Data setup).

First, generate `new_data/` from the Sportec XML files.
This generates 5 FPS data with synthetic ball-missing masks for all 7 Sportec matches, using GT Alive labels.

```bash
python preprocess/make_new_data.py
```

Then run the existing preprocessing pipeline.

```bash
cd preprocess
python make_inputs.py
python make_ball5.py
python step1_base_npz.py
python step2_tree_pD_lomo.py
python relabel_controlled_possession.py --raw ../sportec_raw --out ../work/labels
python build_npz.py
cd ..
```

The generated training datasets are saved to `work/dataset/`.


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

**5. Predict (run in `<comp>`)**
```
python <this repo>/run_predict.py --repo . --tree deadball_out/deadball_frames.csv --out YSAL.csv
```
- `--data-dir` sets the data folder (the default comes from `games.asset_paths`).
- No training is needed, and it runs on CPU in about 20 s.

**6. Validate**
```
python evaluation/validate_submission.py YSAL.csv
```

## Checks done on this repository

Folder and file names such as `recovery_bench/`, `recovery_bench_check/`, `gk_test/`, `apply_combos.py`, `combo_submissions/` and `kaggle/combo_experiment.ipynb` refer to earlier internal working folders; they are not part of this repository.

- `make_inputs.py` gives the same player columns as the earlier `recovery_bench/inputs` for all 7 matches (max difference 3e-14 m), so `step1_base_npz.py` output is unchanged. The ball columns come from `new_data` and differ from that benchmark's own ball mask; step1 does not read them.
- `make_ball5.py` gives the same values as the earlier `gk_test/*_ball5.csv` for all 7 matches.
- `run_predict.py` gives the same output as the earlier submission CSV built in the internal working folder: 0 of 199,319 frames differ.
- `deadball/predict_atd.py` reproduces the earlier `deadball_frames.csv` exactly.
- `train_v1.py` and `train_e4ha.py` each ran for 1 epoch without errors.

## Sources of copied code

Paths in parentheses such as `recovery_bench_check/` and `apply_combos.py` are earlier internal working folders, not included here.

| Path | Source |
|---|---|
| `third_party/pa_sm/` | V1 repository `barcelona-possession_ad` (team member's repo), package `pa_sm/`. Unmodified |
| `third_party/models_new.py` | `mtah-e4` repository (`models_new.py`: Set Attention / PMA frame encoder). Unmodified |
| `preprocess/step1_*.py`, `step2_*.py`, `features.py`, `config.py`, `common.py` | earlier data-build code (`recovery_bench_check/data_build/`). Only paths were changed, and step2's evaluation table was dropped |
| `train/train_*.py` | `data_build/train_and_eval_combo.py` (= `kaggle/combo_experiment.ipynb`). The `mid`-label parts only, with the same computation |
| `run_predict.py` | `recovery_bench_check/apply_combos.py`. Only the final-model branch was kept |

## Data policy
Competition data (`data/`, `deadball_out/`, submission CSVs, anything BAR-ATH) is not stored in this repository, and `.gitignore` blocks it.
