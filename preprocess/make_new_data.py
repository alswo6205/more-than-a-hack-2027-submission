"""Create 5-FPS new_data/<match>_with_gt.csv from Sportec XML.

Uses the original, unmodified ball_missing.mask_generator.generate_mask
implementation supplied by the project author. ALL 7 matches (including
J03WPY validation) use their own Sportec GT Alive labels as the mask reference.

From the cloned repository root:
    python preprocess/make_new_data.py           # all seven matches
    python preprocess/make_new_data.py J03WPY    # single match

Input: sportec_raw/DFL_04_03_positions_raw_observed_*_DFL-MAT-<match>.xml
Output: new_data/<match>_with_gt.csv (5 FPS), work/masks/<match>_nd_mask.npy,
        work/masks/<match>_report.json

No work/base NPZ is needed: GT Alive and original ball visibility are read
straight from the same Sportec XML consumed by the existing make_ball5.py.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ball_missing.mask_generator import generate_mask
from make_ball5 import read_ball
from paths import MATCHES, NEW_DATA, SPORTEC_RAW, WORK


def make_one(match: str, seed: int) -> None:
    files = glob.glob(str(Path(SPORTEC_RAW) / f"DFL_04_03_positions_raw_observed_*_DFL-MAT-{match}.xml"))
    if len(files) != 1:
        raise FileNotFoundError(f"{match}: expected one Sportec position XML in {SPORTEC_RAW}, found {len(files)}")

    ball = read_ball(files[0])
    # Exactly the same sampling as the existing preprocess/make_ball5.py.
    first = ball.groupby("period").frame_id.transform("min")
    ball = ball.loc[(ball.frame_id - first) % 5 == 0].copy().reset_index(drop=True)
    if ball.empty or set(ball.period.unique()) != {1, 2}:
        raise ValueError(f"{match}: both halves must be present after 5-FPS sampling")

    frames = ball.frame_id.to_numpy(dtype=np.int64)
    period = ball.period.to_numpy(dtype=np.int64)
    gt_alive = ball.status.to_numpy(dtype=np.int64) != 0   # gt_orig != 2
    # Equates to np.isfinite(z['ball'][ix]).all(axis=1) in the original script.
    observed = np.isfinite(ball[["x", "y", "z"]].to_numpy(dtype=float)).all(axis=1)

    nd_mask = np.zeros(len(ball), dtype=bool)
    reports = {}
    for p in (1, 2):
        ix = np.flatnonzero(period == p)
        syn, details = generate_mask(
            gt_alive[ix], observed[ix], frames[ix],
            seed=seed + 101 * p, trials=12,
        )
        nd_mask[ix] = syn
        reports[f"p{p}"] = details

    # Preserve the original mask diagnostics and boolean mask separately.
    masks_dir = Path(WORK) / "masks"
    masks_dir.mkdir(exist_ok=True, parents=True)
    np.save(masks_dir / f"{match}_nd_mask.npy", nd_mask)
    (masks_dir / f"{match}_report.json").write_text(
        json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # The existing preprocess/make_inputs.py requires the HEAD columns below.
    # Convert Sportec centre-origin ball coordinates to pitch-corner metres,
    # matching step1_base_npz.py (its ball array contains unmasked raw values).
    bx = ball.x.to_numpy(dtype=float) + 52.5
    by = 34.0 - ball.y.to_numpy(dtype=float)
    bx[nd_mask] = np.nan
    by[nd_mask] = np.nan
    half_first = ball.groupby("period").frame_id.transform("min").to_numpy(dtype=np.int64)
    poss = ball.poss.to_numpy(dtype=np.int64)
    status = ball.status.to_numpy(dtype=np.int64)
    gt_label = np.where(status == 0, "D", np.where(poss == 1, "H", "A"))

    output = pd.DataFrame({
        "frame_id": frames,
        "period": period,
        "time_s": (frames - half_first) / 25.0,
        "ball_x": bx,
        "ball_y": by,
        "ball_masked": nd_mask.astype(np.uint8),
        "gt_label": gt_label,
        "alive": gt_alive.astype(np.uint8),
        "ball_possession": poss,
        "ball_status": status,
    })
    out_dir = Path(NEW_DATA)
    out_dir.mkdir(exist_ok=True, parents=True)
    file = out_dir / f"{match}_with_gt.csv"
    output.to_csv(file, index=False)
    print(f"[new_data] {match}: {len(output)} frames, "
          f"synthetic={nd_mask.mean():.2%}, GT Alive used -> {file}", flush=True)


def main() -> None:
    requested = sys.argv[1:] or MATCHES
    unknown = sorted(set(requested) - set(MATCHES))
    if unknown:
        raise ValueError(f"Unknown match(es): {unknown}. Allowed: {MATCHES}")
    train = [m for m in MATCHES if m != "J03WPY"]
    # Exact seeds from the user's previous create() implementation.
    seeds = {m: 42 + 503 * i for i, m in enumerate(train)}
    seeds["J03WPY"] = 9921
    for m in requested:
        make_one(m, seed=seeds[m])


if __name__ == "__main__":
    main()
