"""Sportec position XML -> work/inputs/<match>_input.csv (5 FPS player positions; same format and player values as the old recovery_bench/inputs).

Players: kloppy metrica coordinates x 105 / x 68 (origin = pitch corner), slots in metadata player order
(home_<i> = i-th home player, away_<i> = i-th away player; empty when the player is not on the pitch).
Frames, period, time_s and the ball columns are taken from NEW_DATA/<match>_with_gt.csv.
  python preprocess/make_inputs.py [match ...]
"""
import glob, os, sys
import numpy as np, pandas as pd
from kloppy import sportec
from paths import SPORTEC_RAW, NEW_DATA, BENCH_INPUTS, MATCHES

HEAD = ["frame_id", "period", "time_s", "ball_x", "ball_y", "ball_masked"]


def one(pattern):
    hit = glob.glob(os.path.join(SPORTEC_RAW, pattern))
    if len(hit) != 1:
        raise FileNotFoundError(f"{pattern}: {len(hit)} files in {SPORTEC_RAW}")
    return hit[0]


def main():
    os.makedirs(BENCH_INPUTS, exist_ok=True)
    for m in (sys.argv[1:] or MATCHES):
        nd = pd.read_csv(os.path.join(NEW_DATA, f"{m}_with_gt.csv"), usecols=HEAD)
        ds = sportec.load_tracking(meta_data=one(f"metadata/DFL_02_01_matchinformation_*_DFL-MAT-{m}.xml"),
                                   raw_data=one(f"DFL_04_03_positions_raw_observed_*_DFL-MAT-{m}.xml"),
                                   only_alive=False, coordinates="metrica")
        home, away = ds.metadata.teams
        slot = {p.player_id: f"home_{i}" for i, p in enumerate(home.players)}
        slot.update({p.player_id: f"away_{i}" for i, p in enumerate(away.players)})
        names = list(slot.values()); col = {n: k for k, n in enumerate(names)}
        row = {fid: k for k, fid in enumerate(nd.frame_id.to_numpy())}
        X = np.full((len(nd), len(names), 2), np.nan)
        for f in ds.frames:
            k = row.get(f.frame_id)
            if k is None:
                continue
            for p, d in f.players_data.items():
                if d.coordinates is not None:
                    X[k, col[slot[p.player_id]]] = (d.coordinates.x * 105, d.coordinates.y * 68)
        out = nd.copy()
        for n, j in col.items():
            out[n + "_x"] = X[:, j, 0]; out[n + "_y"] = X[:, j, 1]
        out.to_csv(os.path.join(BENCH_INPUTS, f"{m}_input.csv"), index=False)
        print(m, len(out), "frames", len(home.players), "home", len(away.players), "away", flush=True)


if __name__ == "__main__":
    main()
