"""Sportec position XML -> work/ball5/<match>_ball5.csv (ball x, y, z, BallPossession, BallStatus at 5 FPS).

Every 5th frame from the first frame of each half (same frames as new_data and work/inputs).
Coordinates are the raw Sportec values (metres, pitch centre = origin).
  python preprocess/make_ball5.py [match ...]
"""
import glob, os, re, sys
import pandas as pd
from paths import SPORTEC_RAW, BALL5, MATCHES

ATTR = re.compile(r'(\w+)="([^"]*)"')


def read_ball(path):
    rows, inball, per = [], False, 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if "<FrameSet" in line:
                inball = 'TeamId="BALL"' in line
                if inball:
                    per = 1 if "firstHalf" in line else 2
                continue
            if inball and "<Frame " in line:
                a = dict(ATTR.findall(line))
                rows.append((int(a["N"]), per, a["X"], a["Y"], a["Z"], int(a.get("BallPossession", 0)), int(a.get("BallStatus", 0))))
    df = pd.DataFrame(rows, columns=["frame_id", "period", "x", "y", "z", "poss", "status"])
    for c in ("x", "y", "z"):
        df[c] = df[c].astype(float)
    return df.sort_values("frame_id").reset_index(drop=True)


def main():
    os.makedirs(BALL5, exist_ok=True)
    for m in (sys.argv[1:] or MATCHES):
        hit = glob.glob(os.path.join(SPORTEC_RAW, f"DFL_04_03_positions_raw_observed_*_DFL-MAT-{m}.xml"))
        assert len(hit) == 1, (m, hit)
        df = read_ball(hit[0])
        first = df.groupby("period").frame_id.transform("min")
        out = df[(df.frame_id - first) % 5 == 0]
        out.to_csv(os.path.join(BALL5, f"{m}_ball5.csv"), index=False)
        print(m, len(out), "frames", flush=True)


if __name__ == "__main__":
    main()
