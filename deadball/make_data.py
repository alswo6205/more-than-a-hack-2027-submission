"""
Sportec 경기마다 특징 파일 만들기 (깨끗 / 흉내 h1 / 흉내 h2).
  python make_data.py              # 7경기 전부
결과: data/{경기}_{clean|h1|h2}.npz  (F: 표준화 전 특징, y: 정지=1, period, names)
필요: raw/{경기}.npz  (sportec_load.py로 만듦)
"""
import os as _os; _os.chdir(_os.path.dirname(_os.path.abspath(__file__)))  # 어디서 실행해도 deadball 폴더 기준
import os, sys, numpy as np
from config import SPORTEC_MATCHES
from preprocess import speed_acc
from features import build
from noise_sim import simulate

os.makedirs("data", exist_ok=True)
matches = sys.argv[1:] or SPORTEC_MATCHES
for i, m in enumerate(matches):
    d = np.load(f"raw/{m}.npz")
    X, team, per, alive = d["X"].astype(float), d["team"], d["period"], d["alive"]
    for cond in ("clean", "h1", "h2"):
        out = f"data/{m}_{cond}.npz"
        if os.path.exists(out):
            continue
        Fs, ys, ps = [], [], []
        for p in np.unique(per):                       # 하프마다 따로 (메모리 절약, 문맥도 하프 안에서만)
            sel = np.where(per == p)[0]
            Xp = X[sel]
            if cond == "clean":
                X2, t2 = Xp, team
            else:
                seed = 1000 * i + 10 * int(p) + (1 if cond == "h1" else 2)
                X2, t2 = simulate(Xp, team, cond, seed=seed)
            spd, acc, _ = speed_acc(X2)
            F, idx, names = build(X2, t2, per[sel], spd, acc)
            Fs.append(F); ys.append(~alive[sel][idx]); ps.append(per[sel][idx])
            del X2, spd, acc
        F, y, pp = np.vstack(Fs), np.concatenate(ys), np.concatenate(ps)
        np.savez_compressed(out, F=F, y=y.astype(np.int8), period=pp, names=np.array(names))
        print(m, cond, F.shape, flush=True)
