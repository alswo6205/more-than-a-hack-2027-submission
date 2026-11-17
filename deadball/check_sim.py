"""잡음 흉내가 대회 측정값과 맞는지 확인: python check_sim.py [경기ID]"""
import os as _os; _os.chdir(_os.path.dirname(_os.path.abspath(__file__)))  # 어디서 실행해도 deadball 폴더 기준
import sys, numpy as np
from noise_sim import simulate
from noise_stats import compute_stats
m = sys.argv[1] if len(sys.argv) > 1 else "J03WR9"
d = np.load(f"raw/{m}.npz"); X = d["X"].astype(float); team = d["team"]; per = d["period"]
for prof in ("h1", "h2"):
    Xh = X[per == 1]
    X2, t2 = simulate(Xh, team, prof, seed=1)
    compute_stats(X2, t2, 25, label=f"흉내 {prof} ({m} 전반)")
