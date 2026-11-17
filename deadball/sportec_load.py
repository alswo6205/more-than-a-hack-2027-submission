"""
Sportec 공개 데이터 7경기를 내려받아 raw/{경기}.npz 로 저장 (학습용, 공개 데이터).
  pip install kloppy
  python sportec_load.py              # 7경기 전부 (경기당 몇 분)
저장 내용: X (프레임, 선수, 2) 미터·가운데 원점 / team / alive(경기 중) / period
"""
import os as _os; _os.chdir(_os.path.dirname(_os.path.abspath(__file__)))  # 어디서 실행해도 deadball 폴더 기준
import os, sys, time, glob, numpy as np
from kloppy import sportec
from config import SPORTEC_MATCHES

# 로컬 Sportec 원본 폴더 (positions xml + metadata/). 환경변수 SPORTEC_RAW 로 바꿀 수 있음
SRC = os.environ.get("SPORTEC_RAW", os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "sportec_raw"))

def _one(pattern):
    hits = glob.glob(os.path.join(SRC, pattern))
    if len(hits) != 1:
        raise FileNotFoundError(f"{pattern}: {len(hits)}개 발견 ({SRC})")
    return hits[0]

os.makedirs("raw", exist_ok=True)
for mid in (sys.argv[1:] or SPORTEC_MATCHES):
    out = f"raw/{mid}.npz"
    if os.path.exists(out):
        print("있음", out); continue
    t = time.time()
    meta = _one(f"metadata/DFL_02_01_matchinformation_*_DFL-MAT-{mid}.xml")
    pos = _one(f"DFL_04_03_positions_raw_observed_*_DFL-MAT-{mid}.xml")
    ds = sportec.load_tracking(meta_data=meta, raw_data=pos, only_alive=False, coordinates="sportec")
    home, away = ds.metadata.teams
    P = list(home.players) + list(away.players)
    team = np.array([0] * len(home.players) + [1] * len(away.players))
    col = {p.player_id: i for i, p in enumerate(P)}
    n = len(ds.frames)
    X = np.full((n, len(P), 2), np.nan, np.float32)
    alive = np.zeros(n, bool); period = np.zeros(n, np.int8)
    for k, f in enumerate(ds.frames):
        alive[k] = f.ball_state is not None and f.ball_state.value == "alive"
        period[k] = f.period.id
        for p, pdat in f.players_data.items():
            if pdat.coordinates is not None:
                X[k, col[p.player_id]] = (pdat.coordinates.x, pdat.coordinates.y)
    np.savez_compressed(out, X=X, team=team, alive=alive, period=period)
    print(f"{mid}: {n} 프레임, 경기 중 비율 {alive.mean():.2f}, {time.time()-t:.0f}초", flush=True)
