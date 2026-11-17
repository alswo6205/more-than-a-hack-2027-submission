"""
학습된 모델을 대회 데이터(ATD)에 적용해서 프레임마다 정지 확률과 D 구간을 뽑습니다.
대회 저장소 루트(games.py, starter/ 가 있는 곳)에서 실행하세요.

  python predict_atd.py --model 모델폴더 [--gaps gaps_over_3s.csv] [--out 출력폴더]

  --model  train.py final 이 만든 폴더 (model.joblib, meta.json)
  --gaps   (선택) 긴 공 결측 CSV. start_frame, end_frame 컬럼(video frame). 결측마다 D 비율을 붙여 줌
  --out    출력 폴더 (기본 deadball_out)

출력
  deadball_frames.csv      video_frame, half, graded, p_dead, dead  (모든 프레임)
  deadball_intervals.csv   D 구간 목록 (start_frame, end_frame, length_frames, source)
  gaps_with_dead.csv       (--gaps 줬을 때) 결측마다 D 비율과 결측 안 D 구간
화면에는 요약 숫자만 출력합니다.

처리 순서 (학습 때와 같은 코드를 씀)
  ① ATD 0~1 좌표 → 미터·가운데 원점   ② 전처리(순간이동 끊기, 조각 안에서만 속도)
  ③ 특징 347개   ④ 경기마다 표준화(전반 + 후반 킥오프 이후 프레임 기준)
  ⑤ 모델 확률   ⑥ 학습 때 고른 후처리
  ⑦ 후반 킥오프 전(110808~113741)은 무조건 D, 채점 안 하는 구간은 X(=dead 비움)
"""
import os, sys, json, argparse
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from config import FPS, HALF1, HALF2, HALF2_KICKOFF, HALF_SPLIT
from preprocess import atd_to_meters, speed_acc
from features import build, standardize
from evalx import post


def load_atd_arrays():
    """대회 저장소 starter/load_atd.py 로 ATD를 읽어 (X 0~1, team, video_frame) 반환."""
    sys.path.insert(0, os.path.join(os.getcwd(), "starter"))
    from load_atd import load
    ds = load()
    pid_team = {p.player_id: ti for ti, t in enumerate(ds.metadata.teams) for p in t.players}
    df = ds.to_df()
    pids = [c[:-2] for c in df.columns if c.endswith("_x") and c[:-2] in pid_team]
    X = np.stack([np.stack([df[f"{p}_x"].to_numpy(float), df[f"{p}_y"].to_numpy(float)], 1) for p in pids], 1)
    team = np.array([pid_team[p] for p in pids])
    vf = df["frame_id"].to_numpy() - 1          # video frame = frame_id - 1 (load_atd.py)
    return X, team, vf


def intervals(mask, vf):
    """True 가 이어진 구간 → [(시작 video frame, 끝 video frame)]"""
    e = np.diff(np.r_[0, mask.astype(int), 0])
    s, t = np.where(e == 1)[0], np.where(e == -1)[0] - 1
    return [(int(vf[a]), int(vf[b])) for a, b in zip(s, t)]


def run(X01, team, vf, model_dir, out_dir, gaps_csv=None):
    import joblib
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    meta = json.load(open(os.path.join(model_dir, "meta.json"), encoding="utf-8"))

    # ① 좌표 변환
    x, y = atd_to_meters(X01[..., 0], X01[..., 1])
    X = np.stack([x, y], -1)
    period = np.where(vf < HALF_SPLIT, 1, 2)
    # ② 전처리
    spd, acc, jump = speed_acc(X)
    # ③ 특징
    F, idx, names = build(X, team, period, spd, acc)
    if names != meta["names"]:
        raise RuntimeError("특징 이름이 학습 때와 다릅니다. 학습과 같은 버전의 코드를 쓰세요.")
    v5 = vf[idx]
    graded5 = ((v5 >= HALF1[0]) & (v5 <= HALF1[1])) | ((v5 >= HALF2[0]) & (v5 <= HALF2[1]))
    ref5 = ((v5 >= HALF1[0]) & (v5 <= HALF1[1])) | ((v5 >= HALF2_KICKOFF) & (v5 <= HALF2[1]))
    # ④ 표준화 (학습 때 켰으면)
    if meta["standardize"]:
        F = standardize(F, ref5)
    # ⑤ 확률 (채점 구간만 의미 있음)
    p5 = clf.predict_proba(F)[:, 1]
    # ⑥ 후처리: 채점 구간 안에서, 하프별로
    per5 = period[idx]
    d5 = np.zeros(len(p5), bool)
    g = np.where(graded5)[0]
    d5[g] = post(p5[g], per5[g], meta["post_kind"], meta["post_param"], meta["post_thr"])
    # 5 Hz → 25 fps (가장 가까운 앞 특징 프레임 값)
    pos = np.clip(np.searchsorted(idx, np.arange(len(vf)), side="right") - 1, 0, len(idx) - 1)
    p_all, d_all = p5[pos], d5[pos]
    graded = ((vf >= HALF1[0]) & (vf <= HALF1[1])) | ((vf >= HALF2[0]) & (vf <= HALF2[1]))
    pre_ko = (vf >= HALF2[0]) & (vf < HALF2_KICKOFF)
    # ⑦ 고정 규칙
    d_all = d_all & graded
    d_all[pre_ko] = True

    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame({"video_frame": vf, "half": period, "graded": graded.astype(int),
                  "p_dead": np.round(p_all, 4),
                  "dead": np.where(graded, d_all.astype(int), -1)}).to_csv(
        os.path.join(out_dir, "deadball_frames.csv"), index=False)   # dead: 1=D, 0=경기 중, -1=채점 안 함
    rows = [(a, b, b - a + 1, "pre_kickoff" if a >= HALF2[0] and b < HALF2_KICKOFF else "model")
            for a, b in intervals(d_all, vf)]
    iv = pd.DataFrame(rows, columns=["start_frame", "end_frame", "length_frames", "source"])
    iv.to_csv(os.path.join(out_dir, "deadball_intervals.csv"), index=False)

    print(f"채점 프레임 {graded.sum()}개 중 D {d_all.sum()}개 ({d_all.sum()/graded.sum()*100:.1f}%)")
    print(f"D 구간 {len(iv)}개, 길이 중앙값 {iv.length_frames.median()/FPS:.1f}초")
    print(f"순간이동으로 끊은 지점 {int(jump.sum())}개")

    if gaps_csv:
        gp = pd.read_csv(gaps_csv)
        dead_by_vf = pd.Series(d_all, index=vf)
        frac, inside = [], []
        for a, b in zip(gp["start_frame"], gp["end_frame"]):
            seg = dead_by_vf.loc[a:b]
            frac.append(round(float(seg.mean()), 3) if len(seg) else np.nan)
            m = seg.to_numpy()
            inside.append(";".join(f"{s}-{e}" for s, e in intervals(m, seg.index.to_numpy())))
        gp["dead_frac"] = frac; gp["dead_intervals"] = inside
        gp.to_csv(os.path.join(out_dir, "gaps_with_dead.csv"), index=False)
        print(f"긴 결측 {len(gp)}개 중 D가 절반 넘는 결측 {(gp.dead_frac > 0.5).sum()}개, "
              f"D가 하나도 없는 결측 {(gp.dead_frac == 0).sum()}개")
    print("저장:", out_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(HERE, "model"))
    _g = os.path.join(os.path.dirname(HERE), "data", "tactical", "gaps_over_3s.csv")
    ap.add_argument("--gaps", default=_g if os.path.exists(_g) else None)   # 기본: 저장소 data/tactical/gaps_over_3s.csv
    ap.add_argument("--out", default="deadball_out")
    a = ap.parse_args()
    if a.gaps and not os.path.exists(a.gaps):   # 잘못된 경로(예: README의 '결측CSV경로')면 기본 파일로 대체
        print(f"경고: --gaps 파일 없음 ({a.gaps}) → {_g if os.path.exists(_g) else '생략'}")
        a.gaps = _g if os.path.exists(_g) else None
    X01, team, vf = load_atd_arrays()
    run(X01, team, vf, a.model, a.out, a.gaps)
