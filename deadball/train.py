"""
학습과 평가.

  python train.py eval      # 경기 하나씩 빼며 7번 테스트 → 성능 표 출력 (results_eval.csv)
  python train.py final     # 최종 모델 학습 → model/model.joblib, model/meta.json

옵션 (환경변수)
  TRAIN_CONDS=h1,h2      학습에 쓸 데이터 조건 (clean / h1 / h2 중에서, 기본 h1,h2 = 잡음 흉내)
  TEST_CONDS=h1,h2       eval에서 평가할 조건
  STANDARDIZE=1          경기마다 표준화 (1=켬, 0=끔)
  MATCHES=J03WMX,...     쓸 경기 목록 (기본 7경기)

순서
 1. data/{경기}_{조건}.npz 읽기 (make_data.py 결과)
 2. 경기마다 표준화 (켜져 있으면)
 3. HistGradientBoosting 학습 → 프레임마다 정지 확률
 4. 후처리(기준값·메디안 필터·Viterbi)는 검증 경기로만 고름
"""
import os as _os; _os.chdir(_os.path.dirname(_os.path.abspath(__file__)))  # 어디서 실행해도 deadball 폴더 기준
import os, sys, json, numpy as np, warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingClassifier
from config import SPORTEC_MATCHES
from features import standardize
from evalx import frame_scores, stop_scores, post, tune_post

TRAIN_CONDS = os.environ.get("TRAIN_CONDS", "h1,h2").split(",")
TEST_CONDS = os.environ.get("TEST_CONDS", "h1,h2").split(",")
STD = os.environ.get("STANDARDIZE", "1") == "1"
M = os.environ.get("MATCHES", ",".join(SPORTEC_MATCHES)).split(",")   # 쓸 경기 (기본 7경기)
VAL = os.environ.get("VAL", "J03WQQ" if "J03WQQ" in M else M[-1])   # final에서 후처리를 고를 검증 경기


def load(m, cond):
    d = np.load(f"data/{m}_{cond}.npz")
    F = standardize(d["F"], np.ones(len(d["F"]), bool)) if STD else d["F"]
    return F, d["y"].astype(bool), d["period"], list(d["names"])


def model():
    return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, l2_regularization=1.0, random_state=0)


def fit(matches):
    X, y = [], []
    for m in matches:
        for c in TRAIN_CONDS:
            F, yy, _, _ = load(m, c); X.append(F); y.append(yy)
    return model().fit(np.vstack(X), np.concatenate(y))


def evaluate():
    rows = []
    for i, te in enumerate(M):
        va = M[(i + 1) % len(M)]
        tr = [m for m in M if m not in (te, va)]
        clf = fit(tr)
        Fv, yv, pv, _ = load(va, TEST_CONDS[0])
        kind, prm, thr = tune_post(clf.predict_proba(Fv)[:, 1], yv, pv)
        for c in TEST_CONDS:
            Ft, yt, pt, _ = load(te, c)
            pd = post(clf.predict_proba(Ft)[:, 1], pt, kind, prm, thr)
            r = {"test": te, "cond": c, **frame_scores(yt, pd), **stop_scores(yt, pd, pt), "post": f"{kind}:{prm}:{thr}"}
            rows.append(r)
            print(f"{te} {c}: 프레임 F1 {r['f_f1']:.3f}  정지 구간 F1 {r['s_f1']:.3f}  "
                  f"시작/끝 ±2초 {r['start_within2']:.2f}/{r['end_within2']:.2f}  후처리 {r['post']}", flush=True)
    import pandas as pd_
    df = pd_.DataFrame(rows); df.to_csv("results_eval.csv", index=False)
    print("\n=== 평균 (학습 %s, 표준화 %s) ===" % (TRAIN_CONDS, STD))
    print(df.groupby("cond")[["f_f1", "s_f1", "start_within2", "end_within2",
                              "rec_5-10s", "rec_10-30s", "rec_30-infs"]].agg(["mean", "std"]).round(3).to_string())


def final():
    os.makedirs("model", exist_ok=True)
    # 1) 검증 경기를 뺀 6경기로 학습 → 검증 경기로 후처리 고르기
    tr = [m for m in M if m != VAL]
    clf = fit(tr)
    Fv, yv, pv, names = load(VAL, TEST_CONDS[0])
    kind, prm, thr = tune_post(clf.predict_proba(Fv)[:, 1], yv, pv)
    print("고른 후처리:", kind, prm, thr)
    # 2) 7경기 전부로 다시 학습해서 저장
    clf = fit(M)
    import joblib
    joblib.dump(clf, "model/model.joblib")
    json.dump({"names": names, "post_kind": kind, "post_param": prm, "post_thr": thr,
               "standardize": STD, "train_conds": TRAIN_CONDS}, open("model/meta.json", "w"), ensure_ascii=False, indent=1)
    print("저장: model/model.joblib, model/meta.json")


if __name__ == "__main__":
    {"eval": evaluate, "final": final}[sys.argv[1] if len(sys.argv) > 1 else "eval"]()
