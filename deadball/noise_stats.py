"""
대회 ATD의 선수 트래킹 잡음 통계를 숫자로만 뽑는 스크립트.
- 출력은 요약 숫자(평균, 분위수, 비율)뿐. 프레임별 값은 저장하지 않음.
- 대회 저장소 루트에서 실행:  python noise_stats.py
- 필요: kloppy (requirements-starter.txt), numpy, pandas

같은 계산을 Sportec(깨끗한 데이터, 잡음 흉내 데이터)에도 해서 비교할 수 있게
compute_stats(X, team, fps) 함수 하나로 만들어 둠.
"""
import sys, os
import numpy as np

FPS = 25
PITCH = (105.3, 68.0)                      # games.py 실측값 (DATA.md)
HALVES = [(11185, 90651), (110808, 190197)]  # 채점 구간 (video frame)


def runs(mask):
    """True가 연속된 구간 길이 목록 (프레임)."""
    m = np.r_[False, mask, False].astype(int)
    d = np.diff(m)
    return np.where(d == -1)[0] - np.where(d == 1)[0]


def pct(a, qs=(10, 25, 50, 75, 90)):
    a = np.asarray(a, float)
    if len(a) == 0:
        return "없음"
    return " / ".join(f"{np.percentile(a, q):.2f}" for q in qs)


def compute_stats(X, team, fps=FPS, label=""):
    """
    X: (프레임 수, 트랙 칸 수, 2) 미터 좌표, 없으면 NaN
    team: (트랙 칸 수,) 0 또는 1
    """
    n, S, _ = X.shape
    vis = ~np.isnan(X[..., 0])
    minutes = n / fps / 60
    print(f"\n========== {label}  ({n} 프레임, {minutes:.1f}분, 트랙 칸 {S}개) ==========")

    # 1. 프레임마다 추적되는 사람 수
    cnt = vis.sum(1)
    c0, c1 = vis[:, team == 0].sum(1), vis[:, team == 1].sum(1)
    print("\n[1] 프레임당 추적 인원")
    print(f"  전체 평균 {cnt.mean():.2f}명, 분위수(10/25/50/75/90) {pct(cnt)}")
    print(f"  팀별 평균 {c0.mean():.2f} / {c1.mean():.2f}")
    print(f"  22명 초과 프레임 {np.mean(cnt > 22)*100:.2f}%, 한 팀 11명 초과 {np.mean((c0 > 11) | (c1 > 11))*100:.2f}%")
    print(f"  20명 미만 {np.mean(cnt < 20)*100:.2f}%, 18명 미만 {np.mean(cnt < 18)*100:.2f}%")
    print(f"  선수-프레임 결측률(22명 기준) {max(0, 1 - cnt.mean()/22)*100:.2f}%")

    # 2. 트랙 칸이 살아 있는 시간 / 끊긴 시간
    alive, gaps = [], []
    for s in range(S):
        r = runs(vis[:, s])
        alive += list(r)
        # 칸이 처음 나타난 뒤 ~ 마지막 사라지기 전 사이의 끊김만 셈
        idx = np.where(vis[:, s])[0]
        if len(idx) > 1:
            inner = ~vis[idx[0]:idx[-1] + 1, s]
            gaps += list(runs(inner))
    alive, gaps = np.array(alive) / fps, np.array(gaps) / fps
    print("\n[2] 트랙 칸 연속 생존 시간 (초)")
    print(f"  구간 수 {len(alive)} (분당 {len(alive)/minutes:.1f}개), 분위수 {pct(alive)}")
    print(f"  1초 미만 조각 비율 {np.mean(alive < 1)*100:.1f}%, 10초 미만 {np.mean(alive < 10)*100:.1f}%")
    print("  칸 안의 끊김 길이 (초)")
    print(f"  끊김 수 {len(gaps)} (분당 {len(gaps)/minutes:.1f}개), 분위수 {pct(gaps)}")

    # 3. 위치 순간이동 (신원 바뀜·오검출 의심): 한 프레임 사이 이동 거리
    step = np.linalg.norm(np.diff(X, axis=0), axis=2)          # (n-1, S), 미터
    step_ok = step[~np.isnan(step)]
    v = step_ok * fps                                           # m/s
    print("\n[3] 한 프레임 사이 이동 (같은 트랙 칸)")
    print(f"  속도로 환산한 분위수(50/90/99/99.9) {pct(v, (50, 90, 99, 99.9))} m/s")
    for thr in (12, 20, 50):
        k = np.sum(v > thr)
        print(f"  {thr} m/s 초과(사람이 못 내는 속도) {k}회, 분당 {k/minutes:.2f}회")

    # 4. 좌표 떨림: 1초 이동평균과의 차이 (고주파 잡음 크기)
    from scipy.ndimage import uniform_filter1d
    res = []
    for s in range(S):
        r = runs(vis[:, s]); st = 0
        idx = np.where(np.diff(np.r_[False, vis[:, s], False].astype(int)) == 1)[0]
        for a, ln in zip(idx, r):
            if ln >= 3 * fps:
                seg = X[a:a + ln, s]
                sm = uniform_filter1d(seg, fps, axis=0, mode="nearest")
                res.append(np.linalg.norm(seg - sm, axis=1)[fps:-fps])
    res = np.concatenate(res) if res else np.array([])
    print("\n[4] 좌표 떨림 (1초 이동평균과의 거리, m)")
    print(f"  분위수(50/90/99) {pct(res, (50, 90, 99))}")

    # 5. 가속도 크기 (모델이 가장 많이 쓴 특징의 재료)
    Xs = uniform_filter1d(np.nan_to_num(X), 5, axis=0); Xs[np.isnan(X)] = np.nan
    V = (Xs[10:] - Xs[:-10]) / 0.4
    sp = np.linalg.norm(V, axis=2)
    A = np.abs(sp[10:] - sp[:-10]) / 0.4
    a_ok = A[~np.isnan(A)]
    print("\n[5] 속도·가속도 (0.4초 차분, 특징 계산과 같은 방식)")
    sp_ok = sp[~np.isnan(sp)]
    print(f"  속도 분위수(50/90/99) {pct(sp_ok, (50, 90, 99))} m/s")
    print(f"  가속도 분위수(50/90/99) {pct(a_ok, (50, 90, 99))} m/s²")


def load_atd():
    """대회 저장소의 starter/load_atd.py 로 ATD를 불러와 채점 구간만 (X, team) 으로 변환."""
    sys.path.insert(0, os.path.join(os.getcwd(), "starter"))
    from load_atd import load
    ds = load()
    teams = ds.metadata.teams
    pid_team = {p.player_id: ti for ti, t in enumerate(teams) for p in t.players}
    df = ds.to_df()
    pids = [c[:-2] for c in df.columns if c.endswith("_x") and c[:-2] in pid_team]
    X = np.stack([np.stack([df[f"{p}_x"].to_numpy(float) * PITCH[0],
                            df[f"{p}_y"].to_numpy(float) * PITCH[1]], 1) for p in pids], 1)
    team = np.array([pid_team[p] for p in pids])
    vf = df["frame_id"].to_numpy() - 1                     # video frame = frame_id - 1
    keep = np.zeros(len(vf), bool)
    for a, b in HALVES:
        keep |= (vf >= a) & (vf <= b)
    return X, team, keep, vf


if __name__ == "__main__":
    X, team, keep, vf = load_atd()
    # 하프별로 따로 계산 (하프타임을 사이에 끼우지 않기 위해), 마지막에 전체 요약은 각 하프 출력을 보면 됨
    for h, (a, b) in enumerate(HALVES, 1):
        m = (vf >= a) & (vf <= b)
        compute_stats(X[m], team, FPS, label=f"대회 ATD {h}하프")
