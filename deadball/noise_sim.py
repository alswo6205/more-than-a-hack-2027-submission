"""
잡음 흉내 (Sportec 학습 데이터에만). 대회 ATD 측정값(noise_stats.py)에 맞춘 규칙 기반 무작위.

넣는 잡음
 1. 개별 결측    선수 한 명이 잠깐 사라짐. 절반은 같은 슬롯으로, 절반은 새 슬롯으로 다시 나타남
 2. 동시 결측    경기장 한 구역(길이 방향 띠)의 선수 여러 명이 같이 사라짐 (카메라 밖 가정)
 3. 슬롯 바뀜    가까이 있는 두 선수의 슬롯이 그 순간부터 서로 바뀜 (같은 팀/상대 팀 모두)
                 → 작은 순간이동 + 상대 팀끼리면 팀 표시가 틀어짐
 4. 먼 순간이동  슬롯 하나가 잠깐 몇 m 떨어진 곳으로 튐 (오검출)
 5. 중복 검출    선수 한 명을 옆에 한 번 더 그림 (같은 팀)
 좌표 떨림은 넣지 않음 (대회 데이터 떨림 = Sportec과 같음)

PROFILES: 대회 1하프 모양(h1), 2하프 모양(h2). 학습엔 둘 다 씀.
"""
import numpy as np
from config import FPS

PROFILES = {
    # *_rate: 분당 사건 수 / *_med: 길이 중앙값(초) / ind_same: 같은 슬롯으로 돌아올 확률
    # flip: 새 슬롯이 처음 flip_med초 동안 팀을 잘못 달 확률, dup_flip: 중복 슬롯이 팀을 잘못 달 확률 / swap_dmax: 슬롯이 바뀌는 최대 거리(m)
    "h1": dict(ind_rate=2.5, ind_same=0.8, ind_med=6.0, ind_long=0.25,
               grp_rate=0.9, grp_med=6.0, grp_band=(15, 35),
               swap_rate=4.0, swap_dmax=1.0, far_rate=0.0, dup_rate=0.4, dup_med=5.0, dup_flip=0.3, flip=0.35, flip_med=10.0),
    "h2": dict(ind_rate=2.8, ind_same=0.8, ind_med=6.0, ind_long=0.25,
               grp_rate=0.7, grp_med=5.0, grp_band=(15, 30),
               swap_rate=13.0, swap_dmax=1.0, far_rate=0.15, dup_rate=1.1, dup_med=6.0, dup_flip=0.3, flip=0.5, flip_med=10.0),
}


def _dur(rng, med, long_frac):
    """결측 길이(초): 대부분 로그정규(중앙값 med), 일부는 30~200초의 긴 결측."""
    if rng.random() < long_frac:
        return rng.uniform(30, 150)
    return float(np.exp(rng.normal(np.log(med), 0.8)))


def simulate(X, team, profile="h1", seed=0, fps=FPS):
    """X: (n, S, 2) 깨끗한 좌표. 반환: 잡음 넣은 X2 (슬롯 수가 늘 수 있음), team2."""
    p = PROFILES[profile]
    rng = np.random.default_rng(seed)
    n = len(X)
    minutes = n / fps / 60
    cols = [X[:, s].copy() for s in range(X.shape[1])]
    tms = list(team)

    def on_pitch(t):
        return [s for s in range(len(cols)) if not np.isnan(cols[s][t, 0])]

    # 3. 슬롯 바뀜: 두 선수가 아주 가까이 붙은 순간(0.4~1.2 m)을 찾아 그 순간부터 미래를 맞바꿈
    n_swap = rng.poisson(p["swap_rate"] * minutes)
    tries = 0
    while n_swap > 0 and tries < n_swap * 60:
        tries += 1
        t = int(rng.integers(1, n))
        act = [s for s in range(X.shape[1]) if not np.isnan(cols[s][t, 0])]
        if len(act) < 2:
            continue
        P = np.array([cols[s][t] for s in act])
        D = np.linalg.norm(P[:, None] - P[None], axis=2); D[np.diag_indices(len(act))] = np.inf
        i, j = np.unravel_index(np.argmin(D), D.shape)
        if not (0.4 <= D[i, j] <= p["swap_dmax"]):
            continue
        a, b = act[i], act[j]
        tmp = cols[a][t:].copy(); cols[a][t:] = cols[b][t:]; cols[b][t:] = tmp
        n_swap -= 1

    # 4. 먼 순간이동: 0.2~2초 동안 2~8 m 옆으로
    for _ in range(rng.poisson(p["far_rate"] * minutes)):
        t = rng.integers(0, n); act = on_pitch(t)
        if not act:
            continue
        s = rng.choice(act); ln = int(rng.uniform(0.2, 2) * fps)
        ang = rng.uniform(0, 2 * np.pi); r = rng.uniform(2, 8)
        cols[s][t:t + ln] += np.array([np.cos(ang), np.sin(ang)]) * r

    # 5. 중복 검출: 0.5~1.5 m 옆에 복제 (새 슬롯)
    for _ in range(rng.poisson(p["dup_rate"] * minutes)):
        t = rng.integers(0, n); act = on_pitch(t)
        if not act:
            continue
        s = rng.choice(act); ln = int(np.exp(rng.normal(np.log(p["dup_med"]), 0.7)) * fps)
        c = np.full((n, 2), np.nan)
        off = rng.normal(0, 1, 2); off = off / np.linalg.norm(off) * rng.uniform(0.5, 1.5)
        c[t:t + ln] = cols[s][t:t + ln] + off
        cols.append(c); tms.append(tms[s] if rng.random() > p["dup_flip"] else 1 - tms[s])

    def drop(s, t0, ln):
        """슬롯 s를 t0부터 ln 프레임 지움. ind_same 확률로 같은 슬롯, 아니면 새 슬롯으로 다시 나타남."""
        t1 = min(n, t0 + ln)
        cols[s][t0:t1] = np.nan
        if t1 < n and rng.random() > p["ind_same"]:
            c = np.full((n, 2), np.nan); c[t1:] = cols[s][t1:]
            cols[s][t1:] = np.nan
            if rng.random() < p["flip"]:
                # 팀을 잘못 붙인 채로 잠깐(중앙값 flip_med초) 추적하다가, 다시 바른 팀 새 슬롯으로 넘어감
                t2 = min(n, t1 + int(np.exp(rng.normal(np.log(p["flip_med"]), 0.7)) * fps))
                c2 = np.full((n, 2), np.nan); c2[t2:] = c[t2:]; c[t2:] = np.nan
                cols.append(c); tms.append(1 - tms[s])
                cols.append(c2); tms.append(tms[s])
            else:
                cols.append(c); tms.append(tms[s])

    # 1. 개별 결측
    for _ in range(rng.poisson(p["ind_rate"] * minutes)):
        t = rng.integers(0, n); act = on_pitch(t)
        if act:
            drop(rng.choice(act), t, int(_dur(rng, p["ind_med"], p["ind_long"]) * fps))

    # 2. 동시 결측: 길이 방향 띠(폭 15~30 m) 안 선수 전부
    for _ in range(rng.poisson(p["grp_rate"] * minutes)):
        t = rng.integers(0, n); act = on_pitch(t)
        if not act:
            continue
        bw = rng.uniform(*p["grp_band"]); x0 = rng.uniform(-52.5, 52.5 - bw)
        ln = int(np.exp(rng.normal(np.log(p["grp_med"]), 0.8)) * fps)
        for s in act:
            if x0 <= cols[s][t, 0] <= x0 + bw:
                drop(s, t, ln)

    X2 = np.stack(cols, 1)
    keep = ~np.all(np.isnan(X2[..., 0]), axis=0)          # 한 번도 안 쓰인 슬롯 제거
    return X2[:, keep], np.array(tms)[keep]
