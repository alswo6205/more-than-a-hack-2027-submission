"""
전처리: 대회 데이터와 Sportec 둘 다 이 코드를 똑같이 거칩니다.

입력  X: (프레임 수, 슬롯 수, 2) 미터 좌표, 가운데 원점. 없는 값은 NaN
출력  speed, acc: (프레임 수, 슬롯 수) m/s, m/s². 계산할 수 없는 곳은 NaN

하는 일
1. 순간이동 찾기: 같은 슬롯이 한 프레임 사이 JUMP_SPEED 넘게 움직이면 그 지점에서 슬롯을 끊음
   (그 슬롯만. 같은 프레임의 다른 슬롯은 건드리지 않음)
2. 조각 번호 매기기: 끊긴 지점이나 NaN이 나오면 새 조각
3. 속도·가속도는 같은 조각 안에서만 계산 → 순간이동·결측 경계에서 가짜 속도가 생기지 않음
위치(X)는 하나도 지우거나 바꾸지 않습니다. 조각을 다시 잇지도 않습니다.
"""
import numpy as np
from config import FPS, JUMP_SPEED, MAX_SPEED, VEL_HALF, SMOOTH, ATD_PITCH_L, ATD_PITCH_W


def atd_to_meters(x01, y01):
    """대회 ATD 0~1 좌표 → 가운데 원점 미터 (Sportec과 같은 좌표계)."""
    return x01 * ATD_PITCH_L - ATD_PITCH_L / 2, y01 * ATD_PITCH_W - ATD_PITCH_W / 2


def fragment_ids(X, jump_speed=JUMP_SPEED, fps=FPS):
    """슬롯마다 조각 번호. NaN 프레임은 -1. 순간이동 또는 결측 뒤에 새 번호."""
    valid = ~np.isnan(X[..., 0])
    step = np.zeros(valid.shape)
    step[1:] = np.linalg.norm(X[1:] - X[:-1], axis=2) * fps          # m/s, NaN이면 NaN
    jump = np.zeros(valid.shape, bool)
    jump[1:] = step[1:] > jump_speed                                  # 순간이동 (NaN 비교는 False)
    start = valid & (np.r_[[np.zeros(valid.shape[1], bool)], ~valid[:-1]] | jump)
    start[0] = valid[0]
    frag = np.cumsum(start, axis=0)                                   # 슬롯마다 증가하는 번호
    frag = np.where(valid, frag, -1)
    return frag, jump


def _shift(a, k, fill):
    """a[t+k] (범위 밖은 fill)."""
    out = np.full_like(a, fill)
    if k > 0:
        out[:-k] = a[k:]
    elif k < 0:
        out[-k:] = a[:k]
    else:
        out[:] = a
    return out


def _same_frag(frag, k1, k2):
    """t+k1 과 t+k2 가 같은 조각인가 (둘 다 유효해야 함). 조각은 연속이므로 양 끝이 같으면 사이도 같음."""
    a, b = _shift(frag, k1, -1), _shift(frag, k2, -2)
    return (a == b) & (a >= 0)


def speed_acc(X, fps=FPS):
    """조각 안에서만 속도·가속도 계산."""
    frag, jump = fragment_ids(X, fps=fps)
    h = SMOOTH // 2
    # 1) 위치 이동평균 (같은 조각 안에서만)
    ok_s = _same_frag(frag, -h, h)
    Xz = np.nan_to_num(X)
    cs = np.cumsum(np.concatenate([np.zeros((1,) + Xz.shape[1:]), Xz]), axis=0)
    n = len(X)
    lo = np.clip(np.arange(n) - h, 0, n); hi = np.clip(np.arange(n) + h + 1, 0, n)
    Xs = (cs[hi] - cs[lo]) / SMOOTH
    Xs[~ok_s] = np.nan
    # 2) 속도: (Xs[t+K] - Xs[t-K]) / (2K/fps), 같은 조각 안에서만
    K = VEL_HALF
    ok_v = _same_frag(frag, -K - h, K + h)
    V = (_shift(Xs, K, np.nan) - _shift(Xs, -K, np.nan)) / (2 * K / fps)
    spd = np.linalg.norm(V, axis=2)
    spd[~ok_v] = np.nan
    spd[spd > MAX_SPEED] = np.nan
    # 3) 가속도 크기: |spd[t+K] - spd[t-K]| / 0.4초 (양쪽 속도가 다 있어야 함)
    acc = np.abs(_shift(spd, K, np.nan) - _shift(spd, -K, np.nan)) / (2 * K / fps)
    return spd, acc, jump
