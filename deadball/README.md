# 데드볼(D) 탐지 — 학습부터 대회 데이터 적용까지

선수 움직임만으로 "경기 중 / 정지"를 판별하는 모델. 공 정보는 쓰지 않아 긴 공 결측 안에서도 작동.
학습은 Sportec 공개 7경기(정답 있음), 대회 데이터는 적용만 (대회 데이터를 학습·튜닝에 쓰지 않음).

## 설치
```
pip install numpy scipy pandas scikit-learn joblib kloppy
```

## 실행 순서

### 1. Sportec 내려받기 (한 번만, 경기당 몇 분)
```
python sportec_load.py
```
→ `raw/J03W??.npz` 7개

### 2. (선택) 잡음 흉내가 대회 데이터와 비슷한지 확인
```
python check_sim.py J03WMX
```
→ 대회 noise_stats.py 결과와 숫자 비교 (평균 추적 인원, 순간이동 횟수, 가속도 분포 등)

### 3. 학습용 특징 만들기 (경기당 1~2분, 메모리 최대 약 2 GB)
```
python make_data.py
```
→ `data/{경기}_{clean|h1|h2}.npz` 21개
- clean: 깨끗한 Sportec / h1: 대회 1하프 모양 잡음 / h2: 대회 2하프 모양 잡음

### 4. 성능 확인: 경기 하나씩 빼며 7번 테스트 (약 20~40분)
```
python train.py eval
```
→ 화면에 경기별·평균 성능, `results_eval.csv`
- 옵션(환경변수): `TRAIN_CONDS=clean` (흉내 없이 학습), `STANDARDIZE=0` (표준화 끔)
- 예: 흉내 없이 학습한 경우와 비교 → `TRAIN_CONDS=clean python train.py eval`

### 5. 최종 모델 학습
```
python train.py final
```
→ `model/model.joblib`, `model/meta.json` (특징 이름, 고른 후처리, 표준화 여부)

### 6. 대회 데이터에 적용 (대회 저장소 루트에서)
```
cd 대회저장소
python 경로/deadball/predict_atd.py --model 경로/deadball/model --gaps 경로/gaps_over_3s.csv --out deadball_out
```
- `starter/load_atd.py`로 `data/tactical/BAR-ATH_tactical_atd.xml/.txt`를 읽음 (noise_stats.py와 같은 방식)
- `--gaps`는 선택. `start_frame`, `end_frame` 컬럼(video frame)이 있으면 됨
- 출력 (`deadball_out/`)
  - `deadball_frames.csv`: video_frame, half, graded, p_dead(정지 확률), dead(1=D, 0=경기 중, -1=채점 안 함)
  - `deadball_intervals.csv`: D 구간 (start_frame, end_frame, length_frames, source)
  - `gaps_with_dead.csv`: 긴 결측마다 D 비율(dead_frac)과 결측 안 D 구간

## 파일
| 파일 | 하는 일 |
|---|---|
| config.py | 설정값 (경기장 크기, 채점 구간, 기준값) |
| sportec_load.py | Sportec 내려받기·변환 |
| noise_sim.py | 잡음 흉내 (h1, h2 프로필) |
| noise_stats.py | 잡음 통계 (대회 데이터용 실행 + 흉내 확인용 함수) |
| check_sim.py | 흉내 데이터에 noise_stats 적용 |
| preprocess.py | 좌표 변환, 순간이동 지점에서 슬롯 끊기, 조각 안에서만 속도·가속도 |
| features.py | 특징 347개, 경기마다 표준화 |
| make_data.py | Sportec 학습용 특징 파일 만들기 |
| evalx.py | 평가 지표, 후처리(기준값·메디안 필터·Viterbi) |
| train.py | 7번 테스트(eval), 최종 모델(final) |
| predict_atd.py | 대회 데이터 적용 |

## 처리 흐름
| 단계 | 대회 데이터 | Sportec |
|---|---|---|
| 좌표 | 0~1 → 미터·가운데 원점 (105.3×68.0) | 이미 미터 |
| 잡음 흉내 | 안 함 | h1, h2 |
| 전처리 | 순간이동(12 m/s 초과) 지점에서 그 슬롯만 끊기, 조각 안에서만 속도·가속도 | 같음 |
| 특징 | 347개 (하프 경계 안 넘음) | 같음 |
| 표준화 | 전반 + 후반 킥오프 이후 프레임 기준 | 경기 전체 기준 |
| 후처리 | 학습 때 검증 경기로 고른 것 | |
| 고정 규칙 | 후반 킥오프 전(110808~113741) = D, 채점 밖 = -1 | |

## 주의
- 대회 ATD 컬럼 이름은 `{슬롯}_x`, `{슬롯}_y` 형식을 가정 (noise_stats.py와 같은 로더, 로컬에서 동작 확인됨)
- 특징 코드를 바꾸면 반드시 make_data → train final → predict 순서로 다시 돌릴 것 (predict가 특징 이름이 다르면 멈춤)
- 학습에 쓰지 않은 정보: 대회 경기의 정답·영상 판단. 흉내 숫자는 대회 데이터의 요약 통계(라벨 아님)로 맞춤
