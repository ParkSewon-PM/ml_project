# ProbeDensity TEI 준비 커리큘럼

> 목표: 파이썬 기초부터 시작해서 이 프로젝트를 면접에서 자신있게 설명할 수 있는 수준까지
> 예상 소요: 매일 2-3시간 기준 약 2주

---

## PART 0 — Python 기초 (Day 1~3)

TEI에서 코드를 직접 짜진 않지만, 코드를 읽고 "왜 이렇게 했는지" 설명하려면 기초가 필요함.

### Day 1: Python 문법 기초
- [ ] 변수, 타입 (int, float, str, list, dict)
- [ ] for/while 반복문, if/else 조건문
- [ ] 함수 정의 (def), return, 파라미터
- [ ] import 구문 이해 (from X import Y)
- **실습**: `src/utils/config.py` 읽어보기 (42줄, 가장 짧고 간단)

### Day 2: pandas 기초
- [ ] DataFrame이란? (엑셀 시트라고 생각)
- [ ] pd.read_csv(), pd.read_parquet()
- [ ] df.groupby(), df.mean(), df.std()
- [ ] df.merge() — SQL의 JOIN과 같음
- [ ] df.fillna(), df.dropna() — 결측치 처리
- **실습**: `src/features/basic_stats.py` 읽어보기 — pandas로 속도 통계 계산하는 코드

### Day 3: numpy + 클래스
- [ ] numpy array란? (빠른 숫자 배열)
- [ ] np.mean(), np.std(), np.array()
- [ ] class, __init__, self 이해
- [ ] 상속 (class Child(Parent))
- **실습**: `src/models/base.py` 읽어보기 — 모든 모델의 부모 클래스 (41줄)

---

## PART 1 — 프로젝트 전체 그림 이해 (Day 4)

### Day 4: "이 프로젝트가 뭔지" 설명할 수 있게
- [ ] README.md 정독
- [ ] 핵심 질문에 답할 수 있어야 함:
  - **문제**: 교통 밀도(density)를 왜 측정해야 하고, 기존 방법의 한계는?
  - **접근**: 프로브 차량의 GPS+가속도 데이터만으로 어떻게 밀도를 추정?
  - **결과**: 단일 프로브 MAE 2.50 / R² 0.934, 5-프로브 정렬 융합 MAE 1.78 / R² 0.964 (−29% MAE), 배포용 Bayesian+CF 링크 융합 MAE 2.18 / R² 0.951
  - **아키텍처**: Offline ML Pipeline → Backend Server → Map Dashboard
- [ ] `configs/default.yaml` 읽기 — 프로젝트 설정 전체 구조 파악

**이 단계 끝나면 말할 수 있어야 하는 것:**
> "프로브 차량의 속도/가속도 패턴에서 31개 피처를 추출하고, 6가지 ML 모델을 비교해서 도로 밀도를 예측하는 시스템입니다. 단일 프로브 한계를 극복하기 위해 다중 프로브 베이지안 융합까지 구현했습니다."

---

## PART 2 — 데이터 생성 파이프라인 (Day 5~6)

### Day 5: SUMO 시뮬레이션
- [ ] SUMO란? (오픈소스 교통 시뮬레이터)
- [ ] `src/simulation/scenario_generator.py` — 35K개 시나리오를 어떻게 만드는지
  - density, speed_limit, num_lanes 조합으로 시나리오 매트릭스 생성
- [ ] `src/simulation/network_builder.py` — SUMO 도로 네트워크 XML 생성
- [ ] `src/simulation/runner.py` — TraCI로 SUMO 실행, FCD(Floating Car Data) 수집

### Day 6: Ground Truth + 프로브 추출
- [ ] `src/simulation/ground_truth.py` — Edie의 일반화된 정의로 밀도 계산
  - **핵심 개념**: Edie's generalized definitions (시공간 영역에서의 밀도/흐름 계산)
- [ ] `src/simulation/probe_extractor.py` — 전체 차량 중 프로브 차량만 추출
- [ ] `src/simulation/trajectory_collector.py` — 프로브 궤적을 6채널로 정리 (VX, VY, AX, AY, speed, brake)

**TEI에서 나올 수 있는 질문:**
> "왜 실제 데이터 대신 시뮬레이션 데이터를 사용했나요?"
> → 실제 데이터는 ground truth(실제 밀도)를 알 수 없음. 시뮬레이션에서는 모든 차량의 위치를 알 수 있어서 정확한 밀도 라벨링 가능.

---

## PART 3 — 피처 엔지니어링 (Day 7~8)

### Day 7: 피처 레지스트리 패턴
- [ ] `src/features/registry.py` — @register_feature 데코레이터 패턴
  - 각 피처 함수가 자기를 레지스트리에 등록하는 구조
- [ ] `src/features/pipeline.py` — 등록된 피처들을 모아서 한번에 추출
- [ ] 왜 이렇게 설계했는지 설명할 수 있어야 함:
  - 피처 추가/제거가 쉬움 (설정 파일로 on/off)
  - 각 피처 모듈이 독립적

### Day 8: 31개 피처 이해
- [ ] `src/features/basic_stats.py` — mean_speed, std_speed, cv_speed 등
- [ ] `src/features/acceleration.py` — 가속/감속 패턴 (car-following theory 기반)
- [ ] `src/features/brake_patterns.py` — 제동 빈도, 강도
- [ ] `src/features/time_series.py` — 자기상관, FFT, 엔트로피
- [ ] 각 피처가 **왜 밀도를 반영하는지** 직관적으로 설명할 수 있어야 함:
  - 밀도 높음 → 속도 낮고 변동 큼 → 급제동 많음 → 피처값 변화

**TEI에서 나올 수 있는 질문:**
> "31개 피처 중 가장 중요한 건 뭐였나요?"
> → SHAP 분석 결과 기반으로 답하기 (mean_speed, std_speed, brake_count 등 상위 피처)

---

## PART 4 — ML 모델 (Day 9~10)

### Day 9: 모델 아키텍처 이해
- [ ] `src/models/base.py` — BaseEstimator 인터페이스 (fit, predict, save, load)
- [ ] `src/models/tabular.py` — XGBoost, LightGBM (트리 기반 앙상블)
  - **핵심**: gradient boosting이 뭔지 한 줄로 설명할 수 있게
- [ ] `src/models/lstm.py` — 시계열 딥러닝 모델
- [ ] `src/models/cnn1d.py` — 1D 합성곱으로 궤적 패턴 학습
- [ ] `src/models/fd_models.py` — 교통공학 Fundamental Diagram 베이스라인
- [ ] `src/models/factory.py` — 모델 이름으로 인스턴스 생성 (팩토리 패턴)

### Day 10: 다중 프로브 융합
- [ ] `src/models/multi_probe.py` — 핵심 혁신
  - LSTMEncoder / CNN1DEncoder로 각 프로브 임베딩
  - AttentionPooling으로 프로브 간 가중 결합
  - CFScorePooling — car-following 이론 기반 가중치
- [ ] 단일 프로브 vs 다중 프로브 결과 차이 (정렬 N=1→N=5: MAE 2.50 → 1.78, R² 0.934 → 0.964)
- [ ] 왜 프로브가 많아지면 정확도가 올라가는지 직관적 설명

**TEI에서 나올 수 있는 질문:**
> "6개 모델 중 왜 XGBoost를 최종 배포 모델로 선택했나요?"
> → 정확도 비슷하지만 추론 속도 빠르고, 메모리 효율적, 해석 가능성(SHAP)

---

## PART 5 — 학습/평가 파이프라인 (Day 11)

### Day 11: 학습 + 평가
- [ ] `src/training/trainer_tabular.py` — 테이블 모델 학습 루프
- [ ] `src/training/trainer_dl.py` — 딥러닝 학습 루프 (에폭, 배치, 로스)
- [ ] `src/training/hyperopt.py` — Optuna로 하이퍼파라미터 최적화
- [ ] `src/training/cross_validation.py` — grouped K-fold (시나리오별 그룹)
- [ ] `src/evaluation/metrics.py` — R², MAE, RMSE 이해
- [ ] `src/evaluation/shap_analysis.py` — SHAP으로 피처 중요도 해석

**TEI에서 나올 수 있는 질문:**
> "cross-validation을 어떻게 했나요? 왜 grouped K-fold를 사용했나요?"
> → 같은 시나리오의 프로브들이 train/test에 동시에 들어가면 data leakage 발생. 시나리오 단위로 그룹을 나눠서 방지.

---

## PART 6 — 배포 & 실시간 시스템 (Day 12~13)

### Day 12: FastAPI 백엔드
- [ ] `src/api/app.py` — FastAPI 앱 구조, 엔드포인트 목록
- [ ] `src/api/inference.py` — 모델 로딩 → 예측 서빙
- [ ] `src/api/ingest.py` — 실시간 프로브 데이터 수신 (1002줄, 가장 복잡)
  - SessionManager, LinkBuffer, WebSocket 대시보드
- [ ] `src/api/ensemble.py` — 베이지안 링크 레벨 앙상블
- [ ] `src/api/map.py` — Leaflet 지도 API

### Day 13: 스트리밍 + GIS + 인프라
- [ ] `src/streaming/fusion.py` — Kalman 필터로 GPS+IMU 센서 융합
- [ ] `src/streaming/consumer.py` — Kafka에서 메시지 소비 → ML 추론
- [ ] `src/gis/link_matcher.py` — GPS 좌표 → 도로 링크 매칭
- [ ] `Dockerfile` + `docker-compose.yml` — 배포 구조
- [ ] Cloud Run 배포 흐름 이해

**TEI에서 나올 수 있는 질문:**
> "실시간 시스템에서 가장 어려웠던 기술적 챌린지는?"
> → 프로브들의 이동 경로가 다른 상황에서 같은 도로 링크에 대한 예측을 어떻게 합치는지 (overlap-aware link-level fusion)

---

## PART 7 — TEI 발표 준비 (Day 14)

### Day 14: 스토리 만들기
- [ ] 5분 프레젠테이션 구조:
  1. **문제 정의** (30초): 교통 밀도 측정의 한계
  2. **접근법** (1분): 시뮬레이션 → 피처 엔지니어링 → 모델 비교
  3. **핵심 결과** (1분): 정렬 5-프로브 MAE 1.78 / MAPE 24.6% / R² 0.964 (단일 N=1 대비 MAE −29%), 배포용 Bayesian+CF MAE 2.18 / R² 0.951
  4. **기술적 깊이** (1.5분): 다중 프로브 융합, 배포 아키텍처
  5. **배운 점** (1분): 트레이드오프, 한계, 개선 방향

- [ ] 예상 질문 20개에 대한 답변 준비
- [ ] "왜 이 기술을 선택했는지" 설명할 수 있는 트레이드오프 정리

---

## 매일 체크리스트

각 Day가 끝나면:
1. 읽은 파일의 핵심 함수 3개를 자기 말로 설명해보기
2. "왜 이렇게 설계했는지" 한 문장으로 정리
3. TEI 예상 질문 1개에 답변 써보기
