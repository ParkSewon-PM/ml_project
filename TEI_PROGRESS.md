# TEI 학습 진행상황

## 현재 위치: Day 4 (Part 1 — 프로젝트 전체 그림)

### Part 0 — Python 기초
- [x] Day 1: Python 문법 기초 (src/utils/config.py) — import, 함수, 딕셔너리, 재귀
- [x] Day 2: pandas 기초 (src/features/basic_stats.py) — DataFrame, numpy, 데코레이터, 통계
- [x] Day 3: numpy + 클래스 (src/models/base.py) — 클래스, 추상 클래스, @abstractmethod, @classmethod

### Part 1 — 프로젝트 전체 그림
- [ ] Day 4: README, 아키텍처, 설정

### Part 2 — 데이터 생성
- [x] Day 5: SUMO 시뮬레이션 (scenario_generator, network_builder, runner, trajectory_collector, probe_extractor) + 1km 리샘플링은 Day 10에서 다룸
- [x] Day 6: Ground Truth (Edie 공식으로 밀도/흐름 계산)

### Part 3 — 피처 엔지니어링
- [x] Day 7: 레지스트리 패턴 (registry.py, pipeline.py) + 데코레이터, 싱글톤, **kwargs
- [x] Day 8: 피처 파일들 (basic_stats 구조 동일, 스킵)

### Part 4 — ML 모델
- [x] Day 9: 모델 아키텍처 (XGBoost, LightGBM, LSTM, CNN-1D, FD Baseline) + 단일 프로브 R²≈0.45 정체
- [x] Day 10: 다중 프로브 융합 (Aligned N=5 MAE 1.78 / R² 0.964 vs Deployed Bayesian+CF N=5 MAE 2.18 / R² 0.951) + 1km 리샘플링 + CF Score (additive/multiplicative) + 베이지안/softmax 앙상블 + Feature-level 스프레드 + DeepSets + v0.5.0 split-leakage fix

### Part 5 — 학습/평가
- [x] Day 11: GroupKFold 5-fold, Optuna, Early stopping, R²/RMSE/MAE/MAPE, SHAP

### Part 6 — 배포
- [x] Day 12: FastAPI (ingest→inference→ensemble→map), Docker+Cloud Run, WebSocket, API/Pydantic/Swagger
- [x] Day 13: GIS(grid+heading), Kalman 2D, Kafka/Pub-Sub(추상클래스 패턴), 전체 기술스택, GCP(Cloud Run+Cloud SQL+Pub/Sub+AR+Secret Manager), CI/CD(GitHub Actions)

### Part 7 — 발표 준비
- [x] Day 14: 5분 스크립트 + 예상 질문 20개 답변 완성

---

## 학습 노트

### Day 1 Notes:
(여기에 배운 내용 정리)
