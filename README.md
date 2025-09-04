# 📊 Node Metrics Forecasting

서버에서 수집한 **Node Exporter 리소스 메트릭(CPU, 메모리 등)** 데이터를 기반으로  
**XGBoost + Optuna**를 활용해 리소스 사용량을 시계열 예측하는 ML 파이프라인입니다.

---

## 🚀 Features
- 데이터 로드 및 전처리 (`src/data/load_data.py`, `src/data/preprocess.py`)
- 특징 생성 (Lag, Rolling window, Fourier, Holiday 등)
- 모델 학습: **XGBoost**
- 하이퍼파라미터 최적화: **Optuna (Bayesian Optimization)**
- 모델/스케일러 저장 및 재사용 (`joblib`)
- 결과 메타데이터 JSON 저장

---

## 📂 Project Structure

ntels_Project/
├── src/
│   ├── data/
│   │   ├── load_data.py
│   │   └── preprocess.py
│   ├── features/
│   │   └── feature_engineering.py
│   └── modeling/
│       └── train_tune.py
├── models/              # 학습된 모델 및 스케일러 저장
├── data/                # (Git에 업로드하지 않음, .gitignore 처리)
├── main.py              # Entry point
└── README.md

---

## ⚙️ Installation
```bash
git clone https://github.com/Manggee/node-exporter_ML.git
cd node-exporter_ML
python -m venv .venv
source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt


⸻

🖥️ Usage

1. 데이터 준비
	•	data/node_metrics.csv 파일은 GitHub에 업로드하지 않습니다.
	•	직접 Node Exporter에서 수집하거나, 제공된 샘플 데이터를 다운로드하세요.

2. 학습 실행

python src/modeling/train_tune.py --trials 100 --out_prefix best_model

3. 추론 실행

python main.py --csv data/node_metrics.csv


⸻

📈 Results
	•	최종 모델: models/best_model_xgb_model.joblib
	•	스케일러: models/best_model_scaler.joblib
	•	구성/결과: models/best_model_config.json
	•	검증 MAE: ~0.12 (예시)

⸻

🛠 Tech Stack
	•	Python 3.10+
	•	pandas, numpy, scikit-learn
	•	xgboost
	•	optuna
	•	joblib

⸻

⚠️ Note
	•	data/ 폴더의 원본 데이터는 크기 문제로 GitHub에 포함되지 않습니다.
	•	대용량 데이터는 별도로 관리하세요 (예: Git LFS, DVC, Releases 등).

---
