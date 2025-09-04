import argparse
import json
import random
import sys
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple
import itertools

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import joblib
import optuna  # Optuna 라이브러리 import

from src.data.load_data import load_and_prepare, DATA_PATH
from src.data.preprocess import preprocess_load_metric
from src.features.feature_engineering import make_features


# -----------------------------
# Config / Utilities
# -----------------------------
@dataclass
class FeatureConfig:
    lags: List[int]
    rolling_window: List[int]
    use_all_stats: bool
    horizon: int
    use_holidays: bool
    use_fourier: bool


@dataclass
class ModelConfig:
    n_estimators: int
    max_depth: int
    learning_rate: float
    subsample: float
    colsample_bytree: float
    reg_alpha: float
    reg_lambda: float
    min_child_weight: float


@dataclass
class TrialResult:
    feat_cfg: FeatureConfig
    model_cfg: ModelConfig
    mae: float
    n_train: int
    n_valid: int


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


def time_split(
        X: pd.DataFrame, y: pd.Series, valid_ratio: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    assert 0 < valid_ratio < 1
    split_idx = int(len(X) * (1 - valid_ratio))
    X_tr, X_va = X.iloc[:split_idx].copy(), X.iloc[split_idx:].copy()
    y_tr, y_va = y.iloc[:split_idx].copy(), y.iloc[split_idx:].copy()
    return X_tr, X_va, y_tr, y_va


def build_dataset(df_processed: pd.DataFrame, feat_cfg: FeatureConfig) -> Tuple[pd.DataFrame, pd.Series]:
    df_feat = make_features(
        df_processed,
        lags=feat_cfg.lags,
        rolling_window=feat_cfg.rolling_window,
        use_all_stats=feat_cfg.use_all_stats,
        use_holidays=feat_cfg.use_holidays,
        use_fourier=feat_cfg.use_fourier,
    )
    target_base = "mean" if "mean" in df_feat.columns else "value"
    target_col = f"target_t+{feat_cfg.horizon}"
    df_feat[target_col] = df_feat[target_base].shift(-feat_cfg.horizon)
    df_feat = df_feat.dropna()

    y = df_feat[target_col].copy()
    X = df_feat.drop(columns=[target_col]).copy()
    return X, y


def train_evaluate(
        X: pd.DataFrame, y: pd.Series, model_cfg: ModelConfig, valid_ratio: float = 0.2
) -> Tuple[float, Dict[str, Any]]:
    X_tr, X_va, y_tr, y_va = time_split(X, y, valid_ratio=valid_ratio)

    float_cols = X_tr.select_dtypes(include=["float32", "float64"]).columns
    scaler = StandardScaler()

    X_tr = X_tr.copy();
    X_va = X_va.copy()
    X_tr.loc[:, float_cols] = scaler.fit_transform(X_tr[float_cols])
    X_va.loc[:, float_cols] = scaler.transform(X_va[float_cols])

    model = XGBRegressor(
        n_estimators=model_cfg.n_estimators,
        max_depth=model_cfg.max_depth,
        learning_rate=model_cfg.learning_rate,
        subsample=model_cfg.subsample,
        colsample_bytree=model_cfg.colsample_bytree,
        reg_alpha=model_cfg.reg_alpha,
        reg_lambda=model_cfg.reg_lambda,
        min_child_weight=model_cfg.min_child_weight,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )

    model.fit(X_tr, y_tr, verbose=False)

    pred = model.predict(X_va)
    mae = mean_absolute_error(y_va, pred)
    info = {
        "n_train": len(X_tr),
        "n_valid": len(X_va),
        "scaler": scaler,
        "num_cols": list(float_cols),
        "feature_names": list(X_tr.columns),
        "valid_ratio": valid_ratio,
    }
    return mae, info


# 베이지안 최적화를 위한 objective 함수 👇
def objective(trial, df_processed, valid_ratio: float = 0.2):  # <<< 1. 여기에 valid_ratio 추가
    # 피처 하이퍼파라미터 탐색 공간 정의
    lags = trial.suggest_categorical('lags', ((1, 2, 3), (1, 2, 3, 4, 5)))
    rolling_window = trial.suggest_categorical('rolling_window', ((3, 6), (3, 6, 12)))
    use_all_stats = trial.suggest_categorical('use_all_stats', (True, False))
    use_holidays = trial.suggest_categorical('use_holidays', (True, False))
    use_fourier = trial.suggest_categorical('use_fourier', (True, False))
    horizon = trial.suggest_categorical('horizon', (1, 3))

    # 모델 하이퍼파라미터 탐색 공간 정의
    n_estimators = trial.suggest_int('n_estimators', 200, 700)
    max_depth = trial.suggest_int('max_depth', 4, 10)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.2, log=True)
    subsample = trial.suggest_float('subsample', 0.7, 1.0)
    colsample_bytree = trial.suggest_float('colsample_bytree', 0.7, 1.0)
    reg_alpha = trial.suggest_float('reg_alpha', 0.001, 10, log=True)
    reg_lambda = trial.suggest_float('reg_lambda', 0.001, 100, log=True)
    min_child_weight = trial.suggest_float('min_child_weight', 0.1, 100, log=True)

    feat_cfg = FeatureConfig(
        lags=lags,
        rolling_window=rolling_window,
        use_all_stats=use_all_stats,
        horizon=horizon,
        use_holidays=use_holidays,
        use_fourier=use_fourier
    )

    model_cfg = ModelConfig(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        min_child_weight=min_child_weight
    )

    try:
        X, y = build_dataset(df_processed, feat_cfg)
        if len(X) < 200:
            return float('inf')  # 데이터가 부족하면 무한대 반환

        # <<< 2. train_evaluate 호출 시 valid_ratio 전달
        mae, _ = train_evaluate(X, y, model_cfg, valid_ratio=valid_ratio)
        return mae
    except Exception as e:
        # 에러 발생 시 어떤 에러인지 출력하면 디버깅에 도움이 됩니다.
        print(f"An error occurred during trial: {e}")
        return float('inf')  # 에러 발생 시 무한대 반환


def main():
    import os
    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_PATH = BASE_DIR / "data" / "node_metrics.csv"

    # 모델 저장 경로 설정 👇
    MODELS_DIR = BASE_DIR / "models"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)  # 폴더가 없으면 생성

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=DATA_PATH, help="원본 CSV 경로")
    parser.add_argument("--valid_ratio", type=float, default=0.2, help="검증 비율")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trials", type=int, default=100, help="최적화 시도 횟수")
    parser.add_argument("--out_prefix", default="best_model", help="산출물 파일명 접두사")
    args = parser.parse_args()

    set_seed(args.seed)

    # print("📥 로드 중...")
    # df_raw = load_and_prepare(args.csv)
    # print("✅ raw:", df_raw.shape)
    #
    # print("🔄 전처리 중...")
    # df_processed = preprocess_load_metric(df_raw)
    # print("✅ 전처리된 데이터:", df_processed.shape)
    #
    # mid_point = len(df_processed) // 2
    # df_part1 = df_processed.iloc[:mid_point].copy()
    # df_part2 = df_processed.iloc[mid_point:].copy()
    #
    # print(f"✅ df_part1 크기: {len(df_part1)}")
    # print(f"✅ df_part2 크기: {len(df_part2)}")
    #
    # # 전처리 데이터 저장 경로 변경 👇
    # df_part1_path = BASE_DIR / "data" / f"{args.out_prefix}_preprocessed_part1.csv"
    # df_part2_path = BASE_DIR / "data" / f"{args.out_prefix}_preprocessed_part2.csv"
    # df_part1.to_csv(df_part1_path)
    # df_part2.to_csv(df_part2_path)
    # print(f"✅ 전처리된 데이터 저장 완료: {df_part1_path} 와 {df_part2_path}")
    #
    # df_processed = df_part1

    # 전처리된 데이터를 바로 불러와서 사용 👇
    preprocessed_path = BASE_DIR / "data" / f"{args.out_prefix}_preprocessed_part1.csv"

    print(f"📥 전처리된 데이터 로드 중... ({preprocessed_path})")
    try:
        df_processed = pd.read_csv(preprocessed_path, index_col='timestamp', parse_dates=['timestamp'])
        print(f"✅ 전처리된 데이터 로드 완료: {df_processed.shape}")
    except FileNotFoundError:
        print(f"❌ 파일을 찾을 수 없습니다: {preprocessed_path}", file=sys.stderr)
        sys.exit(1)

    study = optuna.create_study(direction='minimize')
    print(f"🔍 베이지안 최적화 시작. 총 {args.trials}회 탐색.")
    study.optimize(lambda trial: objective(trial, df_processed), n_trials=args.trials, show_progress_bar=True)

    # 모든 탐색 결과를 CSV로 저장 👇
    print("\n✅ 모든 탐색 결과를 CSV로 저장 중...")
    try:
        df_trials = study.trials_dataframe()
        df_trials.to_csv(MODELS_DIR / f"{args.out_prefix}_trials.csv", index=False)
        print(f"✅ 모든 탐색 결과가 {MODELS_DIR / f'{args.out_prefix}_trials.csv'} 파일에 저장되었습니다.")
    except Exception as e:
        print(f"❌ 탐색 결과를 CSV로 저장하는 중 오류가 발생했습니다: {e}")

    best_trial = study.best_trial

    print("\n🏁 최적 구성으로 재학습 & 저장 중...")
    feat_cfg = FeatureConfig(**{k: v for k, v in best_trial.params.items() if k in FeatureConfig.__annotations__})
    model_cfg = ModelConfig(**{k: v for k, v in best_trial.params.items() if k in ModelConfig.__annotations__})

    X, y = build_dataset(df_processed, feat_cfg)
    X_tr, X_va, y_tr, y_va = time_split(X, y, valid_ratio=args.valid_ratio)

    float_cols = X_tr.select_dtypes(include=["float32", "float64"]).columns
    scaler = StandardScaler()

    X_tr.loc[:, float_cols] = scaler.fit_transform(X_tr[float_cols])
    X_va.loc[:, float_cols] = scaler.transform(X_va[float_cols])

    model = XGBRegressor(
        n_estimators=model_cfg.n_estimators,
        max_depth=model_cfg.max_depth,
        learning_rate=model_cfg.learning_rate,
        subsample=model_cfg.subsample,
        colsample_bytree=model_cfg.colsample_bytree,
        reg_alpha=model_cfg.reg_alpha,
        reg_lambda=model_cfg.reg_lambda,
        min_child_weight=model_cfg.min_child_weight,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
        verbosity=0
    )
    model.fit(X_tr, y_tr, verbose=False)

    pred = model.predict(X_va)
    mae = mean_absolute_error(y_va, pred)

    meta = {
        "trial": {
            'feat_cfg': asdict(feat_cfg),
            'model_cfg': asdict(model_cfg),
            'mae': float(mae)
        },
        "final_valid_mae": float(mae),
        "feature_names": list(X_tr.columns),
        "valid_ratio": args.valid_ratio,
        "csv_path": str(args.csv),
    }

    # 최종 모델, 스케일러, config 저장 경로 변경 👇
    joblib.dump(model, MODELS_DIR / f"{args.out_prefix}_xgb_model.joblib")
    joblib.dump(scaler, MODELS_DIR / f"{args.out_prefix}_scaler.joblib")
    with open(MODELS_DIR / f"{args.out_prefix}_config.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("\n✅ 저장 완료")
    print(f"  - 모델: {MODELS_DIR / f'{args.out_prefix}_xgb_model.joblib'}")
    print(f"  - 스케일러: {MODELS_DIR / f'{args.out_prefix}_scaler.joblib'}")
    print(f"  - 구성/결과: {MODELS_DIR / f'{args.out_prefix}_config.json'}")
    print(f"  - 최종 검증 MAE: {mae:.4f}")
    print(f"  - 최적 FeatureConfig: {asdict(feat_cfg)}")
    print(f"  - 최적 ModelConfig: {asdict(model_cfg)}")


if __name__ == "__main__":
    main()