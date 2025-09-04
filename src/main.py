import argparse
import json
import random
import sys
import os
from datetime import datetime
from pathlib import Path
from dataclasses import asdict
from typing import Dict, Any, Tuple
import matplotlib.font_manager as fm

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import joblib
import optuna
import psycopg2
import psycopg2.extras

from src.data.load_data import load_and_prepare
from src.data.preprocess import preprocess_load_metric
from src.features.feature_engineering import make_features
# train_tune.py에 있는 함수들을 직접 가져오는 것으로 가정합니다.
from src.modeling.train_tune import set_seed, build_dataset, train_evaluate, FeatureConfig, ModelConfig, objective, \
    time_split
from src.modeling.validate import build_dataset_for_validation, save_validation_results_to_db, load_db_config


def main():
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_PATH = BASE_DIR / "src" / "data" / "node_metrics.csv"
    MODELS_DIR = BASE_DIR / "src" / "models"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=DATA_PATH, help="원본 CSV 경로")
    parser.add_argument("--valid_ratio", type=float, default=0.2, help="검증 비율")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trials", type=int, default=100, help="최적화 시도 횟수")
    parser.add_argument("--out_prefix", default="best_model", help="산출물 파일명 접두사")
    parser.add_argument("--db-config", default=BASE_DIR / "src" / "config" / "db_config.json", help="DB 접속 정보 JSON 경로")
    args = parser.parse_args()

    set_seed(args.seed)

    # =========================================================
    # Step 1: 데이터 로드, 전처리 및 분할
    # =========================================================
    print("----- Step 1: 데이터 로드, 전처리 및 분할 -----")
    df_part1_path = BASE_DIR / "src" / "data" / f"{args.out_prefix}_preprocessed_part1.csv"
    df_part2_path = BASE_DIR / "src" / "data" / f"{args.out_prefix}_preprocessed_part2.csv"

    if df_part1_path.exists() and df_part2_path.exists():
        print(f"✅ 전처리된 파일이 이미 존재합니다. {df_part1_path}에서 데이터를 로드합니다.")
        df_processed = pd.read_csv(df_part1_path, index_col='timestamp', parse_dates=True)
    else:
        print("📍전처리된 파일이 없습니다. 원본 데이터 로드 및 전처리를 시작합니다.")
        df_raw = load_and_prepare(DATA_PATH)
        print("원본 데이터 크기:", df_raw.shape)

        df_processed_all = preprocess_load_metric(df_raw)
        print("전처리된 데이터 크기:", df_processed_all.shape)

        mid_point = len(df_processed_all) // 2
        df_part1 = df_processed_all.iloc[:mid_point].copy()
        df_part2 = df_processed_all.iloc[mid_point:].copy()

        df_part1.to_csv(df_part1_path)
        df_part2.to_csv(df_part2_path)
        print(f"✅ 전처리된 데이터 저장 완료: {df_part1_path} 와 {df_part2_path}")

        df_processed = df_part1

    # =========================================================
    # Step 2: 베이지안 최적화를 통한 모델 튜닝
    # =========================================================
    print("\n----- Step 2: 베이지안 최적화 시작 -----")
    study = optuna.create_study(direction='minimize')
    print(f"🔍 총 {args.trials}회 탐색.")
    # objective 함수에 valid_ratio 전달
    objective_func = lambda trial: objective(trial, df_processed, valid_ratio=args.valid_ratio)
    study.optimize(objective_func, n_trials=args.trials, show_progress_bar=True)
    best_trial = study.best_trial

    # 최적 MAE 값을 변수에 저장
    best_mae = best_trial.value
    print(f"\n✨ 최적화 완료! Best MAE: {best_mae:.4f}")

    # 모든 탐색 결과를 CSV로 저장
    print("\n✅ 모든 탐색 결과를 CSV로 저장 중...")
    try:
        df_trials = study.trials_dataframe()
        df_trials.to_csv(MODELS_DIR / f"{args.out_prefix}_trials.csv", index=False)
        print(f"✅ 모든 탐색 결과가 {MODELS_DIR / f'{args.out_prefix}_trials.csv'} 파일에 저장되었습니다.")
    except Exception as e:
        print(f"❌ 탐색 결과를 CSV로 저장하는 중 오류가 발생했습니다: {e}")

    # =========================================================
    # Step 3: 최적 모델 재학습 및 저장
    # =========================================================
    print("\n----- Step 3: 최적 구성으로 모델 재학습 및 저장 -----")
    feat_cfg = FeatureConfig(**{k: v for k, v in best_trial.params.items() if k in FeatureConfig.__annotations__})
    model_cfg = ModelConfig(**{k: v for k, v in best_trial.params.items() if k in ModelConfig.__annotations__})

    # <<< 수정: 최종 모델은 df_part1 전체 데이터로 학습 (데이터 분할 X)
    X_train, y_train = build_dataset(df_processed, feat_cfg)

    float_cols = X_train.select_dtypes(include=["float32", "float64"]).columns
    scaler = StandardScaler()
    X_train.loc[:, float_cols] = scaler.fit_transform(X_train[float_cols])

    model = XGBRegressor(
        n_estimators=model_cfg.n_estimators, max_depth=model_cfg.max_depth, learning_rate=model_cfg.learning_rate,
        subsample=model_cfg.subsample, colsample_bytree=model_cfg.colsample_bytree, reg_alpha=model_cfg.reg_alpha,
        reg_lambda=model_cfg.reg_lambda, min_child_weight=model_cfg.min_child_weight, random_state=42, n_jobs=-1,
        tree_method="hist")
    model.fit(X_train, y_train, verbose=False)

    # <<< 수정: MAE를 새로 계산하지 않고, Optuna가 찾은 최적의 값을 사용
    meta = {
        "trial": {
            'feat_cfg': asdict(feat_cfg),
            'model_cfg': asdict(model_cfg),
            'mae': best_mae  # Optuna의 결과값 사용
        },
        "final_valid_mae": best_mae,  # Optuna의 결과값 사용
        "feature_names": list(X_train.columns),
        "valid_ratio": args.valid_ratio,  # 튜닝 시 사용된 비율 기록
        "csv_path": str(args.csv),
        "training_data_shape": X_train.shape,
    }
    joblib.dump(model, MODELS_DIR / f"{args.out_prefix}_xgb_model.joblib")
    joblib.dump(scaler, MODELS_DIR / f"{args.out_prefix}_scaler.joblib")
    with open(MODELS_DIR / f"{args.out_prefix}_config.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print("✅ 모델 및 설정 파일 저장 완료")
    print(f"  - 최종 검증 MAE (from Optuna): {best_mae:.4f}")

    # =========================================================
    # Step 4: df_part2 데이터로 최종 모델 검증
    # =========================================================
    print("\n----- Step 4: df_part2 데이터로 모델 검증 -----")
    X_val, y_val = build_dataset_for_validation(df_part2_path, feat_cfg)

    # 누락된 컬럼이 있을 경우 0으로 채우기 (훈련 시점에는 있었으나 검증 시점에는 없는 피처)
    missing_cols = set(X_train.columns) - set(X_val.columns)
    for c in missing_cols:
        X_val[c] = 0
    X_val = X_val[X_train.columns]  # 훈련 데이터와 컬럼 순서 일치

    if float_cols is not None:
        cols_to_scale = [c for c in float_cols if c in X_val.columns]
        if cols_to_scale:
            X_val.loc[:, cols_to_scale] = scaler.transform(X_val[cols_to_scale])

    y_pred_np = model.predict(X_val)
    y_pred = np.array([float(x) for x in y_pred_np])
    final_mae = mean_absolute_error(y_val, y_pred)
    horizon = feat_cfg.horizon

    print("\n✅ df_part2 전체 데이터에 대한 예측 완료")
    print(f"  - 예측 시점: t+{horizon} ({int(horizon) * 5}분 후)")
    print(f"  - 최종 검증 MAE (on df_part2): {final_mae:.4f}")

    # DB 저장
    db_config = load_db_config(args.db_config)
    save_validation_results_to_db(final_mae, y_val, y_pred, int(horizon), db_config)

    # =========================================================
    # <<< Step 5: 시각화 결과 저장 (새로 추가된 부분)
    # =========================================================
    font_path = None
    if os.name == 'posix':
        try:
            if 'Darwin' in os.uname():
                font_path = '/System/Library/Fonts/AppleGothic.ttf'
            elif os.path.exists('/usr/share/fonts/truetype/nanum/NanumGothic.ttf'):
                font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
        except:
            pass
    elif os.name == 'nt':
        font_path = 'C:/Windows/Fonts/malgunbd.ttf'

    if font_path and os.path.exists(font_path):
        font_name = fm.FontProperties(fname=font_path).get_name()
        plt.rc('font', family=font_name)
    else:
        for font in fm.fontManager.ttflist:
            if 'Gothic' in font.name or 'Malgun' in font.name or 'Nanum' in font.name:
                plt.rc('font', family=font.name)
                break
    plt.rcParams['axes.unicode_minus'] = False
    # ===== 1. plot 시각화 =====
    print("\n📊 예측 결과 시각화...")

    plt.figure(figsize=(15, 7))
    plt.plot(y_val.index, y_val, label='실제 값 (df_part2)', color='blue', alpha=0.7)
    plt.plot(y_val.index, y_pred, label='예측 값', color='red', linestyle='--')

    plt.title(f"df_part2 예측 결과 비교 (t+{horizon} = {int(horizon) * 5}분 후)")
    plt.xlabel("시점")
    plt.ylabel("Node Load1")
    plt.legend()
    plt.grid(True)

    plt.xticks(rotation=45)
    plt.tight_layout()

    # 결과 이미지 저장하기
    save_dir = Path.home() / "Code" / "ntels_Project" / "result"
    save_dir.mkdir(parents=True, exist_ok=True)
    ts_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    outfile = save_dir / f"{ts_str}_베이지안최적화.png"

    fig = plt.gcf()  # 현재 figure 객체
    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    print(f"plot 저장 완료 : {outfile}")

    plt.show()


if __name__ == "__main__":
    main()