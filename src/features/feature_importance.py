# -*- coding: utf-8 -*-
"""
feature_importance.py
- 저장된 모델/스케일러/구성으로 피처 중요도 산출
- 기본: XGBoost 내장 중요도(feature_importances_ 또는 booster.get_score)
- 옵션: --shap 사용 시 SHAP 요약 플롯 저장(설치되어 있으면)
"""

import argparse
import json
import warnings

import joblib
import pandas as pd
from xgboost import XGBRegressor

from load_data import load_and_prepare
from preprocess import preprocess_load_metric
from feature_engineering import make_features


def rebuild_features(csv_path: str, feat_cfg: dict) -> pd.DataFrame:
    """
    저장된 FeatureConfig로 전체 피처프레임 생성(타깃 제외)
    - load_and_prepare -> preprocess_load_metric -> make_features
    - lag/rolling 때문에 앞부분 NaN은 drop
    """
    df_raw = load_and_prepare(csv_path)
    df_proc = preprocess_load_metric(df_raw)
    df_feat = make_features(
        df_proc,
        lags=feat_cfg["lags"],
        rolling_window=feat_cfg["rolling_window"],
        use_all_stats=feat_cfg["use_all_stats"],
    )
    df_feat = df_feat.dropna()
    return df_feat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="best_model_config.json", help="학습 시 저장된 구성(JSON)")
    parser.add_argument("--model", default="best_model_xgb_model.joblib", help="학습 모델 경로")
    parser.add_argument("--scaler", default="best_model_scaler.joblib", help="스케일러 경로")
    parser.add_argument("--out_csv", default="feature_importance.csv", help="피처 중요도 저장 CSV")
    parser.add_argument("--shap", action="store_true", help="가능하면 SHAP 요약 플롯 저장")
    parser.add_argument("--shap_png", default="shap_summary.png", help="SHAP 요약 플롯 출력 PNG")
    args = parser.parse_args()

    # 구성 로드
    with open(args.config, "r") as f:
        meta = json.load(f)

    # CSV 경로 / FeatureConfig 복원 (train_tune.py 또는 train_tune_fixed.py 양쪽 포맷 지원)
    csv_path = meta.get("csv_path", "../node_metrics.csv")
    if "trial" in meta and "feat_cfg" in meta["trial"]:
        feat_cfg = meta["trial"]["feat_cfg"]
    elif "fixed_feature_config" in meta:
        feat_cfg = meta["fixed_feature_config"]
    else:
        raise RuntimeError("FeatureConfig 정보를 config에서 찾지 못했습니다.")

    # 피처 생성
    X = rebuild_features(csv_path, feat_cfg)

    # 모델/스케일러 로드
    model: XGBRegressor = joblib.load(args.model)
    scaler = joblib.load(args.scaler)

    # 학습 당시 스케일된 컬럼에만 동일하게 스케일 적용
    float_cols = getattr(scaler, "feature_names_in_", None)
    X_scaled = X.copy()
    if float_cols is not None:
        cols_to_scale = [c for c in float_cols if c in X_scaled.columns]
        if cols_to_scale:
            X_scaled.loc[:, cols_to_scale] = scaler.transform(X_scaled[cols_to_scale])
    else:
        # fallback: 실수형만 스케일(정보 부족 시 경고만 출력)
        warnings.warn("스케일러에 feature_names_in_이 없어 fallback 전략을 사용합니다.", RuntimeWarning)
        cols_to_scale = X_scaled.select_dtypes(include=["float32", "float64"]).columns
        if len(cols_to_scale) > 0:
            X_scaled.loc[:, cols_to_scale] = scaler.transform(X_scaled[cols_to_scale])

    # 중요도 계산
    # 1) 기본 속성(feature_importances_)가 있으면 우선 사용
    importances = getattr(model, "feature_importances_", None)
    if importances is not None and len(importances) == X_scaled.shape[1]:
        imp_series = pd.Series(importances, index=X_scaled.columns)
    else:
        # 2) booster.get_score(importance_type='gain')로 대체
        booster = model.get_booster()
        score = booster.get_score(importance_type="gain")
        # 스코어에 없는 피처는 0으로 채움
        imp_series = pd.Series({feat: score.get(feat, 0.0) for feat in X_scaled.columns})

    imp_df = (
        imp_series.sort_values(ascending=False)
        .rename("importance")
        .reset_index()
        .rename(columns={"index": "feature"})
    )
    imp_df.to_csv(args.out_csv, index=False)

    print("✅ 피처 중요도 저장 완료:", args.out_csv)
    print(imp_df.head(20))

    # (선택) SHAP 요약 플롯
    # if args.shap:
    #     try:
    #         import shap
    #         import matplotlib.pyplot as plt
    #         sample = X_scaled.tail(min(2000, len(X_scaled)))  # 너무 크면 최근 일부만
    #         explainer = shap.Explainer(model, sample, feature_names=list(X_scaled.columns))
    #         shap_values = explainer(sample)
    #         shap.plots.beeswarm(shap_values, show=False, max_display=20)
    #         plt.tight_layout()
    #         plt.savefig(args.shap_png, dpi=150)
    #         plt.close()
    #         print("✅ SHAP 요약 플롯 저장 완료:", args.shap_png)
    #     except Exception as e:
    #         print("⚠️ SHAP 분석을 건너뜁니다:", e)


if __name__ == "__main__":
    main()