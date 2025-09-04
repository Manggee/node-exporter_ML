import argparse
import json
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.metrics import mean_absolute_error
import psycopg2
import psycopg2.extras  # 대량 삽입

from src.features.feature_engineering import make_features
from src.modeling.train_tune import FeatureConfig

"""
- df_part1으로 학습된 모델을 df_part2 데이터로 검증하는 스크립트
"""
# ---------------------------
# DB 설정 로더 (외부 파일)
# ---------------------------
def load_db_config(path: str) -> Dict[str, Any]:
    """
    db_config.json에서 접속 정보를 읽어 psycopg2.connect에 맞게 반환.
    - 'database' 키를 'dbname'으로 매핑해 호환성 확보
    - 빈 값/None은 제거
    """
    if not os.path.exists(path):
        print(f"❌ DB 설정 파일을 찾을 수 없습니다: {path}", file=sys.stderr)
        sys.exit(90)

    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # database -> dbname 보정 (psycopg2 표준 키)
    if "database" in cfg and "dbname" not in cfg:
        cfg["dbname"] = cfg.pop("database")

    # 빈 문자열/None 제거
    cfg = {k: v for k, v in cfg.items() if v not in ("", None)}
    return cfg


def build_dataset_for_validation(processed_csv_path: str, feat_cfg: Dict[str, Any]) -> tuple[pd.DataFrame, pd.Series]:
    """df_part2 전체 데이터를 로드하여 피처와 타겟을 생성"""
    try:
        df_proc = pd.read_csv(processed_csv_path, index_col='timestamp', parse_dates=['timestamp'])
    except FileNotFoundError:
        print(f"❌ 파일을 찾을 수 없습니다: {processed_csv_path}", file=sys.stderr)
        sys.exit(1)

    if df_proc.empty:
        print("❌ 로드된 전처리 데이터가 비어 있습니다.", file=sys.stderr)
        sys.exit(2)

    df_feat = make_features(
        df_proc,
        lags=feat_cfg.lags,
        rolling_window=feat_cfg.rolling_window,
        use_all_stats=feat_cfg.use_all_stats,
        use_holidays=feat_cfg.use_holidays,
        use_fourier=feat_cfg.use_fourier
    )

    target_base = "mean" if "mean" in df_feat.columns else "value"
    target_col = f"target_t+{feat_cfg.horizon}"
    df_feat[target_col] = df_feat[target_base].shift(-feat_cfg.horizon)
    df_feat = df_feat.dropna()

    y = df_feat[target_col].copy()
    X = df_feat.drop(columns=[target_col]).copy()

    return X, y


def save_validation_results_to_db(
    mae: float, y_val: pd.Series, y_pred: np.ndarray, horizon: int, db_config: Dict[str, Any]
):
    """검증 결과를 PostgreSQL에 저장 (복합 PK + UPSERT, 외부 파일 설정 사용)"""
    conn = None
    try:
        conn = psycopg2.connect(**db_config)
        cur = conn.cursor()

        # 테이블 생성: 복합 PK (timestamp, horizon)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS validation_results (
                timestamp       TIMESTAMP NOT NULL,
                horizon         INTEGER   NOT NULL,
                actual_value    REAL,
                predicted_value REAL,
                mae             REAL,
                created_at      TIMESTAMP,
                PRIMARY KEY (timestamp, horizon)
            );
        """)

        now = pd.Timestamp.now()
        records = [
            (
                (y_val.index[i].to_pydatetime()
                 if hasattr(y_val.index[i], "to_pydatetime")
                 else y_val.index[i]),
                int(horizon),
                float(y_val.iloc[i]),
                float(y_pred[i]),
                float(mae),
                now.to_pydatetime(),
            )
            for i in range(len(y_val))
        ]

        sql = """
            INSERT INTO validation_results
              (timestamp, horizon, actual_value, predicted_value, mae, created_at)
            VALUES %s
            ON CONFLICT (timestamp, horizon) DO UPDATE SET
              actual_value    = EXCLUDED.actual_value,
              predicted_value = EXCLUDED.predicted_value,
              mae             = EXCLUDED.mae,
              created_at      = EXCLUDED.created_at;
        """
        psycopg2.extras.execute_values(cur, sql, records, page_size=10000)
        conn.commit()
        print(f"✅ 검증 결과 {len(records)}개 UPSERT 완료")

    except psycopg2.Error as e:
        if conn:
            conn.rollback()
        print(f"❌ 데이터베이스 오류: {e}")
        raise
    finally:
        if conn:
            conn.close()


def main():
    BASE_DIR = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  default=str(BASE_DIR / "models" / "best_model_config.json"), help="학습 시 저장된 구성(JSON)")
    parser.add_argument("--model", default=str(BASE_DIR / "models" / "best_model_xgb_model.joblib"), help="학습 모델 경로")
    parser.add_argument("--scaler", default=str(BASE_DIR / "models" / "best_model_scaler.joblib"), help="스케일러 경로")
    parser.add_argument("--out_prefix", default="best_model", help="파일 접두사 (예: model_part1)")
    parser.add_argument("--db-config", default=str(BASE_DIR / "config" / "db_config.json"), help="DB 접속 정보 JSON 경로(깃에 올리지 않기)")
    args = parser.parse_args()

    # 피처 이름 목록, 최적 하이퍼, horizon 등 메타 정보가 들어있는 json 파일 로드
    with open(args.config, "r") as f:
        meta = json.load(f)

    if "trial" in meta and "feat_cfg" in meta["trial"]:
        feat_cfg = meta["trial"]["feat_cfg"]
        # 딕셔너리를 FeatureConfig 객체로 변환 👇
        feat_cfg = FeatureConfig(**feat_cfg)
    else:
        print("❌ FeatureConfig 정보를 찾지 못했습니다. config 파일을 확인하세요.", file=sys.stderr)
        sys.exit(1)

    horizon = feat_cfg.horizon

    BASE_DIR = Path(__file__).resolve().parent.parent

    # 파일 경로들 - 명확한 폴더 구조 반영
    PROCESSED_PATH = BASE_DIR / "data" / f"{args.out_prefix}_preprocessed_part2.csv"
    MODEL_PATH = BASE_DIR / "models" / f"{args.out_prefix}_xgb_model.joblib"
    SCALER_PATH = BASE_DIR / "models" / f"{args.out_prefix}_scaler.joblib"

    model: XGBRegressor = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    # 검증용 X, y 생성
    X_val, y_val = build_dataset_for_validation(PROCESSED_PATH, feat_cfg)

    # 스케일 transform (훈련 기준)
    X_val = X_val.copy()
    float_cols = getattr(scaler, "feature_names_in_", None)
    if float_cols is not None:
        cols_to_scale = [c for c in float_cols if c in X_val.columns]
        if cols_to_scale:
            X_val.loc[:, cols_to_scale] = scaler.transform(X_val[cols_to_scale])

    # 예측 & MAE 계산
    y_pred_np = model.predict(X_val)
    y_pred = np.array([float(x) for x in y_pred_np])
    mae = mean_absolute_error(y_val, y_pred)

    print("\n✅ df_part2 전체 데이터에 대한 예측 완료")
    print(f"  - 예측 시점: t+{horizon} ({int(horizon) * 5}분 후)")
    print(f"  - 검증 MAE: {mae:.4f}")

    # DB 설정 파일 로드 → 저장
    db_config = load_db_config(args.db_config)
    save_validation_results_to_db(mae, y_val, y_pred, int(horizon), db_config)

    # ===== 시각화 코드 시작 =====
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

    fig = plt.gcf() # 현재 figure 객체
    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    print(f"plot 저장 완료 : {outfile}")

    plt.show()
    # ===== plotly 시각화 코드 끝 =====


if __name__ == "__main__":
    main()