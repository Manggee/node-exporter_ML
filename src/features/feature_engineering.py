import datetime
from pathlib import Path

import holidays
import numpy as np
import pandas as pd

from src.data.load_data import load_and_prepare
from src.data.preprocess import preprocess_load_metric


def make_features(df: pd.DataFrame, lags=[1,2,3,4,5], rolling_window=[3,6], use_all_stats: bool = True, use_holidays: bool = True, use_fourier:bool = True) -> pd.DataFrame:
    """
    시계열 예측을 위한 lag/rolling 및 시간 파생 피처를 생성한다.
    - preprocess.py 출력(mean/max/min/std) 또는 단일 'value' 컬럼을 모두 지원
    - 인덱스가 DatetimeIndex가 아니면 자동 변환
    - 2025-06-17 이전 데이터는 제거 (2행)
    :param df: pd.DataFrame -> Datetime 인덱스이거나 'timestamp' 컬럼을 가진 데이터프레임 (컬럼: mean/max/min/std 혹은 value 중 일부)
    :param lags: list[int] -> 몇 시간 전 값을 lag 피처로 만들지
    :param rolling_window: list[int] -> 이동 평균 window 크기 (샘플 수 기준)
    :param use_all_stats: bool -> 모든 통계 피처를 사용할지 여부
    :param use_holidays: bool -> 공휴일 피처를 사용할지 여부
    :param use_fourier: bool -> 푸리에 변환 피처를 사용할지 여부
    :return: pd.DataFrame -> 원본 지표 컬럼 + 파생 피처(lag/rolling) + 시간 피처(hour/weekday)가 포함된 df
    """

    df_feat = df.copy()

    # 1. 인덱스 정리 (인덱스를 반드시 DatetimeIndex로 변환하는 작업)
    if "timestamp" in df_feat.columns:
        df_feat["timestamp"] = pd.to_datetime(df_feat["timestamp"])
        df_feat = df_feat.set_index("timestamp")
    elif not isinstance(df_feat.index, pd.DatetimeIndex):  # isinstance(a, b): a가 b타입인지 확인하는 파이썬 함수
        df_feat.index = pd.to_datetime(df_feat.index)

    # 2. 예측에 쓸 대상 컬럼 선택 (preprocess 출력 우선)
    stats_cols = ["mean", "max", "min", "std"]
    #  모든 컬럼 존재
    if use_all_stats and all(col in df_feat.columns for col in stats_cols):
        target_cols = stats_cols
    # value 컬럼만 존재
    elif "value" in df_feat.columns:
        target_cols = ["value"]
    #  mean(평균) 컬럼만 존재
    elif "mean" in df_feat.columns:
        target_cols = ["mean"]
    else:
        raise ValueError("파생 피처 생성 대상 컬럼을 찾지 못했습니다. (⚠️ value 또는 mean/max/min/std 필요)")

    # 3. lag/rolling mean 생성
    for col in target_cols:  # mean, max, min, std
        for lag in lags:  # 1,2,3,4,5
            df_feat[f"{col}_lag_{lag}"] = df_feat[col].shift(lag)  # 과거값 (1행 전, 2행 전...)
        for w in rolling_window:  # [3,6]
            df_feat[f"{col}_rolling_mean_{w}"] = df_feat[col].rolling(window=w,
                                                                      min_periods=w).mean()  # min_periods=w : w(구간크기)만큼 값이 모이기 전까지는 NaN 유지(초기 결측 최소화)

    # 4. 시간 파생 피처
    df_feat["hour"] = df_feat.index.hour
    df_feat["weekday"] = df_feat.index.weekday
    df_feat['month'] = df_feat.index.month

    # 5. 새로운 피처 추가 (옵션에 따라) 👇
    if use_holidays:
        kr_holidays = holidays.Korea()
        df_feat['is_weekend'] = (df_feat.index.weekday >= 5).astype(int)  # bool을 int로 변환
        df_feat['is_holiday'] = pd.Series(df_feat.index.date).isin(kr_holidays).values.astype(int)  # bool을 int로 변환

    if use_fourier and 'mean' in df_feat.columns:
        fft_result = np.fft.fft(df_feat['mean'].values)
        fft_freq = np.fft.fftfreq(len(df_feat))
        top_freq_indices = np.argsort(np.abs(fft_result))[-3:]
        for i, idx in enumerate(top_freq_indices):
            df_feat[f'fft_real_{i + 1}'] = np.cos(2 * np.pi * fft_freq[idx] * np.arange(len(df_feat)))
            df_feat[f'fft_imag_{i + 1}'] = np.sin(2 * np.pi * fft_freq[idx] * np.arange(len(df_feat)))

    # 6. 2025-06-17 이전 데이터 제거 (초기 NaN과 함께 필터)
    df_feat = df_feat[df_feat.index.date >= datetime.date(2025, 6, 17)]

    # 7. NaN 제거 (초기 lag/rolling 에서 발생하는 NaN)
    df_feat = df_feat.dropna()

    return df_feat


# ===== 디버깅 실행 =====
if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_PATH = BASE_DIR / "data" / "node_metrics.csv"

    print("📥 데이터 로드 중...")
    df_raw = load_and_prepare(DATA_PATH)
    print("✅ 원본 데이터 크기:", df_raw.shape)

    print("\n🔄 전처리 중...")
    df_preprocessed = preprocess_load_metric(df_raw)
    print("✅ 전처리 결과 크기:", df_preprocessed.shape)

    print("\n🛠 피처 엔지니어링 중...")
    df_features = make_features(
        df_preprocessed,
        lags=[1, 2, 3],
        rolling_window=[3, 6],
        use_all_stats=True,
        use_holidays=True,
        use_fourier=True
    )
    print("✅ 최종 피처 크기:", df_features.shape)
    print(df_features.head())

    print("\n🧪 결측치 개수:", df_features.isna().sum().sum())
    print("📊 컬럼 목록:", df_features.columns.tolist())

