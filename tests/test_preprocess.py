import pandas as pd
import numpy as np
import pytest
from src.preprocess import preprocess_load_metric

@pytest.fixture
def sample_df():
    """테스트용 샘플 데이터프레임 생성"""
    timestamps = pd.date_range(start="2025-06-17 00:00", periods=10, freq="2min")
    data = pd.DataFrame({
        "timestamp": timestamps.tolist() * 2,  # 중복 시간대 존재
        "metric": ["node_load1"] * 20,
        "value": np.linspace(1, 20, 20)
    })
    return data

def test_returns_dataframe(sample_df):
    """반환값이 DataFrame인지 확인"""
    result = preprocess_load_metric(sample_df)
    assert isinstance(result, pd.DataFrame)

def test_columns_exist(sample_df):
    """mean, max, min, std 컬럼이 존재하는지 확인"""
    result = preprocess_load_metric(sample_df)
    for col in ["mean", "max", "min", "std"]:
        assert col in result.columns

def test_index_is_datetime(sample_df):
    """인덱스가 datetime 형식인지 확인"""
    result = preprocess_load_metric(sample_df)
    assert pd.api.types.is_datetime64_any_dtype(result.index)

def test_no_missing_values(sample_df):
    """결측치가 없는지 확인"""
    result = preprocess_load_metric(sample_df)
    assert result.isna().sum().sum() == 0

def test_aggregation_correctness(sample_df):
    """5분 단위 리샘플링이 잘 되었는지 확인"""
    result = preprocess_load_metric(sample_df)
    time_deltas = result.index.to_series().diff().dropna().unique()
    assert all(td == pd.Timedelta("5min") for td in time_deltas)