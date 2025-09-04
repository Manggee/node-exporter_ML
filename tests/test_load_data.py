import os
import pandas as pd
import numpy as np
from src.load_data import load_and_prepare

def test_node_metrics_file_exists():
    """node_metrics.csv 파일이 루트 디렉토리에 존재하는지 확인"""
    assert os.path.exists('node_metrics.csv'), (
        "node_metrics.csv 파일이 프로젝트 루트에 존재해야 합니다."
    )

def test_load_and_prepare_returns_dataframe():
    """함수 결과가 DataFrame 인스턴스인지 확인"""
    df = load_and_prepare()
    assert isinstance(df, pd.DataFrame)

def test_timestamp_column_is_datetime():
    """timestamp 컬럼이 datetime 형식으로 변환되었는지 확인"""
    df = load_and_prepare()
    assert pd.api.types.is_datetime64_any_dtype(df['timestamp'])

def test_value_column_dtype():
    """value 컬럼이 float32로 다운캐스팅되었는지 확인"""
    df = load_and_prepare()
    assert df['value'].dtype in [np.float32, np.float64]

def test_categorical_columns_if_present():
    """지정된 컬럼이 존재하면 범주형인지 확인"""
    df = load_and_prepare()
    for col in ['metric', 'cpu', 'mode', 'device', 'device_error', 'fstype', 'mountpoint']:
        if col in df.columns:
            assert isinstance(df[col].dtype, pd.CategoricalDtype)

