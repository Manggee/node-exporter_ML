import pandas as pd

def preprocess_load_metric(df: pd.DataFrame, target_metric: str = "node_load1") -> pd.DataFrame:
    """
    예측 대상 metric만 필터링하고 5분 단위로 리샘플링한 후 결측치를 보간한다.
    :param df: (pd.DataFrame): 전체 데이터프레임
    :param target_metric: (str) 예측할 메트릭 이름 (기본값: 'node_load1')
    :return: pd.DataFrame: timestamp-indexed, 5분 간격으로 정규화된 데이터프레임
    """

    # 1. 해당 metric(node_load1)만 필터링
    df_filtered = df[df["metric"] == target_metric].copy()

    df_filtered = df_filtered[df_filtered["timestamp"] >= "2025-06-17"]

    # 2. timestamp 정렬
    df_filtered = df_filtered.sort_values("timestamp")
    df_filtered["timestamp"] = pd.to_datetime(df_filtered["timestamp"]) # datetime으로 타입 변환

    # 3. 인덱스 설정
    df_filtered.set_index("timestamp", inplace=True)

    # 4. 통계값 추출 (평균, 최대, 최소, 표준편차)
    agg_df = df_filtered["value"].resample("5min").agg(["mean", "max", "min", "std"])
    agg_df = agg_df.fillna(0) # 단일 샘플 구간 등으로 생기는 NaN 제거

    return agg_df


# 디버깅용 실행 코드
# if __name__ == "__main__":
#     from load_data import load_and_prepare
#
#     df_raw = load_and_prepare("../node_metrics.csv")
#     df_processed = preprocess_load_metric(df_raw)
#
#     print("📊 전처리된 통계 시계열 샘플:")
#     print(df_processed.head(10))
#
#     print("\n🧪 결측치 존재 여부:", df_processed.isna().sum().to_dict())
#     print("\n📈 인덱스 간격:", df_processed.index.to_series().diff().dropna().unique())