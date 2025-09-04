from pathlib import Path

import pandas as pd
import ast

# 기본 경로 정의
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "node_metrics.csv"

def load_and_prepare(filepath: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(filepath)

    # labels 컬럼을 딕셔너리로 파싱
    df['labels_dict'] = df['labels'].apply(ast.literal_eval)
    labels_df = df['labels_dict'].apply(pd.Series)
    df_expanded = pd.concat([df.drop(columns=['labels', 'labels_dict']), labels_df], axis=1)

    # 타입 변환
    df_expanded['timestamp'] = pd.to_datetime(df_expanded['timestamp']) # timestamp 컬럼을 datetime으로 변환
    for col in ['metric', 'cpu', 'mode', 'device', 'device_error', 'fstype', 'mountpoint']:
        if col in df_expanded.columns:
            df_expanded[col] = df_expanded[col].astype('category') # 지정된 컬럼이 존재하면 범주형으로 변환

    # float64 → float32로 다운캐스팅 (메모리 절약, 예측 정확도에 영향이 생기는 경우 float64로 유지)
    df_expanded['value'] = pd.to_numeric(df_expanded['value'], downcast='float')

    return df_expanded

# 디버깅용 실행 코드
if __name__ == "__main__":
    from preprocess import preprocess_load_metric  # 전처리 함수 가져오기

    # 1. 데이터 로딩
    df = load_and_prepare("../node_metrics.csv")  # 경로는 필요 시 수정

    # 2. 전처리 실행
    df_preprocessed = preprocess_load_metric(df)

    # 3. 디버깅 출력
    print("✅ 전처리 결과 샘플:")
    print(df_preprocessed.head(10))

    print("\n🧠 인덱스 간격:")
    print(df_preprocessed.index.to_series().diff().dropna().unique())

    print("\n🔍 결측치 존재 여부:", df_preprocessed.isna().sum().to_dict())