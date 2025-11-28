import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load dataset from CSV file.
    Args:
        file_path (str): Path to the dataset (e.g., 'data/fraud_data.csv').
    Returns:
        pd.DataFrame: Loaded dataset.
    """
    df = pd.read_csv(file_path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and fix inconsistencies in the dataset.
    - Convert date columns to datetime
    - Fix errors in 'is_fraud'
    - Cast columns to correct types
    """
    # Convert dates
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'], errors='coerce')
    df['dob'] = pd.to_datetime(df['dob'], errors='coerce')

    # Fix errors in target column
    df['is_fraud'] = df['is_fraud'].astype(str)
    df.loc[df['is_fraud'].str.startswith('1'), 'is_fraud'] = 1
    df.loc[df['is_fraud'].str.startswith('0'), 'is_fraud'] = 0
    df['is_fraud'] = df['is_fraud'].astype(int)

    # Cast numeric columns
    float_cols = ['amt', 'lat', 'long', 'merch_lat', 'merch_long']
    for col in float_cols:
        if col in df.columns:
            df[col] = df[col].astype(float)

    if 'city_pop' in df.columns:
        df['city_pop'] = df['city_pop'].astype(int)

    return df


def transform_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform features:
    - Create 'edad' from dob
    - Extract month from transaction date
    - Drop unnecessary columns
    """
    # Age from dob
    if 'dob' in df.columns:
        df['edad'] = (df['trans_date_trans_time'] - df['dob']).dt.days // 365
        df = df.drop(columns='dob')

    # Month from transaction date
    if 'trans_date_trans_time' in df.columns:
        df['trans_month'] = df['trans_date_trans_time'].dt.month
        df = df.drop(columns='trans_date_trans_time')

    # Drop transaction ID (not predictive)
    if 'trans_num' in df.columns:
        df = df.drop(columns='trans_num')

    # Drop highly correlated merchant coordinates
    if 'merch_lat' in df.columns:
        df = df.drop(columns='merch_lat')
    if 'merch_long' in df.columns:
        df = df.drop(columns='merch_long')

    return df


def scale_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scale numeric features using MinMaxScaler.
    """
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.drop('is_fraud')
    scaler = MinMaxScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical features using LabelEncoder.
    """
    cat_cols = df.select_dtypes(include=['object', 'string']).columns
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])
    return df


def preprocess_pipeline(file_path: str) -> pd.DataFrame:
    """
    Full preprocessing pipeline:
    - Load data
    - Clean data
    - Transform features
    - Scale numeric variables
    - Encode categorical variables
    """
    df = load_data(file_path)
    df = clean_data(df)
    df = transform_features(df)
    df = scale_numeric(df)
    df = encode_categorical(df)
    return df
