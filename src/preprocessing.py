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
    - Remove invalid or NaN rows in critical columns
    """
    # Convert dates
    for col in ['trans_date_trans_time', 'dob']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # Derive is_fraud from type and isFraud
    if 'type' in df.columns and 'isFraud' in df.columns:
        df = df.dropna(subset=['type', 'isFraud'])
        df['is_fraud'] = df['type'].isin(['TRANSFER', 'CASH_OUT']) & (df['isFraud'] == 1)
        df['is_fraud'] = df['is_fraud'].astype(int)
    elif 'is_fraud' in df.columns:
        df['is_fraud'] = pd.to_numeric(df['is_fraud'], errors='coerce')

    # Cast numeric columns safely
    float_cols = ['amt', 'lat', 'long', 'merch_lat', 'merch_long']
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'city_pop' in df.columns:
        df['city_pop'] = pd.to_numeric(df['city_pop'], errors='coerce')

    # Drop rows with NaN in critical columns
    critical_cols = ['amt', 'is_fraud']
    df = df.dropna(subset=[col for col in critical_cols if col in df.columns])

    return df


def transform_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform features:
    - Create 'edad' from dob
    - Extract month from transaction date
    - Drop unnecessary columns
    """
    if 'dob' in df.columns and 'trans_date_trans_time' in df.columns:
        df['edad'] = (df['trans_date_trans_time'] - df['dob']).dt.days // 365
        df = df.drop(columns='dob')

    if 'trans_date_trans_time' in df.columns:
        df['trans_month'] = df['trans_date_trans_time'].dt.month
        df = df.drop(columns='trans_date_trans_time')

    for col in ['trans_num', 'merch_lat', 'merch_long']:
        if col in df.columns:
            df = df.drop(columns=col)

    return df


def scale_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scale numeric features using MinMaxScaler.
    """
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    num_cols = [col for col in num_cols if col != 'is_fraud']
    if num_cols:
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
        df[col] = df[col].fillna("missing")
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
