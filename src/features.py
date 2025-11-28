import pandas as pd

def add_age(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create 'edad' feature from 'dob' and 'trans_date_trans_time'.
    """
    if 'dob' in df.columns and 'trans_date_trans_time' in df.columns:
        df['edad'] = (df['trans_date_trans_time'] - df['dob']).dt.days // 365
        df = df.drop(columns='dob')
    return df


def add_transaction_month(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract month from transaction date.
    """
    if 'trans_date_trans_time' in df.columns:
        df['trans_month'] = df['trans_date_trans_time'].dt.month
        df = df.drop(columns='trans_date_trans_time')
    return df


def drop_identifiers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop non-predictive identifiers like transaction number.
    """
    if 'trans_num' in df.columns:
        df = df.drop(columns='trans_num')
    return df


def drop_highly_correlated(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop merchant coordinates that are highly correlated with client coordinates.
    """
    if 'merch_lat' in df.columns:
        df = df.drop(columns='merch_lat')
    if 'merch_long' in df.columns:
        df = df.drop(columns='merch_long')
    return df


def feature_engineering_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature engineering pipeline:
    - Add age
    - Add transaction month
    - Drop identifiers
    - Drop highly correlated features
    """
    df = add_age(df)
    df = add_transaction_month(df)
    df = drop_identifiers(df)
    df = drop_highly_correlated(df)
    return df
