import pandas as pd
import numpy as np
import os

def remove_duplicates(df):
    return df.drop_duplicates()

def handle_missing_values(df):
    for column in df.columns:
        if df[column].dtype in ["int64","float64"]:
            df[column]=df[column].fillna(df[column].median())
        else:
            df[column] = df[column].fillna(df[column].mode()[0])
    return df
def detect_outliers(df):
    numeric_cols = df.select_dtypes(include=["int64","float64"]).columns
    for col in numeric_cols:
        q1 =  df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df[col] = df[col].clip(lower,upper)
    return df
def encode_categorical_variables(df):
    categorical_columns = df.select_dtypes(include=["object"]).columns
    df = pd.get_dummies(df,columns=categorical_columns, drop_first=True)
    return df
def data_consistency(df):
    df = df.replace([np.inf,-np.inf],np.nan)
    df = df.dropna()
    return df
def preprocess_data(df):
    df = remove_duplicates(df)
    df = handle_missing_values(df)
    df = detect_outliers(df)
    df = encode_categorical_variables(df)
    df = data_consistency(df)
    return df

    





