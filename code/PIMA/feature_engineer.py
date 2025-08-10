import pandas as pd
import numpy as np

def feature_engineer(df):
    df = df.copy()
    # Pima-like engineering
    if 'Glucose' in df.columns and 'BMI' in df.columns:
        df['glucose_x_bmi'] = df['Glucose'] * df['BMI']
    if 'Age' in df.columns and 'Glucose' in df.columns:
        df['glucose_by_age'] = df['Glucose'] / (df['Age'] + 1e-6)
    if 'BMI' in df.columns:
        df['bmi_sq'] = df['BMI']**2
    # safe numeric conversion for any object columns
    for c in df.columns:
        if df[c].dtype == 'object':
            try:
                df[c] = pd.to_numeric(df[c], errors='coerce')
            except:
                pass
    return df
