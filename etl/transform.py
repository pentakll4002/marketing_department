import pandas as pd
import warnings

warnings.filterwarnings('ignore')

def clean_data(df):
    df['MINIMUM_PAYMENTS'].fillna(df['MINIMUM_PAYMENTS'].median(), inplace=True)
    df['CREDIT_LIMIT'].fillna(df['CREDIT_LIMIT'].median(), inplace=True)
    df.drop("CUST_ID", axis = 1, inplace= True)
    return df
