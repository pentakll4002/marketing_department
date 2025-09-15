import pandas as pd

def extract(path: str = "data/raw/marketing.csv"):
    df = pd.read_csv(path)
    return df
