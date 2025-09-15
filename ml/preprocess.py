import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def preprocess_for_ml(df):
    sc = StandardScaler()
    df_scaled = sc.fit_transform(df)
    return df_scaled, sc

def apply_pca(df_scaled, n_components=2):
    pca = PCA(n_components=n_components)
    principal_comp = pca.fit_transform(df_scaled)
    return principal_comp, pca
