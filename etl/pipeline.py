from etl.load import load_data
from etl.transform import clean_data
from ml.preprocess import preprocess_for_ml, apply_pca
from ml.train import train_kmeans, find_optimal_clusters, build_autoencoder, train_autoencoder
from ml.evaluate import evaluate_kmeans, plot_optimal_clusters_scores, plot_correlation_heatmap, plot_cluster_histograms, predict_and_visualize_autoencoder_pca

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

def run_pipeline(file_path='data/raw/marketing_data.csv'):
    # ETL Process
    df = load_data(file_path)
    df_cleaned = clean_data(df.copy())
    df_scaled, sc = preprocess_for_ml(df_cleaned.copy())

    # ML Process - KMeans
    scores_1 = find_optimal_clusters(df_scaled)
    print("KMeans optimal clusters scores:", scores_1)
    plot_optimal_clusters_scores(scores_1, title='Finding the right number of clusters for KMeans')

    # Let's choose 8 clusters based on the elbow method for the first run
    kmeans_model, labels = train_kmeans(df_scaled, n_clusters=8)
    cluster_centers_df = evaluate_kmeans(kmeans_model, sc, df_cleaned.columns)
    print("KMeans Cluster Centers (Inverse Transformed):")
    print(cluster_centers_df)

    # Add cluster labels to the original dataframe for further analysis and visualization
    df_cluster = pd.concat([df_cleaned, pd.DataFrame({'cluster':labels})], axis = 1)
    
    # Visualization of KMeans clusters
    plot_cluster_histograms(df_cluster, df_cleaned.columns)

    # ML Process - Autoencoder and PCA
    input_dim = df_scaled.shape[1]
    autoencoder, encoder = build_autoencoder(input_dim)
    train_autoencoder(autoencoder, df_scaled)
    
    # Predict with encoder and apply PCA
    pca_df = predict_and_visualize_autoencoder_pca(encoder, df_cleaned, labels)
    print("PCA with Autoencoder results:")
    print(pca_df.head())

    # Correlation Heatmap
    plot_correlation_heatmap(df_cleaned)

if __name__ == "__main__":
    run_pipeline()
