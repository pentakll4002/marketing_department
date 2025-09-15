from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.optimizers import Adam
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

def build_autoencoder(input_dim, encoding_dim=10):
    input_df = Input(shape=(input_dim,))

    x = Dense(7, activation='relu')(input_df)
    x = Dense(500, activation='relu', kernel_initializer='glorot_uniform')(x)
    x = Dense(500, activation='relu', kernel_initializer='glorot_uniform')(x)
    x = Dense(2000, activation='relu', kernel_initializer='glorot_uniform')(x)

    encoded = Dense(encoding_dim, activation='relu', kernel_initializer='glorot_uniform')(x)

    x = Dense(2000, activation='relu', kernel_initializer='glorot_uniform')(encoded)
    x = Dense(500, activation='relu', kernel_initializer='glorot_uniform')(x)

    decoded = Dense(input_dim, activation='sigmoid', kernel_initializer='glorot_uniform')(x)

    autoencoder = Model(input_df, decoded)
    encoder = Model(input_df, encoded)

    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return autoencoder, encoder

def evaluate_kmeans(kmeans, sc, df_columns):
    cluster_centers = pd.DataFrame(data = kmeans.cluster_centers_, columns = [df_columns])
    cluster_centers = sc.inverse_transform(cluster_centers)
    cluster_centers = pd.DataFrame(data = cluster_centers, columns = [df_columns])
    return cluster_centers

def plot_optimal_clusters_scores(scores, title="Finding the right number of clusters", xlabel="Clusters", ylabel="Scores"):
    plt.plot(scores, 'bx-')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel) 
    plt.show()

def plot_correlation_heatmap(df):
    corr = df.corr()
    f, ax = plt.subplots(figsize = (20, 20))
    sns.heatmap(corr, annot = True)
    plt.show()

def plot_cluster_histograms(df_cluster, df_columns):
    for i in df_columns:
        plt.figure(figsize = (35, 5))
        for j in range(8):
            plt.subplot(1,8,j+1)
            cluster = df_cluster[df_cluster['cluster'] == j]
            cluster[i].hist(bins = 20)
            plt.title('{}    \nCluster {} '.format(i,j))
        plt.show()

def predict_and_visualize_autoencoder_pca(encoder, df_for_prediction, labels):
    pred = encoder.predict(df_for_prediction)
    
    pca = PCA(n_components=2)
    prin_comp = pca.fit_transform(pred)
    pca_df = pd.DataFrame(data = prin_comp, columns =['pca1','pca2'])
    
    pca_df = pd.concat([pca_df,pd.DataFrame({'cluster':labels})], axis = 1)

    plt.figure(figsize=(10,10))
    ax = sns.scatterplot(x="pca1", y="pca2", hue = "cluster", data = pca_df, palette =['red','green','blue','pink','yellow','gray','purple', 'black'])
    plt.show()
    return pca_df
