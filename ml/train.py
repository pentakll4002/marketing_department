from sklearn.cluster import KMeans
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.optimizers import Adam

def train_kmeans(df_scaled, n_clusters=8, random_state=42):
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')
    kmeans.fit(df_scaled)
    labels = kmeans.labels_
    return kmeans, labels

def find_optimal_clusters(df_scaled, max_clusters=20):
    scores = []
    for i in range(1, max_clusters):
        kmeans = KMeans(n_clusters=i, random_state=42, n_init='auto')
        kmeans.fit(df_scaled)
        scores.append(kmeans.inertia_)
    return scores

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

def train_autoencoder(autoencoder, df_scaled, batch_size=128, epochs=25):
    autoencoder.fit(df_scaled, df_scaled, batch_size = batch_size, epochs = epochs,  verbose = 1)
    return autoencoder
