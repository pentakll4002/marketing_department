import os
import sys
from dataclasses import dataclass
from typing import List

import pandas as pd
from sklearn.cluster import KMeans

from ..exception import CustomException
from ..logger import logging
from ..utils import save_object


@dataclass
class ModelTrainerConfig:
    model_path: str = os.path.join("artifacts", "marketing_kmeans.pkl")
    cluster_centers_path: str = os.path.join("artifacts", "cluster_centers.csv")
    clustered_data_path: str = os.path.join("artifacts", "clustered_marketing_data.csv")


class ModelTrainer:
    def __init__(self, n_clusters: int = 8, max_clusters: int = 20):
        self.model_trainer_config = ModelTrainerConfig()
        self.n_clusters = n_clusters
        self.max_clusters = max_clusters

    def initiate_model_trainer(self, scaled_array, original_df: pd.DataFrame, scaler):
        try:
            logging.info("Starting KMeans model training")

            inertia_scores: List[float] = []
            for i in range(1, self.max_clusters):
                kmeans = KMeans(n_clusters=i, random_state=42, n_init="auto")
                kmeans.fit(scaled_array)
                inertia_scores.append(kmeans.inertia_)

            kmeans_final = KMeans(n_clusters=self.n_clusters, random_state=42, n_init="auto")
            labels = kmeans_final.fit_predict(scaled_array)
            save_object(self.model_trainer_config.model_path, kmeans_final)
            logging.info("Saved trained KMeans model at %s", self.model_trainer_config.model_path)

            cluster_centers = scaler.inverse_transform(kmeans_final.cluster_centers_)
            os.makedirs(os.path.dirname(self.model_trainer_config.cluster_centers_path), exist_ok=True)
            cluster_centers_df = pd.DataFrame(
                cluster_centers,
                columns=original_df.columns,
            )
            cluster_centers_df.to_csv(self.model_trainer_config.cluster_centers_path, index=False)
            logging.info("Saved cluster centers at %s", self.model_trainer_config.cluster_centers_path)

            labeled_df = original_df.copy()
            labeled_df["cluster"] = labels
            os.makedirs(os.path.dirname(self.model_trainer_config.clustered_data_path), exist_ok=True)
            labeled_df.to_csv(self.model_trainer_config.clustered_data_path)
            logging.info("Stored clustered dataset at %s", self.model_trainer_config.clustered_data_path)

            return {
                "model_path": self.model_trainer_config.model_path,
                "cluster_centers_path": self.model_trainer_config.cluster_centers_path,
                "clustered_data_path": self.model_trainer_config.clustered_data_path,
                "inertia_scores": inertia_scores,
            }

        except Exception as e:
            raise CustomException(e, sys)