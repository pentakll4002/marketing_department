import sys

import mlflow

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging


class TrainPipeline:
    def __init__(
        self,
        tracking_uri: str = "sqlite:///mlflow.db",
        experiment_name: str = "MarketingClustering",
        run_name: str = "kmeans_pipeline",
    ):
        self.data_ingestion = DataIngestion()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()
        self.run_name = run_name

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

    def initiate_training_pipeline(self):
        try:
            logging.info("Launching marketing clustering pipeline")
            processed_path = self.data_ingestion.initiate_data_ingestion()
            scaled_array, numeric_df, scaler = self.data_transformation.initiate_data_transformation(processed_path)
            with mlflow.start_run(run_name=self.run_name):
                mlflow.log_param("n_clusters", self.model_trainer.n_clusters)
                mlflow.log_param("max_clusters", self.model_trainer.max_clusters)
                mlflow.log_param("num_features", numeric_df.shape[1])
                artifacts = self.model_trainer.initiate_model_trainer(
                    scaled_array=scaled_array,
                    original_df=numeric_df,
                    scaler=scaler,
                )
                mlflow.log_param("data_path", processed_path)
            logging.info("Training pipeline completed successfully and logged to MLflow")
            return artifacts
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    pipeline = TrainPipeline()
    pipeline.initiate_training_pipeline()