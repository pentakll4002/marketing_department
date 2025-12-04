import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ..exception import CustomException
from ..logger import logging
from ..utils import save_object


@dataclass
class DataTransformationConfig:
    scaler_obj_file_path: str = os.path.join("artifacts", "scaler.pkl")
    transformed_data_path: str = os.path.join("artifacts", "marketing_data_scaled.npy")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    @staticmethod
    def _select_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        df = df.copy()
        if "CUST_ID" in df.columns:
            df.set_index("CUST_ID", inplace=True)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        return df[numeric_cols], numeric_cols

    def initiate_data_transformation(self, data_path: str):
        try:
            logging.info("Starting data transformation for clustering")

            df = pd.read_csv(data_path)
            numeric_df, numeric_cols = self._select_features(df)
            logging.info("Using %d numeric features for clustering", len(numeric_cols))

            scaler = StandardScaler()
            scaled_array = scaler.fit_transform(numeric_df)

            os.makedirs(os.path.dirname(self.data_transformation_config.transformed_data_path), exist_ok=True)
            np.save(self.data_transformation_config.transformed_data_path, scaled_array)
            save_object(self.data_transformation_config.scaler_obj_file_path, scaler)
            logging.info("Saved scaler and transformed array to artifacts directory")

            return scaled_array, numeric_df, scaler

        except Exception as e:
            raise CustomException(e, sys)
