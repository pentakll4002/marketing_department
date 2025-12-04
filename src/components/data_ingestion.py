import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..exception import CustomException
from ..logger import logging


@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "raw_marketing_data.csv")
    processed_data_path: str = os.path.join("artifacts", "marketing_data_clean.csv")


class DataIngestion:
    def __init__(self, source_path: str | None = None):
        self.data_ingestion_config = DataIngestionConfig()
        self.source_path = source_path or os.path.join("data", "marketing_data.csv")

    def _basic_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle duplicates and missing numeric values to stabilise clustering."""
        df = df.copy()
        df.replace({"": np.nan}, inplace=True)
        df.drop_duplicates(inplace=True)

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())

        if "MINIMUM_PAYMENTS" in df.columns and df["MINIMUM_PAYMENTS"].isna().any():
            df["MINIMUM_PAYMENTS"] = df["MINIMUM_PAYMENTS"].fillna(
                df["MINIMUM_PAYMENTS"].median()
            )

        return df

    def initiate_data_ingestion(self) -> str:
        logging.info("Starting marketing data ingestion component")

        try:
            if not os.path.exists(self.source_path):
                raise CustomException(f"Dataset not found at {self.source_path}", sys)

            df = pd.read_csv(self.source_path)
            logging.info("Raw dataset shape: %s", df.shape)

            os.makedirs(os.path.dirname(self.data_ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.data_ingestion_config.raw_data_path, index=False)
            logging.info("Stored raw dataset at %s", self.data_ingestion_config.raw_data_path)

            cleaned_df = self._basic_cleaning(df)
            cleaned_df.to_csv(self.data_ingestion_config.processed_data_path, index=False)
            logging.info(
                "Stored cleaned dataset at %s with shape %s",
                self.data_ingestion_config.processed_data_path,
                cleaned_df.shape,
            )

            return self.data_ingestion_config.processed_data_path

        except Exception as e:
            raise CustomException(e, sys)

if __name__=="__main__":
    obj = DataIngestion()
    processed_path = obj.initiate_data_ingestion()
    logging.info("Processed dataset stored at %s", processed_path)