import sys
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


class PredictPipeline:
    def __init__(self, model_path: str | None = None, scaler_path: str | None = None):
        trainer_config = ModelTrainerConfig()
        transform_config = DataTransformationConfig()
        self.model_path = model_path or trainer_config.model_path
        self.scaler_path = scaler_path or transform_config.scaler_obj_file_path

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        try:
            if features.empty:
                raise CustomException("Input features dataframe is empty", sys)

            logging.info("Loading scaler from %s", self.scaler_path)
            scaler = load_object(self.scaler_path)
            logging.info("Loading KMeans model from %s", self.model_path)
            model = load_object(self.model_path)

            prepared_features = self._prepare_features(features)
            scaled = scaler.transform(prepared_features)
            labels = model.predict(scaled)

            output = features.copy()
            output["cluster"] = labels
            return output

        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
        try:
            work_df = df.copy()
            if "CUST_ID" in work_df.columns:
                work_df = work_df.drop(columns=["CUST_ID"])
            numeric_cols = work_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) != work_df.shape[1]:
                raise CustomException("All features must be numeric for scaling", sys)
            return work_df[numeric_cols]
        except Exception as prep_error:
            raise CustomException(prep_error, sys)


@dataclass
class CustomData:
    customer_id: str
    balance: float
    balance_frequency: float
    purchases: float
    oneoff_purchases: float
    installments_purchases: float
    cash_advance: float
    purchases_frequency: float
    oneoff_purchases_frequency: float
    purchases_installments_frequency: float
    cash_advance_frequency: float
    cash_advance_trx: float
    purchases_trx: float
    credit_limit: float
    payments: float
    minimum_payments: float
    prc_full_payment: float
    tenure: int

    def get_data_as_data_frame(self) -> pd.DataFrame:
        try:
            data: Dict[str, List] = {
                "CUST_ID": [self.customer_id],
                "BALANCE": [self.balance],
                "BALANCE_FREQUENCY": [self.balance_frequency],
                "PURCHASES": [self.purchases],
                "ONEOFF_PURCHASES": [self.oneoff_purchases],
                "INSTALLMENTS_PURCHASES": [self.installments_purchases],
                "CASH_ADVANCE": [self.cash_advance],
                "PURCHASES_FREQUENCY": [self.purchases_frequency],
                "ONEOFF_PURCHASES_FREQUENCY": [self.oneoff_purchases_frequency],
                "PURCHASES_INSTALLMENTS_FREQUENCY": [self.purchases_installments_frequency],
                "CASH_ADVANCE_FREQUENCY": [self.cash_advance_frequency],
                "CASH_ADVANCE_TRX": [self.cash_advance_trx],
                "PURCHASES_TRX": [self.purchases_trx],
                "CREDIT_LIMIT": [self.credit_limit],
                "PAYMENTS": [self.payments],
                "MINIMUM_PAYMENTS": [self.minimum_payments],
                "PRC_FULL_PAYMENT": [self.prc_full_payment],
                "TENURE": [self.tenure],
            }
            return pd.DataFrame(data)
        except Exception as e:
            raise CustomException(e, sys)
