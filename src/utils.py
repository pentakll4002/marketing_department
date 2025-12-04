import os
import sys

import dill
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from src.exception import CustomException
from src.logger import logging


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
        logging.info(f"Saved object to {file_path}")

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(
    X_train,
    y_train,
    X_test,
    y_test,
    models: dict,
    params: dict = None,
    search_method: str = "grid",
    n_iter: int = 20,
):
    try:
        report = {}

        for name, model in models.items():
            logging.info(f"--- Evaluating model: {name} ---")

            param_grid = params.get(name, {}) if params else {}

            if search_method == "random" and len(param_grid) > 0:
                logging.info(f"Using RandomizedSearchCV for {name}")
                search = RandomizedSearchCV(
                    estimator=model,
                    param_distributions=param_grid,
                    n_iter=n_iter,
                    cv=3,
                    n_jobs=-1,
                    random_state=42,
                    scoring="r2",
                )
            else:
                logging.info(f"Using GridSearchCV for {name}")
                search = GridSearchCV(
                    estimator=model,
                    param_grid=param_grid,
                    cv=3,
                    n_jobs=-1,
                    scoring="r2",
                )

            search.fit(X_train, y_train)

            best_model = search.best_estimator_
            logging.info(f"Best params for {name}: {search.best_params_}")

            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            mae = mean_absolute_error(y_test, y_test_pred)
            mse = mean_squared_error(y_test, y_test_pred)
            rmse = np.sqrt(mse)

            report[name] = {
                "best_params": search.best_params_,
                "best_model": best_model,
                "train_r2": train_r2,
                "test_r2": test_r2,
                "mae": mae,
                "mse": mse,
                "rmse": rmse,
            }

        return report

    except Exception as e:
        raise CustomException(e, sys)
