from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

from etl.pipeline import run_etl
from ml.train import train_model

with DAG(
    dag_id="marketing_pipeline",
    start_date=datetime(2025, 1, 1),
    schedule_interval="@daily",
    catchup=False,
    tags=["marketing", "mlflow"],
) as dag:
    
    etl_task = PythonOperator(
        task_id="etl",
        python_callable=run_etl,
    )

    train_task = PythonOperator(
        task_id="train",
        python_callable=train_model,
    )

    etl_task >> train_task
