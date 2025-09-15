FROM apache/airflow:2.9.2-python3.10
COPY requirements.txt .
RUN pip install uv && uv pip install -r requirements.txt
COPY dags /opt/airflow/dags
