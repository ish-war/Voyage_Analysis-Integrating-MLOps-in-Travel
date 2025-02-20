from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

# Import utility scripts
from dags.utils.data_ingestion import DataLoader
from dags.utils.data_transformation import DataTransformer
from dags.utils.model_training import RandomForestModel

# Define file paths
data_file_path = '/opt/airflow/dags/data/flights.csv'

# Define default args for Airflow DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

# Define the DAG
dag = DAG(
    dag_id='flight_price_prediction_dag',
    default_args=default_args,
    description='A DAG for flight price prediction using RandomForest',
    schedule_interval='@daily',
    catchup=False
)

# Function to load data
def load_data():
    data_loader = DataLoader(data_file_path)
    return data_loader.load_data()

# Function to transform data
def transform_data():
    data_loader = DataLoader(data_file_path)
    data_transformer = DataTransformer(data_loader.load_data())
    return data_transformer.transform()

# Function to train model
def train_model():
    data_loader = DataLoader(data_file_path)
    data_transformer = DataTransformer(data_loader.load_data())
    X, Y = data_transformer.transform()
    model = RandomForestModel(X, Y)
    return model.random_forest()

# Define Airflow Tasks
load_data_task = PythonOperator(
    task_id='load_data_task',
    python_callable=load_data,
    dag=dag
)

transform_data_task = PythonOperator(
    task_id='transform_data_task',
    python_callable=transform_data,
    dag=dag
)

random_forest_task = PythonOperator(
    task_id='random_forest_task',
    python_callable=train_model,
    dag=dag
)

# Define Task Order
load_data_task >> transform_data_task >> random_forest_task
