# Flight Price Prediction using Apache Airflow

## Web Interface

![voyage-analysis](https://github.com/user-attachments/assets/af81c08d-10c1-4355-98dd-6b5cb54ba9cb)

## 📌 Project Overview
This project aims to build an automated pipeline for **Flight Price Prediction** using **Apache Airflow**. The pipeline handles data ingestion, transformation, and machine learning model training with **RandomForestRegressor**. The project is containerized using **Docker** and orchestrated with **Apache Airflow**, making it scalable and efficient for continuous workflow automation.

## 🚀 Features
- **Automated Workflow**: Uses Apache Airflow DAGs for scheduling and automation.
- **Data Pipeline**: Extracts, transforms, and loads (ETL) flight data.
- **Machine Learning Model**: Trains a **RandomForestRegressor** for price prediction.
- **Dockerized Deployment**: Fully containerized using **Docker Compose**.
- **Task Orchestration**: Ensures streamlined execution with task dependencies.
- **Modular Codebase**: Organized structure with separate modules for ingestion, transformation, and model training.

## 📁 Project Structure
```
📂 flight-price-prediction
│── docker-compose.yaml       # Docker Compose configuration
│── dags/
│   │── flight_price_dag.py   # Main Airflow DAG
│   ├── utils/
│   │   ├── data_ingestion.py # Data loading script
│   │   ├── data_transformation.py # Data preprocessing
│   │   ├── model_training.py # ML model training
│   ├── data/
│   │   ├── flights.csv       # Flight dataset
│── logs/                     # Airflow logs
│── plugins/                  # Airflow plugins (if any)
```

## 🛠️ Setup and Installation
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/ish-war/voyage_analysis-Integrating-MLOps-in-Travel
cd flight-price-prediction
```

### 2️⃣ Install Docker and Docker Compose
Ensure **Docker** and **Docker Compose** are installed:
- [Install Docker](https://docs.docker.com/get-docker/)
- [Install Docker Compose](https://docs.docker.com/compose/install/)

### 3️⃣ Start the Airflow Services
Run the following command to initialize and start Airflow:
```bash
docker-compose up -d
```
This will start the **Redis, Airflow Scheduler, and Airflow Webserver** containers.

### 4️⃣ Access the Airflow Web UI
Once the services are running, open your browser and go to:
```
http://localhost:8080
```
Use the default credentials:
```
Username: airflow
Password: airflow
```

### 5️⃣ Trigger the DAG
- Navigate to **DAGs** in the Airflow UI.
- Enable and trigger **flight_price_prediction_dag**.

## 🧑‍💻 Usage
- Modify `flights.csv` in `dags/data/` for new datasets.
- Update **data processing scripts** inside `dags/utils/`.
- Train new models by adjusting **hyperparameters** in `model_training.py`.

## 🔥 Troubleshooting
### 1. **Invalid Login in Airflow UI**
If you cannot log in, create an admin user manually:
```bash
docker exec -it airflow_webserver bash
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin
```
Then restart Airflow:
```bash
docker-compose down && docker-compose up -d
```
Login using:
```
Username: admin
Password: admin
```

### 2. **Container Mount Error**
Ensure `dags/`, `logs/`, and `plugins/` directories exist before running `docker-compose`:
```bash
mkdir -p dags/data logs plugins
```

## 📜 License
This project is licensed under the **MIT License**.

## 🤝 Contributing
Contributions are welcome! Feel free to fork this repository and submit a pull request.

## 📬 Contact
For any questions or issues, reach out via [GitHub Issues](https://github.com/ish-war/voyage_analysis-Integrating-MLOps-in-Travel/issues).


