# IVF-Trigger-Day-Decision-Support-System
An end-to-end Machine Learning system that predicts whether a given stimulation day is a Good Day for Trigger or Not a Good Day in an IVF (In Vitro Fertilization) cycle based on patient hormone levels and ultrasound parameters.

This project demonstrates a production-style ML pipeline covering data ingestion, preprocessing, validation, training, experiment tracking, deployment, and monitoring.

📌 Problem Statement

In IVF treatment, selecting the correct trigger day is a critical decision that directly affects egg maturity and treatment success. Traditionally, this decision is made manually by clinicians using hormone reports and follicle measurements, which can be time-consuming and subjective.

This project aims to assist clinicians by providing a data-driven decision support system that predicts trigger day suitability using machine learning.

🎯 Objectives
Build an automated ML pipeline for trigger day prediction
Ensure high-quality data using validation checks
Track experiments and models
Deploy a user-friendly web application
Enable batch and single-patient predictions

🏗️ System Architecture (High Level)
Raw data stored in MySQL
Data preprocessing using pipelines
Cleaned data stored in ClickHouse
Data validation using Great Expectations
Model training & tracking using MLflow
XGBoost model for prediction
Streamlit web app for user interface
Entire system containerized with Docker

ivf-trigger-day-system/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── ingestion.py
│   ├── preprocessing.py
│   ├── validation.py
│   ├── training.py
│   ├── prediction.py
│
├── app.py
├── main.py
├── requirements.txt
├── docker-compose.yml
├── Dockerfile
└── README.md

Features Used
Age
Day of stimulation
AMH (ng/ml)
Estradiol (pg/ml)
Progesterone (ng/ml)
Average follicle size (mm)
Follicle count

🤖 Machine Learning Model
Algorithm: XGBoost Classifier
Reason:
Handles non-linear relationships
Performs well on tabular medical data
High accuracy and robustness

🛠️ Tech Stack
Python
Pandas, NumPy, Scikit-learn
XGBoost
MySQL
ClickHouse
Great Expectations
MLflow
Streamlit
Docker
