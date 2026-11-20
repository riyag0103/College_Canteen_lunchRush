# College Canteen Lunch Rush

## Project Information

- **Course:** MLOps (SCSE3040)
- **Problem ID:** MP002
- **Project:** College Canteen Lunch Rush
- **Domain:** Food Demand Forecasting
- **Task Type:** Regression
- **Target Variable:** `plates_sold`
- **Evaluation Metric:** Mean Absolute Error (MAE)

## Project Overview

The objective of this project is to develop an end-to-end MLOps pipeline to predict the number of food plates sold in a college canteen.

The project uses historical canteen data containing information such as students on campus, menu prices, temperature, examination schedules, special dishes, queue times, day types, and college blocks.

The pipeline will cover data validation, data cleaning, exploratory data analysis, model training, experiment tracking, model deployment, monitoring, and retraining.

## Dataset

The project provides three datasets:

- `train.csv` – Historical training data.
- `later.csv` – Later-period data for drift analysis and model retraining.
- `holdout_features.csv` – Unlabelled data for final predictions.

## Project Objectives

1. Identify and handle data quality issues.
2. Track datasets and versions using DVC.
3. Train and evaluate regression models.
4. Track experiments using MLflow.
5. Build a prediction API using FastAPI.
6. Containerize the application using Docker.
7. Deploy the application using Kubernetes.
8. Monitor data and model performance.
9. Generate predictions for the holdout dataset.

## Team Members

| S. No. | Name | Enrolment Number |
|---|---|---|
| 1 | [Member 1] | [Enrolment Number] |
| 2 | [Member 2] | [Enrolment Number] |
| 3 | [Member 3] | [Enrolment Number] |
| 4 | [Member 4] | [Enrolment Number] |

## Technology Stack

- Python
- Pandas and NumPy
- Scikit-learn
- DVC
- MLflow
- FastAPI
- Docker
- Kubernetes
- Pytest
- Git and GitHub

## Repository Structure

The repository will contain source code, tests, dataset versioning files, model artifacts, API implementation, deployment configurations, and reports.

## Project Status

Project setup and implementation are in progress.
