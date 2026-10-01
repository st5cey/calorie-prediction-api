# Calorie Expenditure Prediction & Production API

An end-to-end Machine Learning pipeline and REST API that predicts caloric burn during physical workouts using physiological markers and exercise metrics. Built with **Python**, **XGBoost**, **Scikit-Learn**, and **FastAPI**.


## Project Overview

Predicting accurate energy expenditure is a core feature in modern fitness wearables and wellness platforms. This project transitions a raw dataset into a production-ready machine learning pipeline and microservice. 

Rather than relying purely on default inputs, this solution leverages **domain-driven feature engineering**, specifically modeling cardiac workload, to achieve exceptional predictive performance while eliminating data leakage through modular Scikit-Learn pipelines.


## Key Features & Results

* **High Predictive Accuracy:** Achieved a Mean Absolute Error (**MAE**) of **1.23 calories** and an **$R^2$ score of 0.9990** on unseen test data.
* **Domain Knowledge Integration:** Engineered `Workload_Index` (Heart_Rate * Duration), which accounted for **>95%** of total model importance.
* **Leakage-Free Architecture:** Built using Scikit-Learn `Pipeline` and `ColumnTransformer` to enforce strict isolation between cross-validation folds and holdout test sets.
* **REST:** Deployed via **FastAPI** with **Pydantic** input validation schema, enabling real-time inference via HTTP POST requests.


##  Tech Stack

* **Language:** Python 3.12
* **Data Processing & EDA:** Pandas, NumPy, Matplotlib, Seaborn
* **Machine Learning:** XGBoost, Scikit-Learn (Random Forest, Ridge Regression)
* **Model Serialization:** Joblib
* **API & Deployment:** FastAPI, Uvicorn, Pydantic, Requests


## Exploratory Data Analysis & Feature Engineering

### 1. Data Cleaning & Integrity
* Merged `calories.csv` and `exercise.csv` on primary key `User_ID`.
* Dropped identifier features (`User_ID`) to prevent false statistical associations.
* Removed duplicates from the dataset records and confirmed zero missing values.

### 2. Domain Feature Engineering
* **`BMI` (Body Mass Index):** Combined height and weight into a standard body composition index:
BMI = Weight (kg)/(Height (cm)/^2)
* **`Workload_Index`:** Derived total metabolic output as:
Workload_Index} = Heart_Rate(bpm) * Duration(min)
* **`Temp_Elevation`:** Measured temperature deviation above normal body baseline:
Temp_Elevation=Body_Temp-37.0


## Model Development & Benchmark

Three candidate algorithms were evaluated using **5-Fold Cross-Validation** on the training set:

| Model | CV MAE (Calories) | CV RMSE | CV $R^2$ Score |
| :--- | :--- | :--- | :--- |
| **XGBoost Regressor** | **1.18** | **1.85** | **0.9991** |
| Random Forest Regressor | 1.45 | 2.30 | 0.9986 |
| Ridge Regression | 11.20 | 15.40 | 0.9420 |

### Final Test Set Evaluation (XGBoost)
* **MAE:** 1.23 calories
* **RMSE:** 1.92 calories
* **$R^2$ Score:** 0.9990

### Diagnostic Analysis
* **Residual Analysis:** Scatter and error distribution plots confirmed zero prediction bias, with errors normally distributed around 0.
* **Feature Importances:** `Workload_Index` dominated model split gains (>95%), validating the hypothesis that physical effort proxies streamline gradient boosted decision trees.


## API Deployment (FastAPI)

The model is wrapped in a lightweight FastAPI application with automated Pydantic schema validation.

### API Architecture
* **`GET /`**: Health check endpoint returning API status.
* **`POST /predict`**: Accepts JSON payload, applies feature engineering dynamically, and returns predicted calories.

### Request Payload Example
```json
{
  "Gender": "female",
  "Age": 25,
  "Height": 165.0,
  "Weight": 60.0,
  "Duration": 30.0,
  "Heart_Rate": 140.0,
  "Body_Temp": 39.5
}