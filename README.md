# 🛢️ Oil Sales Prediction System

An end-to-end Machine Learning solution to analyze and predict the sales volume of edible oil products. This project includes a complete pipeline for data processing, feature engineering, model training, and a REST API for real-time inference.

## 📌 Project Overview

This system is designed to help retailers optimize inventory and pricing strategies by accurately predicting product demand.

* **Problem:** Predicting `volume_sales` based on store location, product attributes, and revenue.
* **Solution:** A Random Forest Regression model trained on historical sales data.
* **Key Insight:** Sales volume is highly correlated with revenue (`value_sales`), and seasonal trends show a significant peak in the Fall.
* **Performance:** The model achieves an $R^2$ score of ~0.98 on test data.

--- 

## 📂 Project Structure

```text
├── api/
│   └── app.py              # FastAPI application & endpoints
├── data/
│   └── oil_sales.csv       # Dataset file
├── models/                 # Saved model artifacts (auto-generated)
│   ├── random_forest_model.pkl
│   ├── scaler.pkl
│   └── encoders.pkl
├── src/
│   ├── data_processing.py  # Data cleaning & validation logic
│   ├── feature_engineering.py # Feature extraction & encoding
│   ├── model.py            # Model training & prediction logic
│   └── inference.py        # Pipeline orchestrator for inference
├── train.py                # Script to trigger model training
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

# 🚀 Setup & Installation

## 1. Prerequisite
Ensure you have **Python 3.9+** installed.

---

## 2. Clone the Repository

```bash
git clone https://github.com/Basel-Mohamed/Marketeers_task.git

cd Marketeers_task
```
## 3. Create a Virtual Environment (Recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```
## 4. Install Dependencies

```bash
pip install -r requirements.txt
```
# 🛠️ How to Run

## Step 1: Train the Model

Before running the API, you must train the model to generate the necessary artifacts (.pkl files) in the models/ directory.

```bash
python train.py
```
* This script will:
* Load the data
* Clean it
* Engineer features
* Train the Random Forest model
* Save the trained artifacts

## Step 2: Start the API Server

Once training is complete, start the FastAPI server:

```bash
uvicorn api.app:app --reload
```
The server will start at:
```bash
http://127.0.0.1:8000
```

# 🔌 API Endpoints

1. Health Check

URL: ```GET /health```
Description: Checks if the API is running and the model is loaded.

2. Predict Sales

URL: ```POST /predict```
Description: Returns the predicted sales volume for a single product record.

Request Body (JSON)

for example:
```json
{
  "city": "RIYADH",
  "store_name": "Main Branch",
  "manufacturer": "Global Oils",
  "brand": "Zahra",
  "class": "Corn",
  "size": "1.5L",
  "price_bracket": "45-55",
  "year": 2024,
  "month": 10,
  "value_sales": 5000.0,
  "average_price": 45.0
}

```

# 📊 Features & Improvements

## ✅ Robust Preprocessing

Handles missing values and outliers automatically

## ✅ Advanced Features

- Generates:

- price_per_liter

- season

- brand_frequency

- Improves prediction accuracy

## ✅ Scalable Architecture

Modular code structure for easy maintenance and scaling

# 🚀 Future Roadmap

* Implement Hyperparameter Tuning (GridSearch / Bayesian Optimization)

* Add Deep Learning models (MLP / TabNet) for comparison

* Dockerize the application for cloud deployment
