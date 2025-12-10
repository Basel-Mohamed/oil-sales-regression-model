# Oil Sales Prediction API Documentation

## How to Run the Application

Follow these steps to train the model and start the API server:

```bash
# 1. Train the model
python train.py

# 2. Start the API
cd api && uvicorn app:app --reload
```

**Base URL:** `http://127.0.0.1:8000`

---

## 1. Health Check
Checks if the API is active and the machine learning model is successfully loaded.

* **Endpoint:** `/health`
* **Method:** `GET`
* **Response:**
  ```json
  {
    "status": "healthy",
    "model": "loaded"
  }



# Predict Sales

Predicts the **volume sales** for a specific oil product based on store location, product attributes, and pricing data.

---

## Endpoint
**URL:** `/predict`  
**Method:** `POST`  
**Headers:**  
- `Content-Type: application/json`

---

## Request Body Schema

The API requires a JSON object with the following fields:

| Field           | Type    | Description                                               |
|-----------------|---------|-----------------------------------------------------------|
| city            | String  | City where the store is located                          |
| store_name      | String  | Full name/ID of the store                                |
| manufacturer    | String  | Product manufacturer                                     |
| brand           | String  | Product brand name                                        |
| class           | String  | Product classification (maps to `class_` internally)     |
| size            | String  | Product size (e.g., `"0.75L"`)                            |
| price_bracket   | String  | Price category (e.g., `"101+"`)                           |
| year            | Integer | Year of sales record                                      |
| month           | Integer | Month of sales record (1–12)                              |
| value_sales     | Float   | Total sales value in currency                             |
| average_price   | Float   | Average unit price                                        |

---

## Example Request (JSON)

```json
{
  "city": "RIYADH",
  "store_name": "HM No  86781  GS-CENTER-RIYADH MAIN RD  RIYADH",
  "manufacturer": "AL HILAL INDUSTRIES",
  "brand": "BAYTNA",
  "class": "SUNFLOWER",
  "size": "0.75L",
  "price_bracket": "101+",
  "year": 2023,
  "month": 1,
  "value_sales": 171.7,
  "average_price": 101.0
}
```

## Example Success Response

The API returns the predicted volume_sales along with the input data for verification.

```json
{
  "predicted_volume_sales": 1.7,
  "input_data": {
    "city": "RIYADH",
    "store_name": "HM No  86781  GS-CENTER-RIYADH MAIN RD  RIYADH",
    "manufacturer": "AL HILAL INDUSTRIES",
    "brand": "BAYTNA",
    "class": "SUNFLOWER",
    "size": "0.75L",
    "price_bracket": "101+",
    "year": 2023,
    "month": 1,
    "value_sales": 171.7,
    "average_price": 101.0
  }
}
  ```

## Error Response (400 Bad Request)
```json
{
  "detail": "Field required"
}
```


## cURL Usage Example
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "city": "RIYADH",
    "store_name": "HM No  86781  GS-CENTER-RIYADH MAIN RD  RIYADH",
    "manufacturer": "AL HILAL INDUSTRIES",
    "brand": "BAYTNA",
    "class": "SUNFLOWER",
    "size": "0.75L",
    "price_bracket": "101+",
    "year": 2023,
    "month": 1,
    "value_sales": 171.7,
    "average_price": 101.0
  }'

  ```