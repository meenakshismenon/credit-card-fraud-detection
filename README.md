# Credit Card Fraud Detection Project (SENTRY)


A complete full-stack machine learning project demonstrating a realistic credit card fraud detection system.

![Dashboard](dashboard.png)
![Dashboard](dashboard2.png)
## Project Structure

```text
/backend
  main.py          # FastAPI application serving ML predictions
/data
  generate_data.py # Synthesizes a realistic, imbalanced dataset matching our inputs
  synthetic_fraud_data.csv
/frontend
  index.html       # Cyberpunk UI layout
  style.css        # Styling rules
  script.js        # Logic for Fetching APIs, charting, and live streaming updates
/models
  train.py         # Model training script with SMOTE and evaluation
  best_model.pkl   # Serialized ML model and preprocessing pipeline
  metrics.json     # Output stats to render charts in frontend
```

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Synthetic Data
Run this script to generate 15000 realistic credit card transactions based on business logic:
```bash
python data/generate_data.py
```

### 3. Train the Model
Train XGBoost, Random Forest, Logistic Regression, etc., evaluate them, and save the best performer.
```bash
python models/train.py
```

### 4. Run the Backend API
Start the FastAPI server:
```bash
python backend/main.py
# The API will be available at http://localhost:8000
```

### 5. Open the Dashboard
Simply open `frontend/index.html` in your web browser. There is no build step required!

## Using SENTRY

1. **Dashboard Overview**: Check the KPI cards and chart to see how well the models performed during training.
2. **Live Transaction Stream**: The right-side panel will slowly populate with fake simulated bank transactions pinging your model every 2.5 seconds.
3. **Manual Override**: Try inputting an amount higher than 5000, or a transaction at an International location at 3 AM to see the model instantly flag it!
