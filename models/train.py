import pandas as pd
import numpy as np
import json
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

print("Loading data...")
df = pd.read_csv('data/synthetic_fraud_data.csv')

# Features and Target
X = df.drop('is_fraud', axis=1)
y = df['is_fraud']

print("Class distribution before SMOTE:")
print(y.value_counts())

# Split into required sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Preprocessing
numerical_cols = ['amount', 'transaction_hour', 'account_age']
categorical_cols = ['merchant_type', 'location']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ])

print("Fitting preprocessor...")
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("Applying SMOTE...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)

print(f"Class distribution after SMOTE (train):")
print(y_train_resampled.value_counts())

# Define models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    'Naive Bayes': GaussianNB(),
    'Decision Tree': DecisionTreeClassifier(random_state=42)
}

results = {}
best_model = None
best_auc = 0
best_model_name = ""

print("Training models...")
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_resampled, y_train_resampled)
    
    # Predict on test
    y_pred = model.predict(X_test_processed)
    y_prob = model.predict_proba(X_test_processed)[:, 1] if hasattr(model, "predict_proba") else y_pred
    
    # Calculate metrics
    cm = confusion_matrix(y_test, y_pred).tolist()
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    results[name] = {
        'Accuracy': round(acc, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'F1-Score': round(f1, 4),
        'AUC-ROC': round(auc, 4),
        'ConfusionMatrix': cm
    }
    
    if auc > best_auc:
        best_auc = auc
        best_model = model
        best_model_name = name

print(f"Best model: {best_model_name} with AUC = {best_auc}")

# Save the preprocessor and the best model together
os.makedirs('models', exist_ok=True)
with open('models/best_model.pkl', 'wb') as f:
    pickle.dump({
        'preprocessor': preprocessor,
        'model': best_model,
        'model_name': best_model_name
    }, f)

# Save metrics for frontend
with open('models/metrics.json', 'w') as f:
    json.dump({
        'best_model': best_model_name,
        'metrics': results
    }, f, indent=4)

print("Training finished. Models and metrics saved.")
