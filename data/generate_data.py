import pandas as pd
import numpy as np
import os

# Set seed for reproducibility
np.random.seed(42)

# Number of samples
n_samples = 15000

# Generating base features
# 1. amount (exponential distribution to simulate long tail of high value transactions)
amount = np.random.exponential(scale=150.0, size=n_samples)

# 2. transaction_hour (0-23)
transaction_hour = np.random.randint(0, 24, size=n_samples)

# 3. merchant_type (categorical)
merchants = ['Groceries', 'Dining', 'Retail', 'Online Subscription', 'Travel', 'Electronics', 'Crypto']
merchant_probs = [0.3, 0.2, 0.2, 0.15, 0.05, 0.05, 0.05]
merchant_type = np.random.choice(merchants, size=n_samples, p=merchant_probs)

# 4. location (categorical)
locations = ['Local', 'Domestic', 'International', 'DarkWeb']
location_probs = [0.6, 0.3, 0.09, 0.01]
location = np.random.choice(locations, size=n_samples, p=location_probs)

# 5. account_age (in days, uniform between 1 and 3650)
account_age = np.random.randint(1, 3650, size=n_samples)

# Initial Dataframe
df = pd.DataFrame({
    'amount': amount,
    'transaction_hour': transaction_hour,
    'merchant_type': merchant_type,
    'location': location,
    'account_age': account_age
})

# Feature engineering: calculating fraud probability based on rules
# Initialize fraud probability with a base low chance
fraud_prob = np.zeros(n_samples) + 0.01

# Rule 1: High amounts are riskier
fraud_prob += np.where(df['amount'] > 1000, 0.1, 0.0)
fraud_prob += np.where(df['amount'] > 5000, 0.2, 0.0)

# Rule 2: Unusual hours (2 AM to 5 AM)
fraud_prob += np.where((df['transaction_hour'] >= 2) & (df['transaction_hour'] <= 5), 0.15, 0.0)

# Rule 3: High risk merchants
fraud_prob += np.where(df['merchant_type'] == 'Crypto', 0.2, 0.0)
fraud_prob += np.where(df['merchant_type'] == 'Electronics', 0.1, 0.0)

# Rule 4: Risky locations
fraud_prob += np.where(df['location'] == 'International', 0.15, 0.0)
fraud_prob += np.where(df['location'] == 'DarkWeb', 0.4, 0.0)

# Rule 5: Very new accounts (less than 30 days)
fraud_prob += np.where(df['account_age'] < 30, 0.1, 0.0)

# Normalize probability
fraud_prob = np.clip(fraud_prob, 0, 0.95) # cap at 95% to maintain some randomness

# Generate labels based on the calculated probability
# Generate random numbers and compare with fraud probability
random_chance = np.random.rand(n_samples)
df['is_fraud'] = (random_chance < fraud_prob).astype(int)

# Check imbalance 
print("Class distribution:")
print(df['is_fraud'].value_counts(normalize=True))

# Save dataset
os.makedirs('data', exist_ok=True)
df.to_csv('data/synthetic_fraud_data.csv', index=False)
print("Data generated successfully at data/synthetic_fraud_data.csv")
