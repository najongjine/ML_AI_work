import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# 1. Load Data
file_path = 'ai_company_adoption.csv'
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Could not find {file_path}. Please ensure the dataset is in the current directory.")

df = pd.read_csv(file_path)

# 2. Select Features and Target
# Features: ai_adoption_rate, productivity_change_percent
# Target: revenue_growth_percent
features = ['ai_adoption_rate', 'productivity_change_percent']
target = 'revenue_growth_percent'

X = df[features]
y = df[target]

print(f"Dataset loaded. Shape: {df.shape}")
print(f"Features: {features}")
print(f"Target: {target}")

# 3. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train LightGBM Model
# Using simple regression parameters
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'verbose': -1,
    'seed': 42
}

print("\n--- Training LightGBM Model ---")
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

model = lgb.train(
    params,
    train_data,
    valid_sets=[train_data, test_data],
    num_boost_round=1000,
    callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=100)]
)

# 5. Evaluate Model
y_pred = model.predict(X_test, num_iteration=model.best_iteration)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\n--- Model Evaluation ---")
print(f"RMSE: {rmse:.4f}")
print(f"R2 Score: {r2:.4f}")

# 6. Save Model
model_filename = 'revenue_model.joblib'
joblib.dump(model, model_filename)
print(f"\nModel saved to '{model_filename}'")
