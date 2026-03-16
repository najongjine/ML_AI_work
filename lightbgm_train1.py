import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import os

# Create plots directory if it doesn't exist
if not os.path.exists('plots'):
    os.makedirs('plots')

# 1. Load Data
file_path = 'ai_company_adoption.csv'
df = pd.read_csv(file_path)

print(f"Dataset shape: {df.shape}")

# 2. EDA - Missing Values
missing_values = df.isnull().sum()
print("\n--- Missing Values ---")
print(missing_values[missing_values > 0])

# 3. EDA - Boxplot for revenue_growth_percent
plt.figure(figsize=(10, 6))
sns.boxplot(x=df['revenue_growth_percent'])
plt.title('Distribution of Revenue Growth Percent')
plt.xlabel('Revenue Growth (%)')
plt.savefig('plots/revenue_growth_boxplot.png')
print("\nBoxplot saved to 'plots/revenue_growth_boxplot.png'")

# 4. Data Preprocessing
# Drop non-predictive ID columns
drop_cols = ['response_id', 'company_id']
df = df.drop(columns=drop_cols)

# Identify categorical and numerical columns
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
num_cols = df.select_dtypes(exclude=['object']).columns.tolist()

# Target column
target = 'revenue_growth_percent'
if target in num_cols:
    num_cols.remove(target)

# Handle Categorical variables using LabelEncoding for LightGBM
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))

# 5. Split Data
X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Train LightGBM Model
train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data, categorical_feature=cat_cols)

params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1
}

print("\n--- Training LightGBM ---")
model = lgb.train(
    params,
    train_data,
    valid_sets=[train_data, test_data],
    num_boost_round=1000,
    callbacks=[lgb.early_stopping(stopping_rounds=50)]
)

# 7. Model Evaluation
y_pred = model.predict(X_test, num_iteration=model.best_iteration)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\n--- Model Performance ---")
print(f"RMSE: {rmse:.4f}")
print(f"R2 Score: {r2:.4f}")

# 8. Feature Importance
plt.figure(figsize=(12, 10))
lgb.plot_importance(model, max_num_features=20, importance_type='gain', ax=plt.gca())
plt.title('Top 20 Feature Importance (Gain)')
plt.tight_layout()
plt.savefig('plots/feature_importance.png')
print("Feature importance plot saved to 'plots/feature_importance.png'")
