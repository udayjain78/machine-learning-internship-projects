import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
import shap
import joblib
import os
from datetime import datetime

# Set random seed for reproducibility
np.random.seed(42)

# Set plotting style
sns.set(style='whitegrid')

# Load all pickle files and concatenate
data_path = 'fraud_detection/dataset/data/'
files = [f for f in os.listdir(data_path) if f.endswith('.pkl')]
df_list = []
for file in files:
    df = pd.read_pickle(os.path.join(data_path, file))
    df_list.append(df)
df = pd.concat(df_list, ignore_index=True)

# Display basic info
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print("\nData types:")
print(df.dtypes)
print("\nFirst 5 rows:")
print(df.head())
print("\nNull values:")
print(df.isnull().sum())

# Summary statistics
print("Summary statistics:")
print(df.describe())

# Fraud distribution
fraud_counts = df['TX_FRAUD'].value_counts()
print(f"\nFraud distribution:\n{fraud_counts}")
print(f"Fraud rate: {df['TX_FRAUD'].mean():.4f}")

# Plot fraud vs non-fraud (commented out for CLI)
# plt.figure(figsize=(6,4))
# sns.countplot(x='TX_FRAUD', data=df)
# plt.title('Fraud vs Non-Fraud Transactions')
# plt.show()

# Amount distribution by fraud class (commented out)
# plt.figure(figsize=(10,6))
# sns.histplot(data=df, x='TX_AMOUNT', hue='TX_FRAUD', bins=50, kde=True)
# plt.title('Transaction Amount Distribution by Fraud Class')
# plt.show()

# Transactions per day
df['TX_DATE'] = df['TX_DATETIME'].dt.date
daily_tx = df.groupby('TX_DATE').size()
# plt.figure(figsize=(12,6))
# daily_tx.plot()
# plt.title('Number of Transactions per Day')
# plt.ylabel('Number of Transactions')
# plt.show()

# Fraud rates by customer and terminal
customer_fraud_rate = df.groupby('CUSTOMER_ID')['TX_FRAUD'].mean().sort_values(ascending=False).head(10)
terminal_fraud_rate = df.groupby('TERMINAL_ID')['TX_FRAUD'].mean().sort_values(ascending=False).head(10)

# Plots commented out

# Correlation heatmap (commented out)

# Extract date/time features
df['TX_HOUR'] = df['TX_DATETIME'].dt.hour
df['TX_DAY'] = df['TX_DATETIME'].dt.day
df['TX_MONTH'] = df['TX_DATETIME'].dt.month
df['TX_DAY_OF_WEEK'] = df['TX_DATETIME'].dt.dayofweek

# Label encode categorical features
le_customer = LabelEncoder()
le_terminal = LabelEncoder()
df['CUSTOMER_ID_ENCODED'] = le_customer.fit_transform(df['CUSTOMER_ID'])
df['TERMINAL_ID_ENCODED'] = le_terminal.fit_transform(df['TERMINAL_ID'])

# Aggregate features
# Customer-based
customer_agg = df.groupby('CUSTOMER_ID').agg(
    CUSTOMER_TX_COUNT=('TRANSACTION_ID', 'count'),
    CUSTOMER_AVG_AMOUNT=('TX_AMOUNT', 'mean'),
    CUSTOMER_FRAUD_COUNT=('TX_FRAUD', 'sum')
).reset_index()
customer_agg['CUSTOMER_FRAUD_RATE'] = customer_agg['CUSTOMER_FRAUD_COUNT'] / customer_agg['CUSTOMER_TX_COUNT']

# Terminal-based
terminal_agg = df.groupby('TERMINAL_ID').agg(
    TERMINAL_TX_COUNT=('TRANSACTION_ID', 'count'),
    TERMINAL_FRAUD_COUNT=('TX_FRAUD', 'sum')
).reset_index()
terminal_agg['TERMINAL_FRAUD_RATE'] = terminal_agg['TERMINAL_FRAUD_COUNT'] / terminal_agg['TERMINAL_TX_COUNT']

# Merge back
df = df.merge(customer_agg[['CUSTOMER_ID', 'CUSTOMER_TX_COUNT', 'CUSTOMER_AVG_AMOUNT', 'CUSTOMER_FRAUD_RATE']], on='CUSTOMER_ID', how='left')
df = df.merge(terminal_agg[['TERMINAL_ID', 'TERMINAL_TX_COUNT', 'TERMINAL_FRAUD_RATE']], on='TERMINAL_ID', how='left')

# Drop unnecessary columns
df = df.drop(['TRANSACTION_ID', 'TX_DATETIME', 'CUSTOMER_ID', 'TERMINAL_ID', 'TX_TIME_SECONDS', 'TX_TIME_DAYS', 'TX_FRAUD_SCENARIO', 'TX_DATE'], axis=1)

print("Feature engineered dataset shape:", df.shape)
print("Columns:", list(df.columns))

# Check fraud ratio
print(f"Fraud ratio: {df['TX_FRAUD'].mean():.4f}")

# Split features and target
X = df.drop('TX_FRAUD', axis=1)
y = df['TX_FRAUD']

# Apply SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print(f"Original dataset shape: {X.shape}")
print(f"Resampled dataset shape: {X_resampled.shape}")
print(f"Resampled fraud ratio: {y_resampled.mean():.4f}")

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

print(f"Train shape: {X_train.shape}")
print(f"Test shape: {X_test.shape}")

# Define models
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
    'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
    'LightGBM': LGBMClassifier(random_state=42)
}

# Function to evaluate model
def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_pred_proba)
    }

    # Confusion Matrix (commented out)
    # cm = confusion_matrix(y_test, y_pred)
    # plt.figure(figsize=(6,4))
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    # plt.title(f'Confusion Matrix - {model.__class__.__name__}')
    # plt.show()

    # ROC Curve (commented out)

    return metrics, model

# Evaluate all models
results = {}
trained_models = {}
for name, model in models.items():
    print(f"Evaluating {name}...")
    metrics, trained_model = evaluate_model(model, X_train, y_train, X_test, y_test)
    results[name] = metrics
    trained_models[name] = trained_model
    print(f"{name} Metrics: {metrics}")
    print("-"*50)

# Feature Importance (commented out plots)

# Model Explainability (SHAP might fail in CLI, commented out)
# explainer = shap.TreeExplainer(trained_models['XGBoost'])
# shap_values = explainer.shap_values(X_test)
# shap.summary_plot(shap_values, X_test, feature_names=X_test.columns)

# Model Evaluation Summary
results_df = pd.DataFrame(results).T
print("Model Comparison:")
print(results_df)

# Baseline rule
baseline_pred = (X_test['TX_AMOUNT'] > 220).astype(int)
baseline_metrics = {
    'Accuracy': accuracy_score(y_test, baseline_pred),
    'Precision': precision_score(y_test, baseline_pred),
    'Recall': recall_score(y_test, baseline_pred),
    'F1-Score': f1_score(y_test, baseline_pred)
}
print(f"Baseline Rule Metrics: {baseline_metrics}")

# Save Model
joblib.dump(trained_models['XGBoost'], 'fraud_detection_model.pkl')
print("Model saved as fraud_detection_model.pkl")

# Save label encoders
joblib.dump(le_customer, 'le_customer.pkl')
joblib.dump(le_terminal, 'le_terminal.pkl')

print("Testing completed successfully!")
