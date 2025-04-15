import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import shap
import matplotlib.pyplot as plt

# Load the derived attributes dataset with labels
df = pd.read_csv('derived_risk_attributes_with_kyc_score_and_labels.csv')

# Select only the necessary features
selected_features = [
    'SSN_24hr_Velocity',
    'SSN_72hr_Velocity',
    'Phone_24hr_Velocity',
    'Phone_72hr_Velocity',
    'Email_Gibberish_Score',
    'KYC_Score'
]

# Define features (X) and target (y)
X = df[selected_features]  # Use only the selected features
y = df['Label']  # The label column is the target

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train an XGBoost Classifier
xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)

# Predict on the test set
y_pred = xgb_model.predict(X_test)
y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]  # Probability of fraud

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"Accuracy: {accuracy:.2f}")
print(f"ROC AUC Score: {roc_auc:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Output a Model Risk Score
# Apply a scaling factor to ensure scores don't reach 100
df['Model_Risk_Score'] = xgb_model.predict_proba(scaler.transform(X))[:, 1] * 95  # Scale down slightly

# Save the results with the model risk score
df.to_csv('derived_risk_attributes_with_model_risk_score_xgb.csv', index=False)

# Display the first few rows with the model risk score (optional)
print(df[['SSN_24hr_Velocity', 'SSN_72hr_Velocity', 'Phone_24hr_Velocity', 'Phone_72hr_Velocity', 'Email_Gibberish_Score', 'KYC_Score', 'Model_Risk_Score', 'Label']].head())

# Feature importance using SHAP values
explainer = shap.Explainer(xgb_model, X_train)
shap_values = explainer(X_test)

# Plot the SHAP summary plot
shap.summary_plot(shap_values, X_test, feature_names=selected_features)

# Test the model with a sample input
print("\nTesting with a sample input:")

# Create a sample input using the same feature names and order as the training data
sample_input = {
    'SSN_24hr_Velocity': 0,
    'SSN_72hr_Velocity': 1,
    'Phone_24hr_Velocity': 1,
    'Phone_72hr_Velocity': 1,
    'Email_Gibberish_Score': 0,
    'KYC_Score': 2
}

# Ensure that the sample input has the same order of columns as X
sample_df = pd.DataFrame([sample_input], columns=selected_features)

# Scale the sample input using the same scaler
scaled_sample = scaler.transform(sample_df)

# Predict the risk score for the sample input
sample_risk_score = xgb_model.predict_proba(scaled_sample)[:, 1] * 95  # Scale down slightly
print(f"Sample input risk score: {sample_risk_score[0]:.2f}")

