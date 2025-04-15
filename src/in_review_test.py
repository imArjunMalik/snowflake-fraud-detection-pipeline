import pandas as pd
from datetime import datetime, timedelta
import re
import joblib

# Load the in-review dataset
df_in_review = pd.read_csv('synthetic_account_opening_data.csv')

# Filter only 'In Review' transactions
in_review_transactions = df_in_review[df_in_review['Attempt_Status'] == 'In Review']

# Convert 'Timestamp' column to datetime
in_review_transactions['Timestamp'] = pd.to_datetime(in_review_transactions['Timestamp'])

# Function to calculate velocities within a specified time window
def calculate_velocity(df, column, time_window):
    velocities = []
    for i, row in df.iterrows():
        current_value = row[column]
        current_time = row['Timestamp']
        start_time = current_time - pd.Timedelta(time_window)
        
        # Count occurrences within the time window
        count = len(df[(df[column] == current_value) & 
                       (df['Timestamp'] >= start_time) & 
                       (df['Timestamp'] <= current_time)])
        velocities.append(count)
    return velocities

# Calculate velocities for SSN, Phone, Address, and Device
in_review_transactions['SSN_24hr_Velocity'] = calculate_velocity(in_review_transactions, 'SSN', '1D')
in_review_transactions['SSN_72hr_Velocity'] = calculate_velocity(in_review_transactions, 'SSN', '3D')

in_review_transactions['Phone_24hr_Velocity'] = calculate_velocity(in_review_transactions, 'Phone_Number', '1D')
in_review_transactions['Phone_72hr_Velocity'] = calculate_velocity(in_review_transactions, 'Phone_Number', '3D')

in_review_transactions['Address_24hr_Velocity'] = calculate_velocity(in_review_transactions, 'Location', '1D')
in_review_transactions['Address_72hr_Velocity'] = calculate_velocity(in_review_transactions, 'Location', '3D')

in_review_transactions['Device_24hr_Velocity'] = calculate_velocity(in_review_transactions, 'Device_Info', '1D')
in_review_transactions['Device_72hr_Velocity'] = calculate_velocity(in_review_transactions, 'Device_Info', '3D')

# Function to calculate gibberish score
def calculate_gibberish_score(email):
    digit_count = len(re.findall(r'\d', email))
    special_char_count = len(re.findall(r'[\W_]', email))
    length = len(email)
    gibberish_patterns = ['123', 'test', 'abc', 'xyz']
    pattern_matches = sum(1 for pattern in gibberish_patterns if pattern in email)
    score = (digit_count + special_char_count + pattern_matches) / max(1, length)
    return score

# Add the email gibberish score for each in-review transaction
in_review_transactions['Email_Gibberish_Score'] = in_review_transactions['Email_Address'].apply(calculate_gibberish_score)

# Function to calculate Phone Risk Score
def calculate_phone_risk_score(row):
    return 0.1 * row['Phone_24hr_Velocity'] + 0.05 * row['Phone_72hr_Velocity']

# Add the phone risk score
in_review_transactions['Phone_Risk_Score'] = in_review_transactions.apply(calculate_phone_risk_score, axis=1)

# Calculate KYC Score as the average of Phone Risk and Email Gibberish Score
in_review_transactions['KYC_Score'] = (in_review_transactions['Phone_Risk_Score'] + in_review_transactions['Email_Gibberish_Score']) / 2

# Ensure all features used in training are included (without 'Email_Gibberish_Score' and 'KYC_Score' if they were not used)
features = [
    'SSN_24hr_Velocity', 'SSN_72hr_Velocity',
    'Phone_24hr_Velocity', 'Phone_72hr_Velocity',
    'Address_24hr_Velocity', 'Address_72hr_Velocity',
    'Device_24hr_Velocity', 'Device_72hr_Velocity'
]

# Load the pre-trained scaler and model
scaler = joblib.load('scaler.pkl')
model = joblib.load('risk_score_model.pkl')

# Filter the dataset to include only these features
X_in_review = in_review_transactions[features]

# Scale the features for in-review transactions
X_in_review_scaled = scaler.transform(X_in_review)

# Calculate model risk scores using the loaded model
in_review_transactions['Model_Risk_Score'] = model.predict_proba(X_in_review_scaled)[:, 1] * 95  # Scale down slightly

# Save the updated dataset with model risk scores
in_review_transactions.to_csv('in_review_transactions_with_risk_scores.csv', index=False)

# Display the first few rows of the updated dataset
print(in_review_transactions[['SSN', 'Phone_Number', 'Model_Risk_Score', 'KYC_Score']].head())
