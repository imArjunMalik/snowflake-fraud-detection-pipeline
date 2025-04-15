from flask import Flask, request, jsonify
import joblib
import pandas as pd
import tensorflow as tf

# Initialize the Flask application
app = Flask(__name__)

# Load the pre-trained model and preprocessor
model = tf.keras.models.load_model('fraud_detection_model_nn.h5')  # or use 'fraud_detection_model_xgboost.pkl'
preprocessor = joblib.load('preprocessor.pkl')

# Function to calculate the custom risk score (from your earlier implementation)
def calculate_risk_score(row):
    score = 100
    phone_risk_score = row.get('Phone_Risk_Score', 0)
    score -= (phone_risk_score * 100 // 20) * 3
    email_gibberish_score = row['Email_Gibberish_Score']
    score -= (email_gibberish_score * 100 // 10) * 1
    if row['SSN_24hr_Velocity'] > 5:
        score -= 10
    if row['SSN_72hr_Velocity'] > 5:
        score -= 5
    if row['Phone_72hr_Velocity'] > 5:
        score -= 5
    if row['Address_72hr_Velocity'] > 5:
        score -= 5
    if row['Device_72hr_Velocity'] > 5:
        score -= 5
    return max(0, min(100, score))

# Define a route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Parse JSON from the incoming request
    input_data = request.json
    
    # Convert the JSON data to a DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Preprocess the data
    preprocessed_data = preprocessor.transform(input_df)
    
    # Make the prediction
    prediction_prob = model.predict(preprocessed_data).flatten()
    prediction = (prediction_prob > 0.5).astype(int)
    
    # Calculate the custom risk score
    risk_score = calculate_risk_score(input_data)
    
    # Prepare the response
    response = {
        'prediction': 'Fraudulent' if prediction[0] == 1 else 'Benign',
        'probability': float(prediction_prob[0]),
        'risk_score': risk_score
    }
    
    return jsonify(response)

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)

"""
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{
  "Device_Info": "MacOS",
  "SSN_24hr_Velocity": 1,
  "SSN_72hr_Velocity": 2,
  "Phone_24hr_Velocity": 1,
  "Phone_72hr_Velocity": 3,
  "Address_24hr_Velocity": 2,
  "Address_72hr_Velocity": 4,
  "Device_24hr_Velocity": 1,
  "Device_72hr_Velocity": 5,
  "Email_Gibberish_Score": 0.2,
  "IP_Risk_Score": 0.01,
  "Phone_Risk_Score": 0.3
}'
"""
