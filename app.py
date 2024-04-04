from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Loading the trained model from the pickle file
with open('trained_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Encode Education_Level
def encode_education_level(value):
    if value == 'Diploma/Degree':
        return 0
    elif value == 'No School':
        return 1
    elif value == 'Primary School':
        return 2
    elif value == 'Secondary School':
        return 3
    else:
        raise ValueError("Invalid value for Education_Level")

# Encode Nutrition_Diet
def encode_nutrition_diet(value):
    if value == 'Balanced Diet':
        return 0
    elif value == 'Low-Carb Diet':
        return 1
    elif value == 'Mediterranean Diet':
        return 2
    else:
        raise ValueError("Invalid value for Nutrition_Diet")

# Encode Smoking_Status
def encode_smoking_status(value):
    if value == 'Current Smoker':
        return 0
    elif value == 'Former Smoker':
        return 1
    elif value == 'Never Smoked':
        return 2
    else:
        raise ValueError("Invalid value for Smoking_Status")

# Encode Sleep_Quality
def encode_sleep_quality(value):
    if value == 'Good':
        return 0
    elif value == 'Poor':
        return 1
    else:
        raise ValueError("Invalid value for Sleep_Quality")

# Encode Depression_Status
def encode_depression_status(value):
    if value == 'No':
        return 0
    elif value == 'Yes':
        return 1
    else:
        raise ValueError("Invalid value for Depression_Status")

# Encode Medication_History
def encode_medication_history(value):
    if value == 'No':
        return 0
    elif value == 'Yes':
        return 1
    else:
        raise ValueError("Invalid value for Medication_History")

# Defining routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extracting features from the form data
    features = ['Sleep_Quality', 'Diabetic', 'Depression_Status', 'Medication_History', 'Education_Level', 'Nutrition_Diet', 'Smoking_Status']
   
    # Extract feature values from the form, encode them, and convert to integers
    feature_values = [
        encode_sleep_quality(request.form['Sleep_Quality']),
        int(request.form['Diabetic']),  #'Diabetic' is already binary
        encode_depression_status(request.form['Depression_Status']),
        encode_medication_history(request.form['Medication_History']),
        encode_education_level(request.form['Education_Level']),
        encode_nutrition_diet(request.form['Nutrition_Diet']),
        encode_smoking_status(request.form['Smoking_Status'])
    ]
    
    # Creating a DataFrame from the form data
    data = pd.DataFrame([feature_values], columns=features)

    # Predicting dementia
    prediction = loaded_model.predict(data)

    # Predicting probability
    prediction_probability = loaded_model.predict_proba(data)[:, 1]

    return render_template('result.html', prediction=prediction[0], prediction_probability=prediction_probability[0])

if __name__ == '__main__':
    app.run(debug=True)
