# -*- coding: utf-8 -*-
"""Dementia_Model_implementation.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1z3zF10rRDo-bk3lRsLdG1_pdpLDjzTQ7
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, chi2
from sklearn import preprocessing

def read_csv(filename):
    '''Read the csv file into a dataframe'''
    try:
        return pd.read_csv(filename)
    except Exception as e:
        print("Error:", str(e))
        return None

def missing_value_treatment(data):
    '''Missing Value treatment'''
    data['Dosage in mg'].fillna(value=0, inplace=True)
    data['Prescription'].fillna("Unknown", inplace=True)
    print("The null values in the dataset:", data.isnull().any().any())

def label_encoder_train(data):
    '''Label Encoding'''
    label_encoder = preprocessing.LabelEncoder()
    columns_to_encode = ['Nutrition_Diet', 'Smoking_Status', 'Physical_Activity', 'Sleep_Quality','Depression_Status','Medication_History','Education_Level']
    encoded_data = {}
    for col in columns_to_encode:
        data[col] = label_encoder.fit_transform(data[col])
        encoded_values = {label: encoded_value for label, encoded_value in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))}
        print(f"Encoded Values for data of {col}:")
        for label, encoded_value in encoded_values.items():
            print(f"{label}: {encoded_value}")
        encoded_data[col] = {
            'label_encoder': label_encoder,
            'encoded_values': encoded_values
        }

    return data, encoded_data

def label_encoder_test(df):
    '''Label Encoding for test data'''
    label_encoder = preprocessing.LabelEncoder()
    columns_to_encode = ['Nutrition_Diet', 'Smoking_Status', 'Sleep_Quality', 'Depression_Status', 'Medication_History', 'Education_Level']
    encoded_data = {}
    for col in columns_to_encode:
        df[col] = label_encoder.fit_transform(df[col])
        encoded_values = {label: encoded_value for label, encoded_value in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))}
        print(f"Encoded Values for synthetic data of {col}:")
        for label, encoded_value in encoded_values.items():
            print(f"{label}: {encoded_value}")
        encoded_data[col] = {
            'label_encoder': label_encoder,
            'encoded_values': encoded_values
        }

    return df



def train_model(X_train, y_train):
    classifier_nb = BernoulliNB()
    classifier_nb.fit(X_train, y_train)
    return classifier_nb

def evaluate_model(classifier, X_test, y_test, threshold=0.5):
    y_pred_probs = classifier.predict_proba(X_test)
    y_pred = (y_pred_probs[:, 1] >= threshold).astype(int)

    '''Calculate and print evaluation metrics'''
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("F1 Score:", f1)
    print("Precision:", precision)
    print("Recall:", recall)

    '''Plot confusion matrix'''
    plot_confusion_matrix(y_test, y_pred)

def plot_confusion_matrix(y_true, y_pred):
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

def model_implementation(data):
    X = data[['Sleep_Quality', 'Diabetic', 'Depression_Status', 'Medication_History', 'Education_Level', 'Nutrition_Diet', 'Smoking_Status']]
    y = data['Dementia']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.28, random_state=0)

    '''Train the model'''
    classifier = train_model(X_train, y_train)

    '''Evaluate the model'''
    evaluate_model(classifier, X_test, y_test, threshold=0.46)
    return classifier

def preprocess_test_data(df):
    '''Preprocess test data'''
    label_encoder_test(df)
    return df

def extract_features(df):
    '''Extract features from test data'''
    return df[['Sleep_Quality', 'Diabetic', 'Depression_Status', 'Medication_History', 'Education_Level', 'Nutrition_Diet', 'Smoking_Status']]

def predict_labels(model, X_test):
    '''Predict labels for test data'''
    return model.predict(X_test)

def print_results(actual_labels, predicted_labels):
    '''Print the actual and predicted labels'''
    print("Actual Labels:", actual_labels)
    print("Predicted Labels:", predicted_labels)

def predict_scores(model, X_test):
    '''Predict scores for test data'''
    return model.predict_proba(X_test)[:, 1]

def main():
    data = read_csv("dementia_patients_health_data.csv")
    df = read_csv("Synthetic_dementia_test_data.csv")

    if data is not None and df is not None:
        '''Preprocess training data'''
        missing_value_treatment(data)
        data, encoded_data = label_encoder_train(data)

        '''Preprocess test data'''
        df = label_encoder_test(df)

        '''Train the model'''
        classifier = model_implementation(data)

        if classifier:
            '''Save only the trained model to a pickle file'''
            with open('trained_model.pkl', 'wb') as file:
                pickle.dump(classifier, file)

            '''Load the trained model from the pickle file'''
            with open('trained_model.pkl', 'rb') as file:
                loaded_model = pickle.load(file)

            if loaded_model:
                '''Extract features from test data'''
                X_test = extract_features(df)

                '''Predict labels for test data'''
                y_pred = predict_labels(loaded_model, X_test)

                actual_labels = df['Dementia'].values

                '''Print the actual and predicted labels'''
                print_results(df['Dementia'].values, y_pred)

                '''Predict scores for test data'''
                scores = predict_scores(loaded_model, X_test)

                '''Print the predicted scores'''
                print("Predicted Scores:", scores)

            else:
                print("Failed to load the trained model from the pickle file.")
        else:
            print("Failed to train the model.")
    else:
        print("Failed to load data.")

if __name__ == "__main__":
    main()

from google.colab import files
files.download('trained_model.pkl')