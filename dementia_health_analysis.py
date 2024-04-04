# -*- coding: utf-8 -*-
"""Dementia_Health_Analysis.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Ct1-0vWEGqZtHxicWl-wL28pCKo1kGXu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
def read_csv(data):
    try:
        return pd.read_csv("dementia_patients_health_data.csv")
    except Exception as e:
        print("Error:", str(e))
    return None

'''Data Checks'''
def shape(data):
    shape=data.shape
    print("The shape of the dataset is :",shape)

def info(data):
    info=data.info()
    print("The shape of the dataset is :",info)

def null_value_check(data):
    null_columns = data.isnull().any()
    if null_columns.any():
        print("DataFrame contains null value in the following columns:")
        print(null_columns[null_columns].index.tolist())
    else:
        print("DataFrame has no null values.")

def duplicate_value_check(data):
    duplicate_rows = data[data.duplicated()]
    if not duplicate_rows.empty:
        print("DataFrame contains duplicate rows:")
        print(duplicate_rows)
    else:
        print("DataFrame has no duplicate rows.")

def outlier_check(data):
    from scipy.stats import zscore
    numeric_columns = data.select_dtypes(include='number').columns
    z_scores = zscore(data[numeric_columns])
    threshold = 2
    outliers = (z_scores > threshold) | (z_scores < -threshold)
    outliers_df = data[numeric_columns][outliers.any(axis=1)]
    if outliers_df.empty:
        print("Dataset contains outliers:")
        print(outliers_df)
    else:
        print("Dataset does not contain outliers.")

def zero_value_check(data):
    zero_value=data[data == 0].any().sum()
    if zero_value.any():
        print("DataFrame contains Zero value in the following columns:")
        print(zero_value[zero_value].index.tolist())
    else:
        print("DataFrame has no zero values.")

'''Data Visualization'''
'''Univariate analyis'''
def age_density(data):
    plt.figure(figsize=(5, 5))
    data['Age'].plot(kind='density', color='black')
    plt.xlabel('Age')
    plt.ylabel('Density')
    plt.title("Representation of Age")
    plt.show()


def bar_NutritionDiet(data):
    diet_counts = data['Nutrition_Diet'].value_counts()
    plt.figure(figsize=(8, 6))
    diet_counts.plot(kind='bar', color='skyblue')
    plt.title('Frequency Distribution of Nutrition Diet')
    plt.xlabel('Nutrition Diet')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


'''Bivariate Analysis'''

def diabetic_cmap(data):
    contingency_table = pd.crosstab(data['Diabetic'], data['Dementia'])
    from scipy.stats import chi2_contingency
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    print(f"Chi-square test p-value: {p}")
    plt.figure(figsize=(8, 6))
    sns.heatmap(contingency_table, annot=True, cmap='Blues', fmt='d')
    plt.title('Contingency Table: Diabetic vs. Dementia')
    plt.xlabel('Dementia')
    plt.ylabel('Diabetic')
    plt.show()

def prescription_countplot(data):
    sns.countplot(x='Prescription', hue='Dementia', data=data)
    plt.title('Prescription vs. Dementia')
    plt.xlabel('Prescription')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.legend(title='Dementia', labels=['0', '1'])
    plt.show()


def Sleepquality_violinplot(data):
    sns.violinplot(x='Dementia', y='Sleep_Quality', data=data)
    plt.title('Distribution of Sleep Quality by Dementia Status')
    plt.xlabel('Dementia')
    plt.ylabel('Sleep Quality')
    plt.show()


def Depression_barplot(data):
    sns.barplot(x='Dementia', y='Depression_Status', data=data, ci='sd')
    plt.xlabel('Dementia')
    plt.ylabel('Mean Depression_Status')
    plt.title('Mean Depression_Status by Dementia Status with Confidence Intervals')
    plt.show()


def medication_barplot(data):
    fig = px.bar(data, x="Dementia", y="Medication_History", title='Medication_History for Dementia')
    fig.show()


def Smoking_countplot(data):
    plt.figure(figsize=(10, 6))
    sns.countplot(data=data, x='Smoking_Status', hue='Dementia')
    plt.title('Dementia vs Smoking_Status')
    plt.xlabel('Smoking_Status')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.legend(title='Dementia', labels=['0', '1'])
    plt.show()


'''Data Preprocessing'''
'''Missing Value treatment'''
def missing_value_treatment(data):
    data['Dosage in mg'].fillna(value=0,inplace=True)
    data['Prescription'].fillna("Unknown",inplace=True)
    print("The null values in the dataset:",data.isnull().any().any())

'''Label Encoding'''
from sklearn import preprocessing
def label_encoder(data):
    label_encoder = preprocessing.LabelEncoder()
    columns_to_encode = ['Nutrition_Diet', 'Smoking_Status', 'Sleep_Quality','Depression_Status','Medication_History',
                         'Education_Level','Prescription','Dominant_Hand','Gender','Family_History',
                         'APOE_ε4','Physical_Activity','Chronic_Health_Conditions']
    for col in columns_to_encode:
        data[col] = label_encoder.fit_transform(data[col])
        #print("The encoded data is:",data[col])

def model_LR(data):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, confusion_matrix
    X = data[['Nutrition_Diet', 'Smoking_Status', 'Sleep_Quality', 'Medication_History',
              'Depression_Status','Education_Level','Diabetic']]
    y = data['Dementia']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.28, random_state=0)
    classifier_LR = LogisticRegression()
    classifier_LR.fit(X_train, y_train)
    y_pred = classifier_LR.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy for Logistic Regression:", accuracy)
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(conf_matrix)

def model_gradient(data):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, confusion_matrix
    X = data[['Nutrition_Diet', 'Smoking_Status', 'Sleep_Quality', 'Medication_History',
              'Depression_Status','Education_Level','Diabetic']]
    y = data['Dementia']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.28, random_state=0)
    classifier_GB = GradientBoostingClassifier()
    classifier_GB.fit(X_train, y_train)
    y_pred = classifier_GB.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy for gradientboosting:", accuracy)
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(conf_matrix)

def model_mlp(data):
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix
    X = data[['Nutrition_Diet', 'Smoking_Status', 'Sleep_Quality', 'Medication_History',
              'Depression_Status','Education_Level','Diabetic']]
    y = data['Dementia']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.28, random_state=0)
    classifier = MLPClassifier()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy for MLP:", accuracy)
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(conf_matrix)


def SGD(data):
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import SGDClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix
    X  = data[['Sleep_Quality','Diabetic','Depression_Status','Medication_History',
               'Education_Level','Nutrition_Diet', 'Smoking_Status']]
    y = data['Dementia']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.28, random_state=42)
    sgd_classifier = SGDClassifier(loss='log', random_state=42)
    sgd_classifier.fit(X_train, y_train)
    y_pred = sgd_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy for Stochastic Gradient Descent:", accuracy)
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(conf_matrix)

def BaggingClassifier(data):
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import BaggingClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix
    X  = data[['Sleep_Quality','Diabetic','Depression_Status','Medication_History',
               'Education_Level','Nutrition_Diet', 'Smoking_Status']]
    y = data['Dementia']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.28, random_state=42)
    base_classifier = DecisionTreeClassifier(random_state=42)
    bagging_classifier = BaggingClassifier(base_estimator=base_classifier, n_estimators=10, random_state=42)
    bagging_classifier.fit(X_train, y_train)
    y_pred = bagging_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy for BaggingClassifier:", accuracy)
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(conf_matrix)

def GaussianProcessClassifier(data):
    from sklearn.model_selection import train_test_split
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import RBF
    from sklearn.metrics import accuracy_score, confusion_matrix
    X  = data[['Sleep_Quality','Diabetic','Depression_Status','Medication_History',
               'Education_Level','Nutrition_Diet', 'Smoking_Status']]
    y = data['Dementia']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.28, random_state=42)
    kernel = 1.0 * RBF(length_scale=1.0)
    gpc = GaussianProcessClassifier(kernel=kernel, random_state=42)
    gpc.fit(X_train, y_train)
    y_pred = gpc.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy for GaussianProcessClassifier:", accuracy)
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(conf_matrix)

def QuadraticDiscriminantAnalysis(data):
    from sklearn.model_selection import train_test_split
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    from sklearn.metrics import accuracy_score, confusion_matrix
    X = data[['Sleep_Quality', 'Diabetic', 'Depression_Status', 'Medication_History',
              'Education_Level', 'Nutrition_Diet', 'Smoking_Status']]
    y = data['Dementia']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.28, random_state=42)
    qda = QuadraticDiscriminantAnalysis()
    qda.fit(X_train, y_train)
    y_pred = qda.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy for Quadratic Discriminant Analysis (QDA):", accuracy)
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(conf_matrix)

def naivebayes(data):
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import BernoulliNB
    from sklearn.metrics import accuracy_score, confusion_matrix
    X = data[['Nutrition_Diet', 'Smoking_Status', 'Sleep_Quality', 'Medication_History',
              'Depression_Status','Education_Level','Diabetic']]
    y = data['Dementia']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.28, random_state=0)
    classifier_nb = BernoulliNB()
    classifier_nb.fit(X_train, y_train)
    y_pred =  classifier_nb.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy of BernoulliNB is:", accuracy)
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(conf_matrix)


def main():
    data = read_csv("dementia_patients_health_data.csv")
    data
    if data is not None:
        shape(data)
        info(data)
        null_value_check(data)
        duplicate_value_check(data)
        outlier_check(data)
        zero_value_check(data)
        age_density(data)
        bar_NutritionDiet(data)
        diabetic_cmap(data)
        prescription_countplot(data)
        Sleepquality_violinplot(data)
        Depression_barplot(data)
        medication_barplot(data)
        Smoking_countplot(data)
        missing_value_treatment(data)
        label_encoder(data)
        model_LR(data)
        model_gradient(data)
        SGD(data)
        model_mlp(data)
        BaggingClassifier(data)
        GaussianProcessClassifier(data)
        QuadraticDiscriminantAnalysis(data)
        naivebayes(data)

    else:
        print("Failed to load data.")

if __name__ == "__main__":
    main()