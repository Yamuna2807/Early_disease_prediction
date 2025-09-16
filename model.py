# -----------------------------------
# model.py - Early Disease Prediction
# -----------------------------------

import os
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# -----------------------------------
# Load Dataset
# -----------------------------------
file_path = os.path.join(os.path.dirname(__file__), "heart_disease_data.csv")
data = pd.read_csv(file_path)

st.title("🩺 Early Disease Prediction using Machine Learning")
st.write("This app predicts the likelihood of **Heart Disease** based on input health data.")

# Show dataset preview
if st.checkbox("Show Dataset Preview"):
    st.dataframe(data.head())

# -----------------------------------
# EDA
# -----------------------------------
if st.checkbox("Show EDA (Exploratory Data Analysis)"):
    st.subheader("Class Distribution")
    st.bar_chart(data['target'].value_counts())

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(data.corr(), annot=False, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# -----------------------------------
# Features & Target
# -----------------------------------
X = data.drop("target", axis=1)
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------------
# Train Model (Logistic Regression)
# -----------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)

st.subheader("📊 Model Performance")
st.write("Accuracy:", accuracy_score(y_test, y_pred))
st.write("Precision:", precision_score(y_test, y_pred))
st.write("Recall:", recall_score(y_test, y_pred))
st.write("F1 Score:", f1_score(y_test, y_pred))

# -----------------------------------
# User Input
# -----------------------------------
st.sidebar.header("Enter Patient Data")

def user_input_features():
    values = []
    for col in X.columns:
        values.append(st.sidebar.number_input(f"Enter {col}:", float(data[col].min()), float(data[col].max()), float(data[col].mean())))
    features = pd.DataFrame([values], columns=X.columns)
    return features

user_data = user_input_features()

# -----------------------------------
# Prediction
# -----------------------------------
if st.button("🔍 Predict"):
    user_scaled = scaler.transform(user_data)
    prediction = model.predict(user_scaled)
    proba = model.predict_proba(user_scaled)[0][1]  # probability of disease

    if prediction[0] == 0:
        st.success(f"🟢 The model predicts: **No Disease** (Risk: {proba*100:.2f}%)")
    else:
        st.error(f"🔴 The model predicts: **Disease** (Risk: {proba*100:.2f}%)")
