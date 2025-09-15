# -----------------------------------
# early_disease_prediction/streamlit_app.py
# -----------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

# -----------------------------------
# Load dataset
# -----------------------------------
st.title("Early Disease Prediction System")
st.write("This app predicts disease likelihood based on health parameters using multiple ML models.")

data = pd.read_csv("heart_disease_data.csv")

# Preprocessing
df = data.drop(['date', 'country', 'occupation'], axis=1)
df['age'] = df['age'].apply(lambda x: x/365).astype(int)
df['height'] = df['height'] / 100
df.drop("id", axis=1, inplace=True)

# Feature Engineering
df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']
df['cholesterol'] = df['cholesterol'].apply(lambda x: 'normal' if x==1 else('above_normal' if x==2 else 'well_above_normal'))
df['gluc'] = df['gluc'].apply(lambda x: 'normal' if x==1 else('above_normal' if x==2 else 'well_above_normal'))
df['map'] = df['ap_lo'] + (df['ap_hi'] - df['ap_lo'])/3
df['bmi'] = df['weight'] / (df['height'])**2
df['sys_dsys_ratio'] = df['ap_hi'] / df['ap_lo']

# Features & Target
features = df[['age','weight', 'cholesterol', 'gluc', 'smoke', 'alco','pulse_pressure', 'map', 'bmi']]
target = df['disease']

num_features = features.select_dtypes(exclude='object').columns
cat_features = features.select_dtypes(include='object').columns 

scaler = StandardScaler()
features[num_features] = scaler.fit_transform(features[num_features])
features = pd.get_dummies(features, columns=['cholesterol', 'gluc'])

x_train, x_test, y_train, y_test = train_test_split(features, target, random_state=42, test_size=0.33)

# -----------------------------------
# Train Models
# -----------------------------------
values = []
labels = []

# Decision Tree
dt_model = DecisionTreeClassifier(random_state=42, max_depth=9, min_samples_split=0.01, min_samples_leaf=0.01)
dt_model.fit(x_train, y_train)
y_pred_dt = dt_model.predict(x_test)
acc_dt = accuracy_score(y_test, y_pred_dt)*100
values.append(acc_dt)
labels.append("Decision Tree")

# Logistic Regression
log_model = LogisticRegression(max_iter=1000, random_state=42)
log_model.fit(x_train, y_train)
y_pred_log = log_model.predict(x_test)
acc_log = accuracy_score(y_test, y_pred_log)*100
values.append(acc_log)
labels.append("Logistic Regression")

# KNN
knn_model = KNeighborsClassifier(n_neighbors=7)
knn_model.fit(x_train, y_train)
y_pred_knn = knn_model.predict(x_test)
acc_knn = accuracy_score(y_test, y_pred_knn)*100
values.append(acc_knn)
labels.append("KNN")

# -----------------------------------
# Show Accuracy Comparison
# -----------------------------------
st.subheader(" Model Accuracy Comparison")
acc_df = pd.DataFrame({"Model": labels, "Accuracy": values})
st.bar_chart(acc_df.set_index("Model"))

# Best model
best_model_name = labels[np.argmax(values)]
st.info(f" Best Model Based on Accuracy: {best_model_name}")

# -----------------------------------
# User Input Section
# -----------------------------------
st.subheader(" Enter Patient Details")

age = st.number_input("Age (in days)", min_value=1000, max_value=40000, step=1)
height = st.number_input("Height (cm)", min_value=50, max_value=250, step=1)
weight = st.number_input("Weight (kg)", min_value=10, max_value=200, step=1)
ap_hi = st.number_input("Systolic BP", min_value=50, max_value=250, step=1)
ap_lo = st.number_input("Diastolic BP", min_value=30, max_value=200, step=1)

cholesterol = st.selectbox(
    "Cholesterol Level", [1, 2, 3],
    format_func=lambda x: {1:"Normal",2:"Above Normal",3:"Well Above Normal"}[x]
)
gluc = st.selectbox(
    "Glucose Level", [1, 2, 3],
    format_func=lambda x: {1:"Normal",2:"Above Normal",3:"Well Above Normal"}[x]
)

smoke = st.radio("Smoking?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
alco = st.radio("Alcohol intake?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
active = st.radio("Physically active?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")


# -----------------------------------
# Predict Button
# -----------------------------------
if st.button("Predict Disease"):
    new_data = pd.DataFrame([{
        "age": age,
        "height": height/100,
        "weight": weight,
        "ap_hi": ap_hi,
        "ap_lo": ap_lo,
        "cholesterol": cholesterol,
        "gluc": gluc,
        "smoke": smoke,
        "alco": alco,
        "active": active
    }])

    # Same preprocessing
    new_data['age'] = (new_data['age']/365).astype(int)
    new_data['pulse_pressure'] = new_data['ap_hi'] - new_data['ap_lo']
    new_data['cholesterol'] = new_data['cholesterol'].apply(lambda x: 'normal' if x==1 else ('above_normal' if x==2 else 'well_above_normal'))
    new_data['gluc'] = new_data['gluc'].apply(lambda x: 'normal' if x==1 else ('above_normal' if x==2 else 'well_above_normal'))
    new_data['map'] = new_data['ap_lo'] + (new_data['ap_hi'] - new_data['ap_lo'])/3
    new_data['bmi'] = new_data['weight'] / (new_data['height'])**2
    new_data['sys_dsys_ratio'] = new_data['ap_hi'] / new_data['ap_lo']

    new_features = new_data[['age','weight','cholesterol','gluc','smoke','alco','pulse_pressure','map','bmi']]
    new_features[num_features] = scaler.transform(new_features[num_features])
    new_features = pd.get_dummies(new_features, columns=['cholesterol', 'gluc'])
    new_features = new_features.reindex(columns=features.columns, fill_value=0)

    # Predictions
    pred_dt = dt_model.predict(new_features)[0]
    pred_log = log_model.predict(new_features)[0]
    pred_knn = knn_model.predict(new_features)[0]

    # Collect predictions
    predictions = {
        "Decision Tree": pred_dt,
        "Logistic Regression": pred_log,
        "KNN": pred_knn
    }

    # Show results
    st.subheader(" Predictions")
    for model_name, pred in predictions.items():
        st.write(f"**{model_name}** → {'Disease' if pred==1 else 'No Disease'}")

    # Best model’s prediction
    best_prediction = predictions[best_model_name]
    st.success(f"Recommended Model: {best_model_name} → {'Disease' if best_prediction==1 else 'No Disease'}")


            
