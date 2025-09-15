# -----------------------------------
# early_disease_prediction/streamlit_app.py
# -----------------------------------

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

# -----------------------------------
# Streamlit Title
# -----------------------------------
st.title("Early Disease Prediction System")
st.write("This app predicts disease likelihood based on health parameters using multiple ML models.")

# -----------------------------------
# Load dataset
# -----------------------------------
data = pd.read_csv("Data_file.csv")

# Preprocessing
df = data.drop(['date', 'country', 'occupation'], axis=1)
df['age'] = (df['age'] / 365).astype(int)
df['height'] = df['height'] / 100
df.drop("id", axis=1, inplace=True)

# Feature Engineering
df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']
df['cholesterol'] = df['cholesterol'].map({1: 'normal', 2: 'above_normal', 3: 'well_above_normal'})
df['gluc'] = df['gluc'].map({1: 'normal', 2: 'above_normal', 3: 'well_above_normal'})
df['map'] = df['ap_lo'] + (df['ap_hi'] - df['ap_lo']) / 3
df['bmi'] = df['weight'] / (df['height'])**2
df['sys_dsys_ratio'] = df['ap_hi'] / df['ap_lo']

# Features & Target
features = df[['age','weight','cholesterol','gluc','smoke','alco','pulse_pressure','map','bmi']]
target = df['disease']

num_features = features.select_dtypes(exclude='object').columns.tolist()
cat_features = features.select_dtypes(include='object').columns.tolist()

scaler = StandardScaler()
features[num_features] = scaler.fit_transform(features[num_features])
features = pd.get_dummies(features, columns=['cholesterol', 'gluc'])

x_train, x_test, y_train, y_test = train_test_split(features, target, random_state=42, test_size=0.33)

# -----------------------------------
# Train Models
# -----------------------------------
values, labels = [], []

# Decision Tree
dt_model = DecisionTreeClassifier(random_state=42, max_depth=9, min_samples_split=0.01, min_samples_leaf=0.01)
dt_model.fit(x_train, y_train)
y_pred_dt = dt_model.predict(x_test)
acc_dt = accuracy_score(y_test, y_pred_dt) * 100
values.append(acc_dt)
labels.append("Decision Tree")

# Logistic Regression
log_model = LogisticRegression(max_iter=1000, random_state=42)
log_model.fit(x_train, y_train)
y_pred_log = log_model.predict(x_test)
acc_log = accuracy_score(y_test, y_pred_log) * 100
values.append(acc_log)
labels.append("Logistic Regression")

# KNN
knn_model = KNeighborsClassifier(n_neighbors=7)
knn_model.fit(x_train, y_train)
y_pred_knn = knn_model.predict(x_test)
acc_knn = accuracy_score(y_test, y_pred_knn) * 100
values.append(acc_knn)
labels.append("KNN")

# -----------------------------------
# Show Accuracy Comparison
# -----------------------------------
st.subheader("Model Accuracy Comparison")
acc_df = pd.DataFrame({"Model": labels, "Accuracy": values})
st.bar_chart(acc_df.set_index("Model"))

# Best model
best_model_name = labels[np.argmax(values)]
st.info(f"Best Model Based on Accuracy: {best_model_name}")

# -----------------------------------
# User Input Section
# -----------------------------------
st.subheader("Enter Patient Details")

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

    # Same preprocessing as training
    new_data['age'] = (new_data['age'] / 365).astype(int)
    new_data['pulse_pressure'] = new_data['ap_hi'] - new_data['ap_lo']
    new_data['map'] = new_data['ap_lo'] + (new_data['ap_hi'] - new_data['ap_lo']) / 3
    new_data['bmi'] = new_data['weight'] / (new_data['height'] ** 2)
    new_data['sys_dsys_ratio'] = new_data['ap_hi'] / new_data['ap_lo']

    # Map categorical features
    new_data['cholesterol'] = new_data['cholesterol'].map({
        1: 'normal',
        2: 'above_normal',
        3: 'well_above_normal'
    })
    new_data['gluc'] = new_data['gluc'].map({
        1: 'normal',
        2: 'above_normal',
        3: 'well_above_normal'
    })
    new_data['smoke'] = new_data['smoke'].map({0: 'non_smoker', 1: 'smoker'})
    new_data['alco'] = new_data['alco'].map({0: 'non_drinker', 1: 'drinker'})
    new_data['active'] = new_data['active'].map({0: 'inactive', 1: 'active'})

    # Scale numerical features
    new_data[num_features] = scaler.transform(new_data[num_features])

    # One-hot encode categorical features
    new_data = pd.get_dummies(new_data, columns=['cholesterol', 'gluc'])

    # Align new_data with training features (to avoid column mismatch)
    new_data = new_data.reindex(columns=features.columns, fill_value=0)

    # Prediction using the best model
    if best_model_name == "Decision Tree":
        prediction = dt_model.predict(new_data)[0]
    elif best_model_name == "Logistic Regression":
        prediction = log_model.predict(new_data)[0]
    else:  # KNN
        prediction = knn_model.predict(new_data)[0]

    st.success(f"Disease Prediction: {'Yes' if prediction==1 else 'No'}")
