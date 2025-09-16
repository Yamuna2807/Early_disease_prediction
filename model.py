import streamlit as st
import numpy as np
import pandas as pd
import os

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -------------------
# Load & Preprocess Data
# -------------------
# Safe path (dataset must be in same folder as this script)
file_path = os.path.join(os.path.dirname(__file__), "heart_disease_data.csv")
data = pd.read_csv(file_path)

# Drop unused columns
df = data.drop(['date','country','occupation'], axis=1)
df['age'] = (df['age'] / 365).astype(int)
df['height'] = df['height'] / 100
df.drop("id", axis=1, inplace=True)

# Feature Engineering
df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']
df['cholesterol'] = df['cholesterol'].apply(
    lambda x: 'normal' if x == 1 else ('above_normal' if x == 2 else 'well_above_normal')
)
df['gluc'] = df['gluc'].apply(
    lambda x: 'normal' if x == 1 else ('above_normal' if x == 2 else 'well_above_normal')
)
df['map'] = df['ap_lo'] + (df['ap_hi'] - df['ap_lo']) / 3
df['bmi'] = df['weight'] / (df['height'])**2

# Final feature set (NO unused columns)
features = df[['age', 'weight', 'cholesterol', 'gluc',
               'smoke', 'alco', 'pulse_pressure', 'map', 'bmi']]
target = df['disease']

# Scale numeric features
num_features = features.select_dtypes(exclude='object').columns
scaler = StandardScaler()
features[num_features] = scaler.fit_transform(features[num_features])
features = pd.get_dummies(features, columns=['cholesterol', 'gluc'])

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(
    features, target, random_state=42, test_size=0.33
)

# -------------------
# Train Models
# -------------------
results = {}

# Decision Tree
dt_model = DecisionTreeClassifier(random_state=42, max_depth=9,
                                  min_samples_split=0.01, min_samples_leaf=0.01)
dt_model.fit(x_train, y_train)
results["Decision Tree"] = accuracy_score(y_test, dt_model.predict(x_test)) * 100

# Logistic Regression
log_model = LogisticRegression(max_iter=1000, random_state=42)
log_model.fit(x_train, y_train)
results["Logistic Regression"] = accuracy_score(y_test, log_model.predict(x_test)) * 100

# KNN
knn_model = KNeighborsClassifier(n_neighbors=7)
knn_model.fit(x_train, y_train)
results["KNN"] = accuracy_score(y_test, knn_model.predict(x_test)) * 100

# Best model
best_model_name = max(results, key=results.get)
best_accuracy = results[best_model_name]

# -------------------
# Streamlit UI
# -------------------
st.title("🩺 Disease Prediction App")

st.subheader("📊 Model Performance")
for model, acc in results.items():
    st.write(f"**{model} Accuracy:** {acc:.2f}%")
st.success(f"✅ Best Model: {best_model_name} ({best_accuracy:.2f}%)")

st.subheader("🔹 Enter Patient Details")

age_days = st.number_input("Age in days", min_value=1, max_value=40000, value=18250)
height_cm = st.number_input("Height (cm)", min_value=50, max_value=250, value=170)
weight = st.number_input("Weight (kg)", min_value=10, max_value=200, value=70)
ap_hi = st.number_input("Systolic BP (ap_hi)", min_value=50, max_value=250, value=120)
ap_lo = st.number_input("Diastolic BP (ap_lo)", min_value=30, max_value=200, value=80)
cholesterol = st.selectbox(
    "Cholesterol", [1, 2, 3],
    format_func=lambda x: {1: "Normal", 2: "Above Normal", 3: "Well Above Normal"}[x]
)
gluc = st.selectbox(
    "Glucose", [1, 2, 3],
    format_func=lambda x: {1: "Normal", 2: "Above Normal", 3: "Well Above Normal"}[x]
)
smoke = st.radio("Smoking?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
alco = st.radio("Alcohol intake?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

if st.button("Predict"):
    # Build input data
    new_data = pd.DataFrame([{
        "age": (age_days / 365),   # keep consistent with training
        "weight": weight,
        "ap_hi": ap_hi,
        "ap_lo": ap_lo,
        "cholesterol": cholesterol,
        "gluc": gluc,
        "smoke": smoke,
        "alco": alco,
        "height": height_cm / 100
    }])

    # Feature Engineering
    new_data['pulse_pressure'] = new_data['ap_hi'] - new_data['ap_lo']
    new_data['cholesterol'] = new_data['cholesterol'].apply(
        lambda x: 'normal' if x == 1 else ('above_normal' if x == 2 else 'well_above_normal')
    )
    new_data['gluc'] = new_data['gluc'].apply(
        lambda x: 'normal' if x == 1 else ('above_normal' if x == 2 else 'well_above_normal')
    )
    new_data['map'] = new_data['ap_lo'] + (new_data['ap_hi'] - new_data['ap_lo']) / 3
    new_data['bmi'] = new_data['weight'] / (new_data['height'])**2

    # Select only final features
    new_features = new_data[['age','weight','cholesterol','gluc',
                             'smoke','alco','pulse_pressure','map','bmi']]

    # Scale numeric
    new_features[num_features] = scaler.transform(new_features[num_features])

    # One-hot encode
    new_features = pd.get_dummies(new_features, columns=['cholesterol','gluc'])

    # Align with training features
    new_features = new_features.reindex(columns=features.columns, fill_value=0)

    # Predictions
    pred_dt = dt_model.predict(new_features)[0]
    pred_log = log_model.predict(new_features)[0]
    pred_knn = knn_model.predict(new_features)[0]

    st.subheader("🔮 Predictions")
    st.write(f"**Decision Tree:** {'Disease' if pred_dt==1 else 'No Disease'}")
    st.write(f"**Logistic Regression:** {'Disease' if pred_log==1 else 'No Disease'}")
    st.write(f"**KNN:** {'Disease' if pred_knn==1 else 'No Disease'}")

    # Best model result
    best_var = {"decision": pred_dt, "logistic": pred_log, "knn": pred_knn}
    key = best_model_name.split()[0].lower()
    st.success(f"✅ Recommended Model: {best_model_name} → "
               f"{'Disease' if best_var[key]==1 else 'No Disease'}")
