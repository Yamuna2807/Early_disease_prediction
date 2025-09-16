# -------------------
# model.py
# -------------------
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -------------------
# Load Data
# -------------------
data = pd.read_csv("heart_disease_data.csv")

# Auto-detect target column (last column of dataset)
target_col = data.columns[-1]
st.write("Detected Target Column:", target_col)

X = data.drop(target_col, axis=1)
y = data[target_col]

# Scale numeric features
num_features = X.select_dtypes(include=np.number).columns
scaler = StandardScaler()
X[num_features] = scaler.fit_transform(X[num_features])

# One-hot encode categorical features (if any)
X = pd.get_dummies(X)

# -------------------
# Train-Test Split
# -------------------
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# -------------------
# Train Models
# -------------------
results = {}

# Decision Tree
dt_model = DecisionTreeClassifier(random_state=42, max_depth=9)
dt_model.fit(x_train, y_train)
y_pred_dt = dt_model.predict(x_test)
results["Decision Tree"] = accuracy_score(y_test, y_pred_dt) * 100

# Logistic Regression
log_model = LogisticRegression(max_iter=1000, random_state=42)
log_model.fit(x_train, y_train)
y_pred_log = log_model.predict(x_test)
results["Logistic Regression"] = accuracy_score(y_test, y_pred_log) * 100

# KNN
knn_model = KNeighborsClassifier(n_neighbors=7)
knn_model.fit(x_train, y_train)
y_pred_knn = knn_model.predict(x_test)
results["KNN"] = accuracy_score(y_test, y_pred_knn) * 100

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

# Collect inputs
inputs = {}
for col in X.columns:
    if col in num_features:
        inputs[col] = st.number_input(f"{col}", value=0.0)
    else:
        options = [0, 1]
        inputs[col] = st.selectbox(f"{col}", options)

# Predict button
if st.button("Predict"):
    new_data = pd.DataFrame([inputs])
    new_data[num_features] = scaler.transform(new_data[num_features])

    # Ensure columns match training
    new_data = new_data.reindex(columns=X.columns, fill_value=0)

    # Predictions
    pred_dt = dt_model.predict(new_data)[0]
    pred_log = log_model.predict(new_data)[0]
    pred_knn = knn_model.predict(new_data)[0]

    # Probabilities
    proba_dt = dt_model.predict_proba(new_data)[0][1]
    proba_log = log_model.predict_proba(new_data)[0][1]
    proba_knn = knn_model.predict_proba(new_data)[0][1]

    # Function to convert 0/1 → human label
    def label_output(pred):
        return "No Disease" if pred == 0 else "Disease"

    st.subheader("🔮 Predictions")
    st.write(f"**Decision Tree:** {label_output(pred_dt)} (Risk: {proba_dt*100:.2f}%)")
    st.write(f"**Logistic Regression:** {label_output(pred_log)} (Risk: {proba_log*100:.2f}%)")
    st.write(f"**KNN:** {label_output(pred_knn)} (Risk: {proba_knn*100:.2f}%)")

    # Best model recommendation
    best_model_pred = {
        "Decision Tree": pred_dt,
        "Logistic Regression": pred_log,
        "KNN": pred_knn
    }[best_model_name]

    st.success(f"✅ Recommended Model: {best_model_name} → Prediction: {label_output(best_model_pred)}")
