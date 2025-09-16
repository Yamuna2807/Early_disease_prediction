# -----------------------------------
# Early Disease Prediction App
# -----------------------------------

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# -----------------------------------
# 1. Title
# -----------------------------------
st.title("🩺 Early Disease Prediction using Machine Learning")

# -----------------------------------
# 2. Upload Dataset
# -----------------------------------
st.sidebar.header("Upload Your CSV File")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader("📊 Dataset Preview")
    st.dataframe(data.head())

    # -----------------------------------
    # 3. EDA (Exploratory Data Analysis)
    # -----------------------------------
    st.subheader("🔎 Exploratory Data Analysis")

    st.write("**Shape of Dataset:**", data.shape)
    st.write("**Columns:**", list(data.columns))

    # Correlation Heatmap
    st.write("**Correlation Heatmap**")
    plt.figure(figsize=(8, 6))
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
    st.pyplot(plt)

    # Target column
    target_col = st.sidebar.selectbox("Select Target Column (Disease/No Disease)", data.columns)

    # Features and Target
    X = data.drop(columns=[target_col])
    y = data[target_col]

    # Standardize Features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # -----------------------------------
    # 4. Model Selection
    # -----------------------------------
    st.sidebar.subheader("Choose a Model")
    model_choice = st.sidebar.selectbox("Select Model", ["Logistic Regression", "Decision Tree", "KNN"])

    if model_choice == "Logistic Regression":
        model = LogisticRegression()
    elif model_choice == "Decision Tree":
        model = DecisionTreeClassifier()
    else:
        model = KNeighborsClassifier()

    # Train Model
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # -----------------------------------
    # 5. Model Evaluation
    # -----------------------------------
    st.subheader("📈 Model Evaluation")
    st.write("**Accuracy:**", accuracy_score(y_test, y_pred))
    st.write("**Precision:**", precision_score(y_test, y_pred, average="weighted"))
    st.write("**Recall:**", recall_score(y_test, y_pred, average="weighted"))
    st.write("**F1 Score:**", f1_score(y_test, y_pred, average="weighted"))

    # -----------------------------------
    # 6. User Prediction
    # -----------------------------------
    st.subheader("🤖 Predict for a New Person")

    # User Inputs
    user_input = []
    for col in X.columns:
        val = st.number_input(f"Enter {col}", value=0.0)
        user_input.append(val)

    if st.button("Predict"):
        user_data = np.array(user_input).reshape(1, -1)
        user_data_scaled = scaler.transform(user_data)
        prediction = model.predict(user_data_scaled)[0]

        if prediction == 0:
            st.success("✅ Prediction: No Disease")
        else:
            st.error("⚠️ Prediction: Disease Detected")

else:
    st.info("👈 Please upload a CSV file to start.")
