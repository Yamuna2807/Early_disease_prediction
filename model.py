# app.py
# ---------------------------------------------------
# 1. Import Libraries
# ---------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# ---------------------------------------------------
# 2. Streamlit Layout
# ---------------------------------------------------
st.title("🩺 Disease Prediction Dashboard")

uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # -----------------------------------
    # 3. Load Dataset
    # -----------------------------------
    data = pd.read_csv(uploaded_file)

    # Feature Engineering
    data['age_years'] = (data['age'] / 365).astype(int)
    data['ap_hi'] = data['ap_hi'].clip(80, 250)
    data['ap_lo'] = data['ap_lo'].clip(40, 150)

    # Drop unnecessary columns (as in your code)
    data = data.drop(["date", "id", "age", "occupation", "country"], axis=1, errors="ignore")

    st.subheader("Dataset Preview")
    st.write(data.head())
    st.write("Shape:", data.shape)

    # -----------------------------------
    # 4. Features & Target
    # -----------------------------------
    X = data.drop("disease", axis=1)
    y = data["disease"]
    feature_names = X.columns.tolist()   # save feature order for prediction

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scaling for Logistic Regression & KNN
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # -----------------------------------
    # 5. Train Models
    # -----------------------------------
    dt_model = DecisionTreeClassifier(random_state=42, max_depth=10)
    dt_model.fit(X_train, y_train)

    log_model = LogisticRegression(max_iter=2000, random_state=42)
    log_model.fit(X_train_scaled, y_train)

    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train_scaled, y_train)

    # -----------------------------------
    # 6. Evaluation Results
    # -----------------------------------
    models = {
        "Decision Tree": (dt_model, X_test, y_test),
        "Logistic Regression": (log_model, X_test_scaled, y_test),
        "KNN": (knn_model, X_test_scaled, y_test)
    }

    st.subheader("🔹 Model Evaluation Results")
    results = {}

    for name, (model, X_eval, y_eval) in models.items():
        y_pred = model.predict(X_eval)
        acc = accuracy_score(y_eval, y_pred)
        results[name] = acc

        st.write(f"**{name}**")
        st.write(f"Accuracy: {acc*100:.2f}%")
        st.text(classification_report(y_eval, y_pred))

        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_eval, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title(f"{name} Confusion Matrix")
        st.pyplot(fig)

    # -----------------------------------
    # 7. Accuracy Comparison
    # -----------------------------------
    st.subheader("📊 Model Accuracy Comparison")
    fig, ax = plt.subplots()
    ax.bar(results.keys(), results.values(), color=["skyblue","lightgreen","lightcoral"])
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    st.pyplot(fig)

    # -----------------------------------
    # 8. Prediction from User Input
    # -----------------------------------
    st.subheader("📝 Predict Disease for a New Patient")

    with st.form("prediction_form"):
        # collect inputs for the features (based on your dataset)
        age_days = st.number_input("Age (days)", min_value=0, max_value=150*365, value=365*30)
        gender = st.selectbox("Gender (0=Female,1=Male)", [0,1])
        height = st.number_input("Height (cm)", min_value=50, max_value=250, value=170)
        weight = st.number_input("Weight (kg)", min_value=20, max_value=200, value=70)
        ap_hi = st.number_input("Systolic BP (ap_hi)", min_value=80, max_value=250, value=120)
        ap_lo = st.number_input("Diastolic BP (ap_lo)", min_value=40, max_value=150, value=80)
        cholesterol = st.selectbox("Cholesterol (1=Normal,2=Above Normal,3=High)", [1,2,3])
        gluc = st.selectbox("Glucose (1=Normal,2=Above Normal,3=High)", [1,2,3])
        smoke = st.selectbox("Smoke (0=No,1=Yes)", [0,1])
        alco = st.selectbox("Alcohol (0=No,1=Yes)", [0,1])
        active = st.selectbox("Physical Activity (0=No,1=Yes)", [0,1])

        submitted = st.form_submit_button("Predict")

        if submitted:
            # match training features
            input_dict = {
                "gender": gender,
                "height": height,
                "weight": weight,
                "ap_hi": ap_hi,
                "ap_lo": ap_lo,
                "cholesterol": cholesterol,
                "gluc": gluc,
                "smoke": smoke,
                "alco": alco,
                "active": active,
                "age_years": age_days // 365,
            }

            input_df = pd.DataFrame([input_dict])[feature_names]

            # scale for models that need scaling
            input_scaled = scaler.transform(input_df)

            # predict with all models
            preds = {
                "Decision Tree": dt_model.predict(input_df)[0],
                "Logistic Regression": log_model.predict(input_scaled)[0],
                "KNN": knn_model.predict(input_scaled)[0]
            }

            st.write("### 🔮 Predictions")
            for model_name, pred in preds.items():
                st.success(f"{model_name}: {'Disease' if pred==1 else 'No Disease'}")

    # -----------------------------------
    # 9. Optional EDA
    # -----------------------------------
    if st.checkbox("Show EDA Visualizations"):
        st.subheader("Disease Distribution")
        fig, ax = plt.subplots()
        sns.countplot(x='disease', data=data, ax=ax)
        st.pyplot(fig)

        for col in ['age_years', 'ap_hi', 'ap_lo', 'height', 'weight']:
            fig, ax = plt.subplots()
            sns.histplot(data=data, x=col, hue='disease', multiple='stack', kde=True, bins=30, ax=ax)
            ax.set_title(f"{col} Distribution by Disease")
            st.pyplot(fig)

        for col in ['age_years', 'ap_hi', 'ap_lo', 'height', 'weight']:
            fig, ax = plt.subplots()
            sns.boxplot(data=data, x='disease', y=col, ax=ax)
            ax.set_title(f"{col} Distribution by Disease")
            st.pyplot(fig)

        fig, ax = plt.subplots(figsize=(12,7))
        correlation_matrix = data.corr()
        sns.heatmap(correlation_matrix.round(2), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
