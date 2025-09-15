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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------------------------------------------------
# 2. Streamlit App Layout
# ---------------------------------------------------
st.title("🩺 Disease Prediction Dashboard")

st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Load dataset
    data = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.write(data.head())
    st.write("Shape:", data.shape)

    # Feature Engineering
    data['age_years'] = (data['age'] / 365).astype(int)
    data['ap_hi'] = data['ap_hi'].clip(80, 250)
    data['ap_lo'] = data['ap_lo'].clip(40, 150)

    # Encode categorical features
    label_encoders = {}
    for col in ["country", "occupation"]:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

    # Drop unused columns
    data = data.drop(["date", "id"], axis=1)

    # Features & Target
    X = data.drop("disease", axis=1)
    y = data["disease"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Standardization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # -----------------------------------
    # 4. Train Models
    # -----------------------------------
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=10),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5)
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
    
        st.subheader(f"🔹 {name} Results")
        st.write(f"**Accuracy:** {acc*100:.2f}%")
        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred))
    
        # Confusion Matrix Plot
        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title(f"{name} Confusion Matrix")
        st.pyplot(fig)

    # -----------------------------------
    # 5. Comparison Chart
    # -----------------------------------
    st.subheader("📊 Model Accuracy Comparison")
    fig, ax = plt.subplots()
    ax.bar(results.keys(), results.values(), color=["blue", "green", "orange"])
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    st.pyplot(fig)

    # ---------------------------------------------------
    # 6. User Input for Disease Prediction
    # ---------------------------------------------------
    st.subheader("📝 Predict Disease for a New Patient")

    with st.form("prediction_form"):
        age = st.number_input("Age (days)", min_value=0, max_value=150*365, value=365*30)
        gender = st.selectbox("Gender", options=[0, 1])  # Assuming 0 = Female, 1 = Male
        height = st.number_input("Height (cm)", min_value=50, max_value=250, value=170)
        weight = st.number_input("Weight (kg)", min_value=20, max_value=200, value=70)
        ap_hi = st.number_input("Systolic BP (ap_hi)", min_value=80, max_value=250, value=120)
        ap_lo = st.number_input("Diastolic BP (ap_lo)", min_value=40, max_value=150, value=80)
        cholesterol = st.selectbox("Cholesterol (1=Normal,2=Above Normal,3=High)", options=[1,2,3])
        gluc = st.selectbox("Glucose (1=Normal,2=Above Normal,3=High)", options=[1,2,3])
        smoke = st.selectbox("Smoke (0=No,1=Yes)", options=[0,1])
        alco = st.selectbox("Alcohol (0=No,1=Yes)", options=[0,1])
        active = st.selectbox("Physical Activity (0=No,1=Yes)", options=[0,1])
        country = st.selectbox("Country", options=list(label_encoders['country'].classes_))
        occupation = st.selectbox("Occupation", options=list(label_encoders['occupation'].classes_))
        submitted = st.form_submit_button("Predict Disease")

        if submitted:
            # Encode categorical inputs
            country_val = label_encoders['country'].transform([country])[0]
            occupation_val = label_encoders['occupation'].transform([occupation])[0]
            age_years = age // 365
            ap_hi = np.clip(ap_hi, 80, 250)
            ap_lo = np.clip(ap_lo, 40, 150)

            # Create input array
            input_data = np.array([[age_years, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active, country_val, occupation_val]])
            input_data = scaler.transform(input_data)

            # Predict with the best model (Decision Tree here)
            prediction = models["Decision Tree"].predict(input_data)[0]

            st.success(f"✅ Predicted Disease Status: {'Disease' if prediction==1 else 'No Disease'}")

    # ---------------------------------------------------
    # 7. Optional EDA Visualizations
    # ---------------------------------------------------
    if st.checkbox("Show EDA Visualizations"):
        st.subheader("Disease Distribution")
        fig, ax = plt.subplots()
        sns.countplot(x='disease', data=data, ax=ax)
        st.pyplot(fig)

        for column in ['age_years', 'ap_hi', 'ap_lo', 'height', 'weight']:
            st.write(f"### Histogram of {column}")
            fig, ax = plt.subplots(figsize=(6,4))
            sns.histplot(data=data, x=column, hue="disease", multiple="stack", kde=True, bins=30, ax=ax)
            ax.set_title(f"{column} Distribution by Disease")
            st.pyplot(fig)

        for column in ['age_years', 'ap_hi', 'ap_lo', 'height', 'weight']:    
            st.write(f"### Boxplot of {column}")
            fig, ax = plt.subplots(figsize=(6,4))
            sns.boxplot(data=data, x='disease', y=column, ax=ax)
            ax.set_title(f"{column} Distribution by Disease")
            st.pyplot(fig)

        st.subheader("Correlation Matrix")
        num_cols = [col for col in data.columns if data[col].dtype != object]
        correlation_matrix = data[num_cols].corr()
        fig, ax = plt.subplots(figsize=(12, 7))
        sns.heatmap(correlation_matrix.round(2), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
