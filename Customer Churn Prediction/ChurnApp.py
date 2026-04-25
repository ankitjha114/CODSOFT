import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.metrics import roc_curve, auc, precision_recall_curve

st.set_page_config(
    page_title="Customer Churn AI Dashboard",
    layout="wide"
)

@st.cache_resource
def load_model():
    with open("customer_churn_model.pkl", "rb") as f:
        model_data = pickle.load(f)
    return model_data["model"], model_data["features_names"]

@st.cache_resource
def load_encoders():
    with open("encoders.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    df = pd.read_csv("Churn_Modelling.csv")
    df = df.drop(columns=["RowNumber", "CustomerId", "Surname"], errors="ignore")
    return df

model, feature_names = load_model()
encoders = load_encoders()
df = load_data()

st.markdown("""
<style>
h1 { text-align:center; color:#00C9A7; }
.stButton>button {
    background-color:#00C9A7;
    color:black;
    font-weight:bold;
    border-radius:10px;
}
</style>
""", unsafe_allow_html=True)

st.title("Customer Churn Prediction Dashboard")

tabs = st.tabs(["Prediction", "Insights", "Model Analysis"])

with tabs[0]:

    st.header("Enter Customer Details")

    col1, col2 = st.columns(2)

    with col1:
        credit_score = st.number_input("Credit Score", 300, 900, 600)
        geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.slider("Age", 18, 100, 30)
        tenure = st.slider("Tenure", 0, 10, 3)

    with col2:
        balance = st.slider("Balance", 0.0, 250000.0, 50000.0, step=500.0)
        num_products = st.slider("Products", 1, 4, 2)
        has_card = st.selectbox("Has Credit Card", [0, 1])
        is_active = st.selectbox("Active Member", [0, 1])
        salary = st.slider("Estimated Salary", 0.0, 200000.0, 50000.0, step=500.0)

    if st.button("Predict Churn"):

        input_data = pd.DataFrame([{
            'CreditScore': credit_score,
            'Geography': geography,
            'Gender': gender,
            'Age': age,
            'Tenure': tenure,
            'Balance': balance,
            'NumOfProducts': num_products,
            'HasCrCard': has_card,
            'IsActiveMember': is_active,
            'EstimatedSalary': salary
        }])

        for col, encoder in encoders.items():
            if col in input_data.columns:
                input_data[col] = encoder.transform(input_data[col])

        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]

        st.markdown("---")

        if prediction == 1:
            st.error("Customer likely to CHURN")
        else:
            st.success("Customer likely to STAY")

        st.metric("Churn Probability", f"{prob*100:.2f}%")
        st.progress(int(prob * 100))

        st.subheader("Why this prediction?")

        importances = model.feature_importances_
        feat_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)

        fig, ax = plt.subplots(figsize=(8,5))
        sns.barplot(x="Importance", y="Feature", data=feat_df.head(10), ax=ax)
        ax.set_title("Top Factors Influencing Churn")
        st.pyplot(fig)

        st.subheader("Retention Strategy")

        suggestions = []

        if balance > 100000:
            suggestions.append("Offer premium banking benefits or loyalty rewards")

        if is_active == 0:
            suggestions.append("Engage customer with offers & notifications")

        if num_products <= 1:
            suggestions.append("Cross-sell more products")

        if age > 50:
            suggestions.append("Provide personalized senior customer plans")

        if not suggestions:
            suggestions.append("Customer is stable, maintain engagement")

        for s in suggestions:
            st.info(s)

with tabs[1]:

    st.markdown("## Customer Insights Dashboard")
    st.markdown("---")

    churn_rate = df["Exited"].mean() * 100
    st.metric("Overall Churn Rate", f"{churn_rate:.2f}%")

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(10,5))
        sns.histplot(df["Age"], kde=True, ax=ax)
        ax.set_title("Age Distribution")
        st.pyplot(fig)
        st.info("Customers aged 40+ show higher churn tendency")

    with col2:
        fig, ax = plt.subplots(figsize=(10,5))
        sns.boxplot(x=df["Exited"], y=df["Balance"], ax=ax)
        ax.set_title("Balance vs Churn")
        st.pyplot(fig)
        st.info("High balance customers are more likely to churn")

with tabs[2]:

    st.header("Model Performance Analysis")

    df_model = df.copy()

    for col in encoders:
        if col in df_model.columns:
            df_model[col] = encoders[col].transform(df_model[col])

    df_model = df_model[feature_names + ["Exited"]]

    X = df_model.drop("Exited", axis=1)
    y = df_model["Exited"]

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model.fit(X_train, y_train)
    y_probs = model.predict_proba(X_test)[:,1]

    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax.plot([0,1], [0,1], linestyle="--")
    ax.set_title("ROC Curve")
    ax.legend()
    st.pyplot(fig)

    precision, recall, _ = precision_recall_curve(y_test, y_probs)

    fig, ax = plt.subplots()
    ax.plot(recall, precision)
    ax.set_title("Precision-Recall Curve")
    st.pyplot(fig)