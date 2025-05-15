import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


st.set_page_config(page_title="Loan Approval Predictor", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("loan_dataset.csv")

    df['Gender'] = df['Gender'].fillna('Male')
    df['Married'] = df['Married'].fillna('Yes')
    df['Dependents'] = df['Dependents'].fillna('0')
    df['Self_Employed'] = df['Self_Employed'].fillna('No')
    df['Credit_History'] = df['Credit_History'].fillna(1).astype(int)
    df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].mean())
    df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(360.0)

    le = LabelEncoder()
    for col in ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']:
        df[col] = le.fit_transform(df[col])

    df['Dep_1'] = df['Dependents'].apply(lambda x: 1 if x == '1' else 0)
    df['Dep_2'] = df['Dependents'].apply(lambda x: 1 if x == '2' else 0)
    df['Dep_3'] = df['Dependents'].apply(lambda x: 1 if x in ['3+', '3'] else 0)
    df['Prop_Semiurban'] = df['Property_Area'].apply(lambda x: 1 if x == 1 else 0)
    df['Prop_Urban'] = df['Property_Area'].apply(lambda x: 1 if x == 2 else 0)

    return df

df = load_data()

# --- Features & Model ---
X = df[['Gender', 'Married', 'Education', 'Self_Employed', 'ApplicantIncome',
        'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History',
        'Prop_Semiurban', 'Prop_Urban', 'Dep_1', 'Dep_2', 'Dep_3', 'age']]
y = df['Loan_Status']

model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# --- Sidebar Navigation ---
page = st.sidebar.selectbox("Navigate", ["Home", "Graph"])

# --- Home Page ---
if page == "Home":
    st.markdown("""
        <div style='text-align: center;
                    font-size: 40px;
                    font-weight: bold;
                    color: #E55050;
                    font-family: "Segoe UI", sans-serif;
                    margin-bottom: 20px;'>
             Loan Approval Prediction
        </div>
    """, unsafe_allow_html=True)

    with st.form("loan_form"):
        st.markdown("""
        <div style='
            text-align: center;
            font-size: 24px;
            font-weight: bold;
            color: #FFFBDE;
            margin-bottom: 20px;
            font-family: "Segoe UI", sans-serif;
        '>
         Enter Loan Application Details
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("Gender", ["Select...", "Male", "Female"], index=0)
            age = st.text_input("Age", value="")
            married = st.selectbox("Married", ["Select...", "Yes", "No"], index=0)
            dependents = st.selectbox("Dependents", ["Select...", "0", "1", "2", "3+"], index=0)
            education = st.selectbox("Education", ["Select...", "Graduate", "Not Graduate"], index=0)
            self_employed = st.selectbox("Self Employed", ["Select...", "Yes", "No"], index=0)

        with col2:
            applicant_income = st.text_input("Applicant Income", value="")
            coapplicant_income = st.text_input("Coapplicant Income", value="")
            loan_amount = st.text_input("Loan Amount", value="")
            loan_term = st.text_input("Loan Term (in months)", value="")
            credit_history = st.selectbox("Credit History", ["Select...", "1", "0"], index=0)
            property_area = st.selectbox("Property Area", ["Select...", "Urban", "Semiurban", "Rural"], index=0)

        submitted = st.form_submit_button("Predict Loan Approval")

        if submitted:
            if "Select..." in [gender, married, dependents, education, self_employed, credit_history, property_area]:
                st.error("Please fill all the fields properly.")
            else:
                try:
                    age = int(age)
                    applicant_income = float(applicant_income)
                    coapplicant_income = float(coapplicant_income)
                    loan_amount = float(loan_amount)
                    loan_term = float(loan_term)
                    credit_history = int(credit_history)
                except ValueError:
                    st.error("Please enter valid numeric values.")
                    st.stop()

            # Encode inputs
            gender = 1 if gender == 'Male' else 0
            married = 1 if married == 'Yes' else 0
            education = 0 if education == 'Graduate' else 1
            self_employed = 1 if self_employed == 'Yes' else 0
            prop_semiurban = 1 if property_area.lower() == 'semiurban' else 0
            prop_urban = 1 if property_area.lower() == 'urban' else 0
            dep_1 = 1 if dependents == '1' else 0
            dep_2 = 1 if dependents == '2' else 0
            dep_3 = 1 if dependents in ['3+', '3'] else 0

            user_data = pd.DataFrame([[gender, married, education, self_employed, applicant_income,
                                       coapplicant_income, loan_amount, loan_term, credit_history,
                                       prop_semiurban, prop_urban, dep_1, dep_2, dep_3, age]],
                                     columns=X.columns)

            prediction = model.predict(user_data)[0]
            score = model.predict_proba(user_data)[0][1]

            result = "✅ Approved" if prediction == 1 else "❌ Not Approved"

            if score >= 0.85:
                suggestion = "Great! Your chances are very strong. 👍"
            elif score >= 0.65:
                suggestion = "Looks good! But a bit risky, keep documents strong. 🧐"
            elif score >= 0.4:
                suggestion = "Hmm, this is a borderline case. Improve credit or income. ⚠️"
            else:
                suggestion = "Low chances. Work on your credit score or lower loan amount. ❗"

            st.markdown(f""" <div class="output-box">
            📢 <strong>Prediction Result:</strong> {result}<br>
            💡 <strong>Suggestion:</strong> {suggestion} <br>
            🔍 <strong>Confidence:</strong> {score:.2f}
            </div>
            """, unsafe_allow_html=True)

elif page == "Graph":
    st.title("📊 Loan Data Visualizations")

    st.subheader("1. Applicant Income Distribution")
    fig1, ax1 = plt.subplots()
    sns.histplot(df['ApplicantIncome'], bins=30, kde=True, color='skyblue', ax=ax1)
    ax1.set_title("Applicant Income Distribution")
    st.pyplot(fig1)

    st.subheader("2. Applicant vs Coapplicant Income")
    fig2, ax2 = plt.subplots()
    sns.boxplot(data=df[['ApplicantIncome', 'CoapplicantIncome']], palette='Set2', ax=ax2)
    ax2.set_title("Income Comparison")
    st.pyplot(fig2)

    st.subheader("3. Average Loan Amount by Credit History")
    avg_loan = df.groupby('Credit_History')['LoanAmount'].mean().reset_index()
    fig3, ax3 = plt.subplots()
    ax3.bar(avg_loan['Credit_History'].astype(str), avg_loan['LoanAmount'], color='lightgreen')
    ax3.set_xlabel("Credit History (0 = Bad, 1 = Good)")
    ax3.set_ylabel("Average Loan Amount")
    ax3.set_title("Loan Amount vs Credit History")
    st.pyplot(fig3)

# --- CSS ---
st.markdown("""
<style>
/* Make sure the button inside the form is styled and centered */
.stForm button {
    display: block;
    margin: 30px auto 10px auto;
    width: 100%;
    height: 3rem;
    background-color: #6a1b9a;  /* Purple */
    color: #FFEDFA !important;
    font-weight: 2000 !important;
    font-size: 1.2rem;
    border-radius: 8px;
    border: none;
    cursor: pointer;
    transition: background-color 0.3s ease, box-shadow 0.3s ease;
    box-shadow: 0 0 10px rgba(106, 27, 154, 0.5);  /* Soft purple glow */
}

/* Hover effect */
.stForm button:hover {
    background-color: #ab47bc;  /* Violet-magenta */
    color: black !important;
    box-shadow: 0 0 20px rgba(171, 71, 188, 0.7);  /* Brighter glow */
}

/* Output Box */
.output-box {
    background: #393E46;
    border-left: 6px solid #c2185b;
    padding: 20px;
    border-radius: 10px;
    font-size: 18px;
    font-family: 'Segoe UI', sans-serif;
    margin-top: 20px;
    color: white;
    box-shadow: 2px 2px 12px rgba(194, 24, 91, 0.1);
}
</style>
""", unsafe_allow_html=True)
