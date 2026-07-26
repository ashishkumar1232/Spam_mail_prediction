import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.set_page_config(
    page_title="Spam Email Detector",
    page_icon="📧",
    layout="centered"
)

st.title("📧 Spam Email Detection")
st.write("Enter an email message below to check whether it is **Spam** or **Ham**.")

email = st.text_area("Email Message", height=200)

if st.button("Predict"):

    if email.strip() == "":
        st.warning("Please enter an email.")
    else:
        features = vectorizer.transform([email])

        prediction = model.predict(features)[0]

        if prediction == 1:
            st.success("✅ Ham Email")
        else:
            st.error("🚨 Spam Email")