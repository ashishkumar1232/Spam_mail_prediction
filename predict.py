import joblib

# Load saved model
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def predict_email(text):

    features = vectorizer.transform([text])

    prediction = model.predict(features)[0]

    if prediction == 1:
        return "Ham"
    else:
        return "Spam"

if __name__ == "__main__":

    email = input("Enter Email:\n")

    result = predict_email(email)

    print("Prediction:", result)