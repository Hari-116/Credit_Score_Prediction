import joblib
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# ✅ Load trained model
model = joblib.load("credit_score_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # ✅ Get input values from the form
        annual_income = float(request.form["annual_income"])
        num_of_loans = int(request.form["num_of_loans"])
        num_of_delayed_payments = int(request.form["num_of_delayed_payments"])
        credit_history_age = float(request.form["credit_history_age"])

        # ✅ Create input array
        features = np.array([[annual_income, num_of_loans, num_of_delayed_payments, credit_history_age]])

        # ✅ Make prediction
        prediction = model.predict(features)[0]

        # ✅ Convert prediction to label
        credit_score_labels = {0: "Poor", 1: "Standard", 2: "Good"}
        predicted_label = credit_score_labels.get(prediction, "Unknown")

        return render_template("index.html", prediction_text=f"Predicted Credit Score: {predicted_label}")

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
print("Sample")
