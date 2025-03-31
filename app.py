from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model and scaler
with open("placement_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

@app.route("/")
def home():
    return render_template("index.html", prediction_text="")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input values (✅ Now includes all 8 features)
        features = [
            float(request.form["CGPA"]),
            int(request.form["Internships"]),
            int(request.form["Projects"]),
            int(request.form["Workshops"]),
            float(request.form["AptitudeTestScore"]),
            float(request.form["SoftSkillsRating"]),
            int(request.form["ExtracurricularActivities"]),  # ✅ Included
            int(request.form["PlacementTraining"])  # ✅ Included
        ]

        # Scale input data
        scaled_features = scaler.transform([features])
        
        # Make prediction
        prediction = model.predict(scaled_features)[0]
        result_text = " ✅🎉 Congratulations! You are Placed! 🎓" if prediction == 1 else "❌😔 Unfortunately, You are Not Placed. Keep Improving! 🚀"

        return render_template("index.html", prediction_text=f"Prediction: {result_text}")

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
