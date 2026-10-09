
from flask import Flask, render_template, request
import pandas as pd
import joblib
import json
import os

app = Flask(__name__)

# Load the trained ML model
MODEL_PATH = "house_price_model.pkl"
RESULTS_PATH = "model_results.json"

model = None
model_name = "Machine Learning Model"

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)

if os.path.exists(RESULTS_PATH):
    with open(RESULTS_PATH, "r") as file:
        results_data = json.load(file)
        model_name = results_data["best_model"]


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    prediction_lakhs = None
    error = None

    if request.method == "POST":
        try:
            # Read values entered in the form
            area = int(request.form["area"])
            bedrooms = int(request.form["bedrooms"])
            bathrooms = int(request.form["bathrooms"])
            floors = int(request.form["floors"])
            parking = int(request.form["parking"])
            location = request.form["location"]
            house_age = int(request.form["house_age"])

            # Validate the inputs
            if not 500 <= area <= 3500:
                raise ValueError("Area must be between 500 and 3500 sq. ft.")

            if not 1 <= bedrooms <= 6:
                raise ValueError("Bedrooms must be between 1 and 6.")

            if not 1 <= bathrooms <= 4:
                raise ValueError("Bathrooms must be between 1 and 4.")

            if not 1 <= floors <= 3:
                raise ValueError("Floors must be between 1 and 3.")

            if not 0 <= parking <= 3:
                raise ValueError("Parking spaces must be between 0 and 3.")

            if not 0 <= house_age <= 40:
                raise ValueError("House age must be between 0 and 40 years.")

            allowed_locations = [
                "Kozhikode",
                "Kochi",
                "Thiruvananthapuram",
                "Thrissur",
                "Kannur"
            ]

            if location not in allowed_locations:
                raise ValueError("Please select a valid location.")

            if model is None:
                raise RuntimeError(
                    "Model not found. Run train_model.py first."
                )

            # Create input in the same format used for training
            input_data = pd.DataFrame([{
                "Area": area,
                "Bedrooms": bedrooms,
                "Bathrooms": bathrooms,
                "Floors": floors,
                "Parking": parking,
                "Location": location,
                "House_Age": house_age
            }])

            # Predict the house price
            prediction = float(model.predict(input_data)[0])
            prediction_lakhs = prediction / 100000

        except (ValueError, KeyError) as e:
            error = str(e)

        except Exception:
            error = "Prediction failed. Please check the model and try again."

    return render_template(
        "index.html",
        prediction=prediction,
        prediction_lakhs=prediction_lakhs,
        error=error,
        model_name=model_name
    )


if __name__ == "__main__":
    app.run(debug=True)
