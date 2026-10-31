
from flask import Flask, render_template, request
from flask_cors import CORS
from pathlib import Path

import pandas as pd
import numpy as np
import pickle

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error


# ==========================================
# 1. INITIALIZE FLASK APPLICATION
# ==========================================

app = Flask(__name__)
CORS(app)

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "Cleaned_Car_data.csv"
MODEL_PATH = BASE_DIR / "LinearRegressionModel.pkl"


# ==========================================
# 2. LOAD DATASET
# ==========================================

print("\nLoading Car Dataset...")

car = pd.read_csv(DATA_PATH)

required_columns = [
    "name",
    "company",
    "year",
    "Price",
    "kms_driven",
    "fuel_type"
]

missing_columns = [
    col for col in required_columns if col not in car.columns
]

if missing_columns:
    raise ValueError(f"Missing dataset columns: {missing_columns}")

car = car[required_columns].copy()

# Convert numeric columns
for column in ["year", "Price", "kms_driven"]:
    car[column] = pd.to_numeric(car[column], errors="coerce")

# Remove missing values
car.dropna(inplace=True)

# Remove invalid prices and kilometres
car = car[
    (car["Price"] > 0) &
    (car["kms_driven"] >= 0)
].copy()

# Clean categorical values
for column in ["name", "company", "fuel_type"]:
    car[column] = car[column].astype(str).str.strip()

car = car[
    (car["name"] != "") &
    (car["company"] != "") &
    (car["fuel_type"] != "")
].copy()

if len(car) < 5:
    raise ValueError("Not enough valid rows to train the model.")

print("Dataset loaded successfully!")
print("Total records:", len(car))


# ==========================================
# 3. PREPARE MACHINE LEARNING DATA
# ==========================================

X = car[
    ["name", "company", "year", "kms_driven", "fuel_type"]
]

y = car["Price"]

categorical_features = [
    "name",
    "company",
    "fuel_type"
]

numerical_features = [
    "year",
    "kms_driven"
]


# ==========================================
# 4. CREATE ML PIPELINE
# ==========================================

preprocessor = ColumnTransformer(

    transformers=[

        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),

        (
            "numerical",
            "passthrough",
            numerical_features
        )

    ]
)

model = Pipeline(

    steps=[

        ("preprocessor", preprocessor),

        ("regressor", LinearRegression())

    ]

)


# ==========================================
# 5. TRAIN AND EVALUATE MODEL
# ==========================================

print("\nTraining Machine Learning Model...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

model.fit(X_train, y_train)

test_predictions = model.predict(X_test)

r2 = r2_score(y_test, test_predictions)

mae = mean_absolute_error(y_test, test_predictions)

print("\nMODEL EVALUATION")
print("--------------------------")
print("R2 Score:", round(r2, 4))
print("Mean Absolute Error:", round(mae, 2))


# ==========================================
# 6. TRAIN FINAL MODEL ON FULL DATASET
# ==========================================

print("\nTraining final model on complete dataset...")

model.fit(X, y)


# ==========================================
# 7. SAVE TRAINED MODEL
# ==========================================

with open(MODEL_PATH, "wb") as file:

    pickle.dump(model, file)

print("\nModel trained and saved successfully!")
print("Model location:", MODEL_PATH)


# ==========================================
# 8. HOME PAGE
# ==========================================

@app.route("/", methods=["GET"])
def index():

    companies = sorted(car["company"].unique())

    car_models = sorted(car["name"].unique())

    years = sorted(
        car["year"].unique(),
        reverse=True
    )

    fuel_types = sorted(car["fuel_type"].unique())

    companies.insert(0, "Select Company")

    return render_template(
        "index.html",
        companies=companies,
        car_models=car_models,
        years=years,
        fuel_types=fuel_types,
        car=car
    )


# ==========================================
# 9. CAR PRICE PREDICTION
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        company = request.form.get("company")
        car_model = request.form.get("car_models")
        year = request.form.get("year")
        fuel_type = request.form.get("fuel_type")
        driven = request.form.get("kilo_driven")

        # Validate input
        if not all([
            company,
            car_model,
            year,
            fuel_type,
            driven
        ]):
            return "Please fill all fields.", 400

        if company == "Select Company":
            return "Please select a valid company.", 400

        year = int(year)
        driven = int(driven)

        if driven < 0 or driven > 1000000:
            return "Invalid kilometres driven.", 400

        # Validate car selection
        valid_car = car[
            (car["company"] == company) &
            (car["name"] == car_model)
        ]

        if valid_car.empty:
            return "Invalid car model.", 400

        if fuel_type not in car["fuel_type"].unique():
            return "Invalid fuel type.", 400

        # Prepare prediction input
        input_data = pd.DataFrame({

            "name": [car_model],

            "company": [company],

            "year": [year],

            "kms_driven": [driven],

            "fuel_type": [fuel_type]

        })

        # Predict
        prediction = model.predict(input_data)

        predicted_price = float(prediction[0])

        if not np.isfinite(predicted_price):
            return "Invalid prediction generated.", 500

        predicted_price = max(0, predicted_price)

        predicted_price = round(predicted_price, 2)

        print("\nPREDICTION RESULT")
        print("--------------------------")
        print("Company:", company)
        print("Model:", car_model)
        print("Year:", year)
        print("Fuel:", fuel_type)
        print("Kilometres:", driven)
        print("Predicted Price: ₹", predicted_price)

        return str(predicted_price)

    except ValueError:
        return "Please enter valid numeric values.", 400

    except Exception as e:

        app.logger.exception("Prediction Error")

        return f"Prediction failed: {str(e)}", 500


# ==========================================
# 10. RUN APPLICATION
# ==========================================

if __name__ == "__main__":

    print("\nStarting Car Price Predictor...")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
        use_reloader=False
    )
