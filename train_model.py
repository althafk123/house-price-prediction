
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

# 1. Load the dataset
df = pd.read_csv("house_prices.csv")

# 2. Remove missing values and duplicate rows
df = df.dropna().drop_duplicates()

# 3. Select input features and target
features = [
    "Area",
    "Bedrooms",
    "Bathrooms",
    "Floors",
    "Parking",
    "Location",
    "House_Age"
]

X = df[features]
y = df["Price"]

# 4. Identify numerical and categorical columns
numeric_features = [
    "Area",
    "Bedrooms",
    "Bathrooms",
    "Floors",
    "Parking",
    "House_Age"
]

categorical_features = ["Location"]

# 5. Convert location text into numerical features
preprocessor = ColumnTransformer(
    transformers=[
        (
            "location",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)

# 6. Split data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# 7. Define the models
models = {
    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=150,
        random_state=42,
        min_samples_leaf=2,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
}

results = []
best_rmse = float("inf")
best_model = None
best_model_name = None

# 8. Train and evaluate every model
for name, regressor in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", regressor)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(
        y_test, predictions
    ) ** 0.5
    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "MAE": round(float(mae), 2),
        "RMSE": round(float(rmse), 2),
        "R2_Score": round(float(r2), 4)
    })

    print(f"\n{name}")
    print(f"MAE: {mae:,.2f} rupees")
    print(f"RMSE: {rmse:,.2f} rupees")
    print(f"R2 Score: {r2:.4f}")

    # Choose the model with the lowest test RMSE
    if rmse < best_rmse:
        best_rmse = rmse
        best_model = pipeline
        best_model_name = name

# 9. Save the selected model
joblib.dump(best_model, "house_price_model.pkl")

# 10. Save model evaluation results
with open("model_results.json", "w") as file:
    json.dump({
        "best_model": best_model_name,
        "results": results
    }, file, indent=4)

print("\n------------------------------")
print("TRAINING COMPLETED")
print("Best model:", best_model_name)
print("Model saved as house_price_model.pkl")
print("Results saved as model_results.json")
print("------------------------------")
