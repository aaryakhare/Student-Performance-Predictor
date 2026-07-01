import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib
from sklearn.metrics import r2_score

BASE_DIR = Path(__file__).resolve().parent

data_path = BASE_DIR / "data" / "student_scores.csv"
model_path = BASE_DIR / "models" / "student_model.pkl"

df = pd.read_csv(data_path)

X = df[["Hours"]]
y = df["Scores"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = r2_score(y_test, predictions)

print(f"Model Accuracy (R² Score): {accuracy:.4f}")

print(f"Slope (Coefficient): {model.coef_[0]:.4f}")
print(f"Intercept: {model.intercept_:.4f}")

joblib.dump(model, model_path)

print("✅ Model trained successfully!")
print(f"Model saved at: {model_path}")