import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
import joblib
from sklearn.metrics import r2_score

BASE_DIR = Path(__file__).resolve().parent.parent

data_path = BASE_DIR / "data" / "student_scores.csv"
model_path = BASE_DIR / "models" / "student_model.pkl"

df = pd.read_csv(data_path)

X = df[["Hours"]]
y = df["Scores"]


model = LinearRegression()
model.fit(X, y)

predictions = model.predict(X)
accuracy = r2_score(y, predictions)

print(f"Model Accuracy (R² Score): {accuracy:.4f}")

joblib.dump(model, model_path)

print("✅ Model trained successfully!")
print(f"Model saved at: {model_path}")