import joblib
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
model_path = BASE_DIR / "models" / "student_model.pkl"

model = joblib.load(model_path)

hours = float(input("Enter study hours: "))

input_df = pd.DataFrame({"Hours": [hours]})
prediction = model.predict(input_df)[0]

print(f"Predicted Score: {prediction:.2f}")