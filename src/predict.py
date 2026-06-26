import joblib
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
model_path = BASE_DIR / "models" / "student_model.pkl"

model = joblib.load(model_path)

hours = float(input("Enter study hours: "))

input_data = pd.DataFrame({"Hours": [hours]})

predicted_score = model.predict(input_data)

print(f"\nPredicted Exam Score: {predicted_score[0]:.2f}")