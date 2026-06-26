from flask import Flask, request, jsonify
import joblib
import pandas as pd
from pathlib import Path
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

BASE_DIR = Path(__file__).resolve().parent
model_path = BASE_DIR / "models" / "student_model.pkl"

model = joblib.load(model_path)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    hours = float(data["hours"])

    input_df = pd.DataFrame({"Hours": [hours]})
    prediction = model.predict(input_df)[0]

    return jsonify({
        "predicted_score": round(prediction, 2)
    })

if __name__ == "__main__":
    app.run(debug=True)