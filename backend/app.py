from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
from pathlib import Path
from flask_cors import CORS

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent
model_path = BACKEND_DIR / "models" / "student_model.pkl"
app = Flask(
    __name__,
    template_folder=str(PROJECT_DIR / "frontend" / "templates"),
    static_folder=str(PROJECT_DIR / "frontend" / "static")
)
CORS(app)

model = joblib.load(model_path)

@app.route("/")
def home():
    return render_template("index.html")

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