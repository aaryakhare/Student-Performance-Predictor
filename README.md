# 🎓 Student Performance Predictor (End-to-End ML Web App)

## 📌 Project Overview

The Student Performance Predictor is an end-to-end Machine Learning web application that predicts a student's exam score based on the number of hours they study.

It demonstrates the complete ML lifecycle:
- Data exploration
- Model training
- Evaluation
- Saving model
- Real-time prediction using a web interface (frontend + backend integration)

## 🚀 Features

- Load and process student dataset (CSV)
- Exploratory Data Analysis (EDA)
- Data visualization using Matplotlib
- Train Linear Regression model
- Evaluate model using R² Score
- Save trained model using Joblib
- REST API using Flask
- Real-time prediction via frontend UI
- Interactive animated web interface

## 📂 Project Structure
│
├── backend/
│ ├── app.py # Flask API
│ ├── train.py # Model training script
│ ├── predict.py # CLI prediction script
│ │
│ ├── data/
│ │ └── student_scores.csv
│ │
│ └── models/
│ └── student_model.pkl
│
├── frontend/
│ ├── index.html
│ ├── style.css
│ └── script.js
│
├── requirements.txt
└── README.md


---

## 🛠 Technologies Used

### 🧠 Machine Learning
- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib

### 🎨 Frontend
- HTML5
- CSS3 (Glassmorphism + Animations)
- JavaScript (Fetch API)

### ⚙ Backend
- Flask
- Flask-CORS

---

## 📊 Machine Learning Algorithm

- Linear Regression

---

## 📈 Model Performance

- **R² Score:** ~0.99 (High Accuracy Model)

---

▶️ How to Run the Project
1️⃣ Install dependencies
    pip install -r requirements.txt
2️⃣ Train the model
    python backend/train.py
3️⃣ Run Flask backend
    python backend/app.py
Backend runs at:
    http://127.0.0.1:5000
4️⃣ Run frontend
    Simply open:
    frontend/index.html