import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression

BASE_DIR = Path(__file__).resolve().parent.parent
csv_path = BASE_DIR / "data" / "student_scores.csv"

df = pd.read_csv(csv_path)

print("===== Complete Dataset =====")
print(df)

print("\n===== First 5 Rows =====")
print(df.head())

print("\n===== Dataset Information =====")
df.info()

print("\n===== Statistical Summary =====")
print(df.describe())

print("\n===== Missing Values =====")
print(df.isnull().sum())

X = df[["Hours"]]
y = df["Scores"]

model = LinearRegression()
model.fit(X, y)

predicted_scores = model.predict(X)

plt.scatter(df["Hours"], df["Scores"], label="Actual Data")

plt.plot(df["Hours"], predicted_scores, label="Regression Line")

plt.title("Study Hours vs Exam Scores")
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")

plt.legend()

plt.show()