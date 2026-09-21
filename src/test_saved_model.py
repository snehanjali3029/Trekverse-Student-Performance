import pandas as pd
import joblib


# Load saved model and preprocessor
model = joblib.load("models/random_forest_model.pkl")
preprocessor = joblib.load("models/preprocessor.pkl")

print("Model loaded successfully.")
print("Preprocessor loaded successfully.")


# Load original dataset
df = pd.read_csv("data/student_mat_cleaned.csv")


# Select one student
sample_student = df.drop(columns=["G3", "G1", "G2"]).iloc[[0]]


# Preprocess the student data
sample_processed = preprocessor.transform(sample_student)


# Make prediction
prediction = model.predict(sample_processed)


print("\n----- SAMPLE PREDICTION -----")
print("Predicted Final Grade:", round(prediction[0], 2))