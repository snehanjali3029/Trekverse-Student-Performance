import pandas as pd
import joblib


def test_saved_model_prediction():
    # Load dataset
    df = pd.read_csv("data/student_mat_cleaned.csv")

    # Remove target and G1/G2 to match training
    X = df.drop(columns=["G3", "G1", "G2"])

    # Take one student record
    student = X.iloc[[0]]

    # Load saved preprocessing pipeline and model
    preprocessor = joblib.load("models/preprocessor.pkl")
    model = joblib.load("models/random_forest_model.pkl")

    # Transform the student data
    processed_student = preprocessor.transform(student)

    # Make prediction
    prediction = model.predict(processed_student)[0]

    # Check that prediction is a number
    assert isinstance(prediction, float)

    # Check that prediction is within valid grade range
    assert 0 <= prediction <= 20