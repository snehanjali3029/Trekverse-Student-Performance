import pandas as pd
import joblib

# Load the dataset
df = pd.read_csv("data/student_mat_cleaned.csv")

# Separate features and target
X = df.drop(columns=["G3", "G1", "G2"])
y = df["G3"]

# Load the saved preprocessing pipeline and model
preprocessor = joblib.load("models/preprocessor.pkl")
model = joblib.load("models/random_forest_model.pkl")


# --------------------------------------------------
# Test 1: Normal input
# --------------------------------------------------

normal_student = X.iloc[[0]]

processed_normal = preprocessor.transform(normal_student)
normal_prediction = model.predict(processed_normal)[0]

print("Test 1 - Normal Input")
print("Predicted Score:", round(normal_prediction, 2))


# --------------------------------------------------
# Test 2: Missing value
# --------------------------------------------------

missing_student = normal_student.copy()

# Make one input value missing
missing_student.loc[missing_student.index[0], "age"] = None

try:
    processed_missing = preprocessor.transform(missing_student)
    missing_prediction = model.predict(processed_missing)[0]

    print("\nTest 2 - Missing Input")
    print("Predicted Score:", round(missing_prediction, 2))

except Exception as e:
    print("\nTest 2 - Missing Input")
    print("Model safely rejected the missing value.")
    print("Reason:", type(e).__name__)


# --------------------------------------------------
# Test 3: Perturbed input
# --------------------------------------------------

perturbed_student = normal_student.copy()

# Change the student's study time
original_studytime = perturbed_student.loc[
    perturbed_student.index[0], "studytime"
]

if original_studytime < 4:
    perturbed_student.loc[
        perturbed_student.index[0], "studytime"
    ] = original_studytime + 1
else:
    perturbed_student.loc[
        perturbed_student.index[0], "studytime"
    ] = original_studytime - 1

processed_perturbed = preprocessor.transform(perturbed_student)
perturbed_prediction = model.predict(processed_perturbed)[0]

print("\nTest 3 - Perturbed Input")
print("Original Study Time:", original_studytime)
print(
    "Changed Study Time:",
    perturbed_student.loc[perturbed_student.index[0], "studytime"]
)
print("Predicted Score:", round(perturbed_prediction, 2))


# --------------------------------------------------
# Summary
# --------------------------------------------------

print("\nRobustness testing completed.")