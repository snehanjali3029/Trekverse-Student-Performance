import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

df = pd.read_csv("data/student_mat_cleaned.csv")

print("Dataset loaded successfully.")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])


# --------------------------------------------------
# 2. Separate features and target
# --------------------------------------------------

y = df["G3"]

# Remove target and G1/G2 to avoid using previous
# period grades as direct predictors of final grade.
X = df.drop(columns=["G3", "G1", "G2"])


# --------------------------------------------------
# 3. Identify feature types
# --------------------------------------------------

numerical_features = [
    "age",
    "Medu",
    "Fedu",
    "traveltime",
    "studytime",
    "failures",
    "famrel",
    "freetime",
    "goout",
    "Dalc",
    "Walc",
    "health",
    "absences"
]

categorical_features = [
    "school",
    "sex",
    "address",
    "famsize",
    "Pstatus",
    "Mjob",
    "Fjob",
    "reason",
    "guardian",
    "schoolsup",
    "famsup",
    "paid",
    "activities",
    "nursery",
    "higher",
    "internet",
    "romantic"
]


# --------------------------------------------------
# 4. Create preprocessing pipeline
# --------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            "passthrough",
            numerical_features
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)


# --------------------------------------------------
# 5. Train-test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# --------------------------------------------------
# 6. Fit preprocessing on training data
# --------------------------------------------------

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)


# --------------------------------------------------
# 7. Create tuned Random Forest model
# --------------------------------------------------

model = RandomForestRegressor(
    n_estimators=50,
    max_depth=10,
    min_samples_split=2,
    random_state=42
)


# --------------------------------------------------
# 8. Train model
# --------------------------------------------------

model.fit(
    X_train_processed,
    y_train
)

print("Random Forest model trained successfully.")


# --------------------------------------------------
# 9. Create models folder
# --------------------------------------------------

os.makedirs("models", exist_ok=True)


# --------------------------------------------------
# 10. Save model
# --------------------------------------------------

joblib.dump(
    model,
    "models/random_forest_model.pkl"
)


# --------------------------------------------------
# 11. Save preprocessor
# --------------------------------------------------

joblib.dump(
    preprocessor,
    "models/preprocessor.pkl"
)


print("\n----- MODEL FILES SAVED -----")

print(
    "Model:",
    "models/random_forest_model.pkl"
)

print(
    "Preprocessor:",
    "models/preprocessor.pkl"
)