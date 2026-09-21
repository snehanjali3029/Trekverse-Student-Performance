import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Load dataset
df = pd.read_csv("data/student_mat_cleaned.csv")

print("----- DATASET -----")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# Separate target and features
y = df["G3"]

# Remove G3, G1 and G2
X = df.drop(columns=["G3", "G1", "G2"])

# Identify numerical and categorical columns
numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

print("\n----- FEATURES -----")
print("Numerical features:", len(numerical_features))
print("Categorical features:", len(categorical_features))

# Create preprocessing pipeline
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

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Fit preprocessing only on training data
X_train_processed = preprocessor.fit_transform(X_train)

# Transform test data
X_test_processed = preprocessor.transform(X_test)

print("\n----- PROCESSED DATA -----")
print("Training shape:", X_train_processed.shape)
print("Testing shape:", X_test_processed.shape)

from sklearn.ensemble import RandomForestRegressor

# ----- TRAIN TUNED RANDOM FOREST -----

print("\n----- TUNED RANDOM FOREST -----")

model = RandomForestRegressor(
    n_estimators=50,
    max_depth=10,
    min_samples_split=2,
    random_state=42
)

model.fit(X_train_processed, y_train)

print("Tuned Random Forest trained successfully.")

# ----- FEATURE IMPORTANCE -----

print("\n----- FEATURE IMPORTANCE -----")

# Get feature names after one-hot encoding
feature_names = preprocessor.get_feature_names_out()

# Get importance values from Random Forest
importance_values = model.feature_importances_

# Create a DataFrame
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance_values
})

# Sort from highest to lowest
importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 15 Most Important Features:")
print(importance_df.head(15))

import matplotlib.pyplot as plt

# ----- FEATURE IMPORTANCE VISUALIZATION -----

top_features = importance_df.head(15).sort_values(
    by="Importance"
)

plt.figure(figsize=(10, 7))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 15 Random Forest Feature Importances")

plt.tight_layout()

plt.savefig(
    "reports/random_forest_feature_importance.png",
    dpi=300
)

plt.close()

print(
    "\nFeature importance graph saved to "
    "reports/random_forest_feature_importance.png"
)
import shap

# ----- SHAP EXPLAINABILITY -----

print("\n----- SHAP EXPLANATION -----")

# Create SHAP explainer for the Random Forest
explainer = shap.TreeExplainer(model)

# Calculate SHAP values for the test data
shap_values = explainer.shap_values(X_test_processed)

print("SHAP values calculated successfully.")
print("SHAP values shape:", shap_values.shape)

# ----- SHAP SUMMARY PLOT -----

print("\n----- SHAP SUMMARY PLOT -----")

plt.figure(figsize=(10, 7))

shap.summary_plot(
    shap_values,
    X_test_processed,
    feature_names=feature_names,
    show=False
)

plt.tight_layout()

plt.savefig(
    "reports/shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("SHAP summary plot saved to reports/shap_summary.png")

# ----- INDIVIDUAL SHAP EXPLANATIONS -----

print("\n----- INDIVIDUAL STUDENT EXPLANATIONS -----")

# Select first 3 students from the test set
selected_students = [0, 1, 2]

for student_index in selected_students:

    actual_grade = y_test.iloc[student_index]
    predicted_grade = model.predict(
        X_test_processed[student_index:student_index + 1]
    )[0]

    print("\nStudent", student_index + 1)
    print("Actual G3:", actual_grade)
    print("Predicted G3:", round(predicted_grade, 2))

    # Get SHAP values for this student
    student_shap = shap_values[student_index]

    # Get the 5 features with largest absolute SHAP values
    top_indices = (
        abs(student_shap)
        .argsort()[::-1][:5]
    )

    print("Top contributing features:")

    for index in top_indices:
        print(
            feature_names[index],
            "SHAP value:",
            round(student_shap[index], 4)
        )

        # ----- ERROR ANALYSIS -----

print("\n----- ERROR ANALYSIS -----")

# Make predictions for all test students
test_predictions = model.predict(X_test_processed)

# Create error DataFrame
error_df = pd.DataFrame({
    "Actual_G3": y_test.values,
    "Predicted_G3": test_predictions
})

# Calculate prediction error
error_df["Error"] = (
    error_df["Predicted_G3"] - error_df["Actual_G3"]
)

# Absolute error
error_df["Absolute_Error"] = (
    abs(error_df["Error"])
)

print("\nError Summary:")
print(error_df["Absolute_Error"].describe())

print("\nTop 10 Largest Prediction Errors:")
print(
    error_df.sort_values(
        by="Absolute_Error",
        ascending=False
    ).head(10)
)
# ----- ACTUAL VS PREDICTED VISUALIZATION -----

print("\n----- ACTUAL VS PREDICTED GRAPH -----")

plt.figure(figsize=(8, 6))

plt.scatter(
    error_df["Actual_G3"],
    error_df["Predicted_G3"],
    alpha=0.7
)

# Perfect prediction reference line
plt.plot(
    [0, 20],
    [0, 20],
    linestyle="--"
)

plt.xlabel("Actual Final Grade (G3)")
plt.ylabel("Predicted Final Grade (G3)")
plt.title("Actual vs Predicted Final Grades")

plt.tight_layout()

plt.savefig(
    "reports/actual_vs_predicted.png",
    dpi=300
)

plt.close()

print(
    "Actual vs predicted graph saved to "
    "reports/actual_vs_predicted.png"
)