import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

df = pd.read_csv("data/student_mat_cleaned.csv")


# --------------------------------------------------
# 2. Create support category
# --------------------------------------------------

def create_support_category(g3):
    if g3 < 8:
        return "High Support"
    elif g3 < 12:
        return "Medium Support"
    else:
        return "Low Support"


df["support_category"] = df["G3"].apply(create_support_category)


# --------------------------------------------------
# 3. Features and target
# --------------------------------------------------

X = df.drop(
    columns=["G3", "G1", "G2", "support_category"]
)

y = df["support_category"]


# --------------------------------------------------
# 4. Feature groups
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
# 5. Preprocessing
# --------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
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
# 6. Train-test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# 7. Balanced Random Forest
# --------------------------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42
)


pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# --------------------------------------------------
# 8. Train
# --------------------------------------------------

pipeline.fit(X_train, y_train)


# --------------------------------------------------
# 9. Predictions
# --------------------------------------------------

y_pred = pipeline.predict(X_test)


# --------------------------------------------------
# 10. Overall metrics
# --------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)


print("\n" + "=" * 60)
print("BALANCED RANDOM FOREST RESULTS")
print("=" * 60)

print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))


# --------------------------------------------------
# 11. Detailed classification report
# --------------------------------------------------

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# --------------------------------------------------
# 12. Confusion Matrix
# --------------------------------------------------

labels = [
    "High Support",
    "Medium Support",
    "Low Support"
]

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)

print("Confusion Matrix:")
print(cm)


# --------------------------------------------------
# 13. High Support recall
# --------------------------------------------------

report = classification_report(
    y_test,
    y_pred,
    output_dict=True,
    zero_division=0
)

high_support_recall = report[
    "High Support"
]["recall"]

print(
    "\nHigh Support Recall:",
    round(high_support_recall, 4)
)