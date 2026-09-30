import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    classification_report
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

X = df.drop(columns=["G3", "G1", "G2", "support_category"])

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
# 7. Random Forest classification model
# --------------------------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# --------------------------------------------------
# 8. Train model
# --------------------------------------------------

pipeline.fit(X_train, y_train)


# --------------------------------------------------
# 9. Predictions
# --------------------------------------------------

y_pred = pipeline.predict(X_test)

y_probability = pipeline.predict_proba(X_test)


# --------------------------------------------------
# 10. Classification report
# --------------------------------------------------

print("\n" + "=" * 60)
print("RANDOM FOREST CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# --------------------------------------------------
# 11. Confusion Matrix
# --------------------------------------------------

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[
        "High Support",
        "Medium Support",
        "Low Support"
    ]
)

print("\nConfusion Matrix:")
print(cm)


disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "High Support",
        "Medium Support",
        "Low Support"
    ]
)

disp.plot()

plt.title("Random Forest Confusion Matrix")
plt.tight_layout()

plt.savefig(
    "reports/classification_confusion_matrix.png",
    dpi=300
)

plt.show()


# --------------------------------------------------
# 12. ROC-AUC
# --------------------------------------------------

classes = [
    "High Support",
    "Medium Support",
    "Low Support"
]

y_test_binary = label_binarize(
    y_test,
    classes=classes
)

roc_auc = roc_auc_score(
    y_test_binary,
    y_probability,
    multi_class="ovr",
    average="weighted"
)

print("\nWeighted ROC-AUC:", round(roc_auc, 4))


# --------------------------------------------------
# 13. Save evaluation result
# --------------------------------------------------

evaluation = pd.DataFrame({
    "Metric": [
        "Weighted ROC-AUC"
    ],
    "Value": [
        roc_auc
    ]
})

evaluation.to_csv(
    "reports/classification_evaluation.csv",
    index=False
)

print("\nEvaluation results saved successfully.")
print("Confusion matrix saved to:")
print("reports/classification_confusion_matrix.png")