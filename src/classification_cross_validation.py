import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier


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


df["support_category"] = df["G3"].apply(
    create_support_category
)


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
# 6. Models
# --------------------------------------------------

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=5
    )
}


# --------------------------------------------------
# 7. Stratified 5-fold Cross Validation
# --------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# --------------------------------------------------
# 8. Evaluate models
# --------------------------------------------------

results = []


for name, model in models.items():

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=[
            "accuracy",
            "precision_weighted",
            "recall_weighted",
            "f1_weighted"
        ]
    )

    results.append({
        "Model": name,
        "Accuracy Mean": scores["test_accuracy"].mean(),
        "Accuracy Std": scores["test_accuracy"].std(),
        "Precision Mean": scores["test_precision_weighted"].mean(),
        "Recall Mean": scores["test_recall_weighted"].mean(),
        "F1 Mean": scores["test_f1_weighted"].mean()
    })


# --------------------------------------------------
# 9. Display results
# --------------------------------------------------

results_df = pd.DataFrame(results)

print("\n" + "=" * 80)
print("5-FOLD STRATIFIED CROSS-VALIDATION RESULTS")
print("=" * 80)

print(
    results_df.sort_values(
        by="F1 Mean",
        ascending=False
    ).to_string(index=False)
)


# --------------------------------------------------
# 10. Save results
# --------------------------------------------------

results_df.to_csv(
    "reports/classification_cross_validation.csv",
    index=False
)

print("\nResults saved to:")
print("reports/classification_cross_validation.csv")