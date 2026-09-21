import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

df = pd.read_csv("data/student_mat_cleaned.csv")

print("----- DATASET -----")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])


# --------------------------------------------------
# 2. Separate features (X) and target (y)
# --------------------------------------------------

# G3 is the final grade that we want to predict.
y = df["G3"]

# Remove G3, G1 and G2 from input features.
# G1 and G2 are previous period grades.
X = df.drop(columns=["G3", "G1", "G2"])


# --------------------------------------------------
# 3. Identify numerical and categorical features
# --------------------------------------------------

numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()


print("\n----- FEATURE TYPES -----")
print("Numerical features:", len(numerical_features))
print("Categorical features:", len(categorical_features))


# --------------------------------------------------
# 4. Create preprocessing pipeline
# --------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)


# --------------------------------------------------
# 5. Split into training and testing data
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\n----- TRAIN / TEST -----")
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))


# --------------------------------------------------
# 6. Preprocess the data
# --------------------------------------------------

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("\n----- PROCESSED DATA -----")
print("Training shape:", X_train_processed.shape)
print("Testing shape:", X_test_processed.shape)


# --------------------------------------------------
# 7. Create baseline model
# --------------------------------------------------

baseline_model = DummyRegressor(strategy="mean")

baseline_model.fit(
    X_train_processed,
    y_train
)


# --------------------------------------------------
# 8. Make baseline predictions
# --------------------------------------------------

y_pred = baseline_model.predict(X_test_processed)


# --------------------------------------------------
# 9. Evaluate baseline model
# --------------------------------------------------

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)


print("\n----- BASELINE MODEL RESULTS -----")
print("Model: Dummy Regressor")
print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)


# --------------------------------------------------
# 10. Train Linear Regression model
# --------------------------------------------------

linear_model = LinearRegression()

linear_model.fit(
    X_train_processed,
    y_train
)


# --------------------------------------------------
# 11. Make Linear Regression predictions
# --------------------------------------------------

linear_pred = linear_model.predict(X_test_processed)


# --------------------------------------------------
# 12. Evaluate Linear Regression
# --------------------------------------------------

linear_mae = mean_absolute_error(
    y_test,
    linear_pred
)

linear_rmse = mean_squared_error(
    y_test,
    linear_pred
) ** 0.5

linear_r2 = r2_score(
    y_test,
    linear_pred
)


print("\n----- LINEAR REGRESSION RESULTS -----")
print("Model: Linear Regression")
print("MAE:", linear_mae)
print("RMSE:", linear_rmse)
print("R2 Score:", linear_r2)


# --------------------------------------------------
# 13. Train Decision Tree Regression model
# --------------------------------------------------

tree_model = DecisionTreeRegressor(
    random_state=42
)

tree_model.fit(
    X_train_processed,
    y_train
)


# --------------------------------------------------
# 14. Make Decision Tree predictions
# --------------------------------------------------

tree_pred = tree_model.predict(X_test_processed)


# --------------------------------------------------
# 15. Evaluate Decision Tree
# --------------------------------------------------

tree_mae = mean_absolute_error(
    y_test,
    tree_pred
)

tree_rmse = mean_squared_error(
    y_test,
    tree_pred
) ** 0.5

tree_r2 = r2_score(
    y_test,
    tree_pred
)


print("\n----- DECISION TREE RESULTS -----")
print("Model: Decision Tree Regression")
print("MAE:", tree_mae)
print("RMSE:", tree_rmse)
print("R2 Score:", tree_r2)


# --------------------------------------------------
# 16. Train Random Forest Regression model
# --------------------------------------------------

random_forest_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

random_forest_model.fit(
    X_train_processed,
    y_train
)


# --------------------------------------------------
# 17. Make Random Forest predictions
# --------------------------------------------------

random_forest_pred = random_forest_model.predict(
    X_test_processed
)


# --------------------------------------------------
# 18. Evaluate Random Forest
# --------------------------------------------------

rf_mae = mean_absolute_error(
    y_test,
    random_forest_pred
)

rf_rmse = mean_squared_error(
    y_test,
    random_forest_pred
) ** 0.5

rf_r2 = r2_score(
    y_test,
    random_forest_pred
)


print("\n----- RANDOM FOREST RESULTS -----")
print("Model: Random Forest Regression")
print("MAE:", rf_mae)
print("RMSE:", rf_rmse)
print("R2 Score:", rf_r2)


# --------------------------------------------------
# 19. Train Gradient Boosting Regression model
# --------------------------------------------------

gradient_model = GradientBoostingRegressor(
    random_state=42
)

gradient_model.fit(
    X_train_processed,
    y_train
)


# --------------------------------------------------
# 20. Make Gradient Boosting predictions
# --------------------------------------------------

gradient_pred = gradient_model.predict(
    X_test_processed
)


# --------------------------------------------------
# 21. Evaluate Gradient Boosting
# --------------------------------------------------

gradient_mae = mean_absolute_error(
    y_test,
    gradient_pred
)

gradient_rmse = mean_squared_error(
    y_test,
    gradient_pred
) ** 0.5

gradient_r2 = r2_score(
    y_test,
    gradient_pred
)


print("\n----- GRADIENT BOOSTING RESULTS -----")
print("Model: Gradient Boosting Regression")
print("MAE:", gradient_mae)
print("RMSE:", gradient_rmse)
print("R2 Score:", gradient_r2)


# --------------------------------------------------
# 22. Train Extra Trees Regression model
# --------------------------------------------------

extra_trees_model = ExtraTreesRegressor(
    n_estimators=100,
    random_state=42
)

extra_trees_model.fit(
    X_train_processed,
    y_train
)


# --------------------------------------------------
# 23. Make Extra Trees predictions
# --------------------------------------------------

extra_trees_pred = extra_trees_model.predict(
    X_test_processed
)


# --------------------------------------------------
# 24. Evaluate Extra Trees
# --------------------------------------------------

extra_trees_mae = mean_absolute_error(
    y_test,
    extra_trees_pred
)

extra_trees_rmse = mean_squared_error(
    y_test,
    extra_trees_pred
) ** 0.5

extra_trees_r2 = r2_score(
    y_test,
    extra_trees_pred
)


print("\n----- EXTRA TREES RESULTS -----")
print("Model: Extra Trees Regression")
print("MAE:", extra_trees_mae)
print("RMSE:", extra_trees_rmse)
print("R2 Score:", extra_trees_r2)


# --------------------------------------------------
# 25. Cross-validation
# --------------------------------------------------

print("\n----- CROSS-VALIDATION -----")

models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        random_state=42
    ),

    "Extra Trees": ExtraTreesRegressor(
        n_estimators=100,
        random_state=42
    )
}


for name, model in models.items():

    scores = cross_val_score(
        model,
        X_train_processed,
        y_train,
        cv=5,
        scoring="neg_mean_absolute_error"
    )

    mae_scores = -scores

    print(f"\n{name}")
    print("MAE scores:", mae_scores)
    print("Mean MAE:", mae_scores.mean())
    print("Std MAE:", mae_scores.std())


# --------------------------------------------------
# 26. Hyperparameter tuning for Random Forest
# --------------------------------------------------

print("\n----- RANDOM FOREST HYPERPARAMETER TUNING -----")

rf = RandomForestRegressor(
    random_state=42
)

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5]
}


grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring="neg_mean_absolute_error",
    n_jobs=-1
)

grid_search.fit(
    X_train_processed,
    y_train
)


print("\nBest Parameters:")
print(grid_search.best_params_)


print("\nBest Cross-Validation MAE:")
print(-grid_search.best_score_)


# --------------------------------------------------
# 27. Evaluate tuned Random Forest on test set
# --------------------------------------------------

print("\n----- TUNED RANDOM FOREST TEST RESULTS -----")

best_rf = grid_search.best_estimator_

best_rf_predictions = best_rf.predict(
    X_test_processed
)

tuned_mae = mean_absolute_error(
    y_test,
    best_rf_predictions
)

tuned_rmse = mean_squared_error(
    y_test,
    best_rf_predictions
) ** 0.5

tuned_r2 = r2_score(
    y_test,
    best_rf_predictions
)


print("Model: Tuned Random Forest")
print("MAE:", tuned_mae)
print("RMSE:", tuned_rmse)
print("R2 Score:", tuned_r2)


# --------------------------------------------------
# END OF MODEL TRAINING
# --------------------------------------------------

print("\n----- MODEL TRAINING COMPLETED -----")
print("Algorithms evaluated:")
print("1. Linear Regression")
print("2. Decision Tree Regression")
print("3. Random Forest Regression")
print("4. Gradient Boosting Regression")
print("5. Extra Trees Regression")
print("6. Dummy Regressor (Baseline)")