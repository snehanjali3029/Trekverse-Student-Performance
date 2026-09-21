import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# Load cleaned dataset
df = pd.read_csv("data/student_mat_cleaned.csv")

print("----- ORIGINAL DATASET -----")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# Target variable
y = df["G3"]

# Remove target and G1/G2 from input features
# G1 and G2 are previous period grades.
# We are excluding them for the early-support prediction model.
X = df.drop(columns=["G3", "G1", "G2"])

print("\n----- TARGET (y) -----")
print("Target column: G3")
print("Target shape:", y.shape)

print("\n----- FEATURES (X) -----")
print("Feature shape:", X.shape)

print("\n----- FEATURE COLUMNS -----")
print(X.columns.tolist())

print("\n----- TARGET VALUES -----")
print(y.head())

print("\n----- CHECK FOR MISSING VALUES -----")
print(X.isnull().sum().sum())

print("\n----- FINAL CHECK -----")
print("X rows:", len(X))
print("y rows:", len(y))

if len(X) == len(y):
    print("X and y have matching rows.")
else:
    print("ERROR: X and y row counts do not match.")

    print("\n----- NUMERICAL FEATURES -----")
numerical_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
print(numerical_features)

print("\n----- CATEGORICAL FEATURES -----")
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
print(categorical_features)

print("\n----- FEATURE TYPE COUNT -----")
print("Numerical features:", len(numerical_features))
print("Categorical features:", len(categorical_features))

print("\n----- CREATING PREPROCESSING PIPELINE -----")

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

print("Preprocessing pipeline created successfully.")
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\n----- TRAIN / TEST SPLIT -----")
print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)
print("Training target:", y_train.shape)
print("Testing target:", y_test.shape)
# Fit preprocessing only on training data
X_train_processed = preprocessor.fit_transform(X_train)

# Apply the already-fitted preprocessing to test data
X_test_processed = preprocessor.transform(X_test)

print("\n----- PROCESSED DATA -----")
print("Processed training data shape:", X_train_processed.shape)
print("Processed testing data shape:", X_test_processed.shape)