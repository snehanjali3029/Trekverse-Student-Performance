import pandas as pd


# Load dataset
df = pd.read_csv("data/student_mat_cleaned.csv")

print("----- ORIGINAL DATASET -----")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])


# Create absence level
def classify_absence(absences):
    if absences <= 5:
        return "Low"
    elif absences <= 10:
        return "Medium"
    else:
        return "High"


df["absence_level"] = df["absences"].apply(classify_absence)


print("\n----- ABSENCE LEVEL COUNTS -----")
print(df["absence_level"].value_counts())


print("\n----- SAMPLE DATA -----")
print(
    df[
        ["absences", "absence_level"]
    ].head(15)
)


print("\n----- AVERAGE G3 BY ABSENCE LEVEL -----")
print(
    df.groupby("absence_level")["G3"].agg(
        ["count", "mean", "median", "min", "max"]
    )
)