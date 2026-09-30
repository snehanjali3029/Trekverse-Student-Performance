import pandas as pd

# Load cleaned dataset
df = pd.read_csv("data/student_mat_cleaned.csv")

# Create support category from final grade G3
def create_support_category(g3):
    if g3 < 8:
        return "High Support"
    elif g3 < 12:
        return "Medium Support"
    else:
        return "Low Support"


df["support_category"] = df["G3"].apply(create_support_category)

# Display category counts
print("\nSupport Category Distribution:")
print(df["support_category"].value_counts())

# Display percentages
print("\nSupport Category Percentage:")
print(
    df["support_category"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

# Show sample records
print("\nSample records:")
print(df[["G3", "support_category"]].head(10))