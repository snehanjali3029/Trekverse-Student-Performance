import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("data/student_mat_cleaned.csv")


# ============================================================
# 2. DATASET SIZE
# ============================================================

print("----- DATASET SIZE -----")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])


# ============================================================
# 3. FIRST 5 ROWS
# ============================================================

print("\n----- FIRST 5 ROWS -----")
print(df.head())


# ============================================================
# 4. COLUMN NAMES
# ============================================================

print("\n----- COLUMN NAMES -----")
print(df.columns.tolist())


# ============================================================
# 5. DATA TYPES
# ============================================================

print("\n----- DATA TYPES -----")
print(df.dtypes)


# ============================================================
# 6. MISSING VALUES
# ============================================================

print("\n----- MISSING VALUES -----")
print(df.isnull().sum())


# ============================================================
# 7. DUPLICATE ROWS
# ============================================================

print("\n----- DUPLICATE ROWS -----")
print("Duplicates:", df.duplicated().sum())


# ============================================================
# 8. FINAL GRADE (G3) SUMMARY
# ============================================================

print("\n----- FINAL GRADE (G3) SUMMARY -----")
print(df["G3"].describe())


# ============================================================
# 9. CATEGORICAL VALUES
# ============================================================

print("\n----- CATEGORICAL VALUES -----")

categorical_columns = [
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

for column in categorical_columns:
    print(f"{column}: {df[column].unique()}")


# ============================================================
# 10. NUMERICAL RANGE CHECKS
# ============================================================

print("\n----- NUMERICAL RANGE CHECKS -----")

range_checks = {
    "age": (15, 22),
    "Medu": (0, 4),
    "Fedu": (0, 4),
    "traveltime": (1, 4),
    "studytime": (1, 4),
    "failures": (0, 4),
    "famrel": (1, 5),
    "freetime": (1, 5),
    "goout": (1, 5),
    "Dalc": (1, 5),
    "Walc": (1, 5),
    "health": (1, 5),
    "absences": (0, 93),
    "G1": (0, 20),
    "G2": (0, 20),
    "G3": (0, 20)
}

for column, (minimum, maximum) in range_checks.items():

    actual_min = df[column].min()
    actual_max = df[column].max()

    print(
        f"{column}: "
        f"actual range = {actual_min} to {actual_max}, "
        f"expected range = {minimum} to {maximum}"
    )


# ============================================================
# 11. CHECK FOR VALUES OUTSIDE EXPECTED RANGES
# ============================================================

print("\n----- VALUES OUTSIDE EXPECTED RANGES -----")

range_error_found = False

for column, (minimum, maximum) in range_checks.items():

    invalid_values = df[
        (df[column] < minimum) |
        (df[column] > maximum)
    ][column]

    if len(invalid_values) > 0:

        range_error_found = True

        print(
            f"{column}: INVALID VALUES FOUND -> "
            f"{invalid_values.tolist()}"
        )

    else:

        print(
            f"{column}: No values outside expected range"
        )

if not range_error_found:
    print("\nNo numerical range errors found.")


# ============================================================
# 12. CHECK G3 = 0
# ============================================================

print("\n----- G3 = 0 CHECK -----")

g3_zero_count = (df["G3"] == 0).sum()

print("Number of students with G3 = 0:", g3_zero_count)

if g3_zero_count > 0:

    print("\nRows where G3 = 0:")

    print(
        df[df["G3"] == 0][
            ["school", "sex", "age", "absences", "G1", "G2", "G3"]
        ]
    )

else:

    print("No students have G3 = 0.")

    # ============================================================
# 13. OUTLIER CHECK USING IQR
# ============================================================

print("\n----- OUTLIER CHECK USING IQR -----")

outlier_columns = [
    "absences",
    "G1",
    "G2",
    "G3",
    "studytime",
    "failures"
]

for column in outlier_columns:

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[
        (df[column] < lower_bound) |
        (df[column] > upper_bound)
    ]

    print(f"\n{column}")
    print(f"Q1: {Q1}")
    print(f"Q3: {Q3}")
    print(f"IQR: {IQR}")
    print(f"Lower Bound: {lower_bound}")
    print(f"Upper Bound: {upper_bound}")
    print(f"Number of outliers: {len(outliers)}")

    if len(outliers) > 0:
        print("Outlier values:")
        print(outliers[column].tolist())


# ============================================================
# 14. G3 DISTRIBUTION GRAPH
# ============================================================

plt.figure(figsize=(8, 5))

plt.hist(
    df["G3"],
    bins=21,
    edgecolor="black"
)

plt.title("Distribution of Final Grades (G3)")
plt.xlabel("Final Grade (G3)")
plt.ylabel("Number of Students")

plt.savefig(
    "reports/g3_distribution.png",
    bbox_inches="tight"
)

plt.close()

print("\nG3 distribution graph saved to reports/g3_distribution.png")
# ============================================================
# 14. STUDY TIME VS FINAL GRADE
# ============================================================

print("\n----- STUDY TIME VS FINAL GRADE -----")

print(
    df.groupby("studytime")["G3"].agg(
        ["count", "mean", "median", "min", "max"]
    )
)

plt.figure(figsize=(8, 5))

df.boxplot(
    column="G3",
    by="studytime"
)

plt.title("Final Grade (G3) by Study Time")
plt.suptitle("")
plt.xlabel("Study Time")
plt.ylabel("Final Grade (G3)")

plt.savefig(
    "reports/g3_by_studytime.png",
    bbox_inches="tight"
)

plt.close()

print(
    "\nStudy time vs final grade graph saved to "
    "reports/g3_by_studytime.png"
)
# ============================================================
# 15. ABSENCES VS FINAL GRADE
# ============================================================

print("\n----- ABSENCES VS FINAL GRADE -----")

print(
    df[["absences", "G3"]].corr()
)

plt.figure(figsize=(8, 5))

plt.scatter(
    df["absences"],
    df["G3"],
    alpha=0.6
)

plt.title("Absences vs Final Grade (G3)")
plt.xlabel("Number of Absences")
plt.ylabel("Final Grade (G3)")

plt.savefig(
    "reports/g3_vs_absences.png",
    bbox_inches="tight"
)

plt.close()

print(
    "\nAbsences vs final grade graph saved to "
    "reports/g3_vs_absences.png"
)
# ============================================================
# 17. PREVIOUS GRADES VS FINAL GRADE
# ============================================================

print("\n----- PREVIOUS GRADES VS FINAL GRADE -----")

print("\nCorrelation between G1 and G3:")
print(df["G1"].corr(df["G3"]))

print("\nCorrelation between G2 and G3:")
print(df["G2"].corr(df["G3"]))


# G1 vs G3

plt.figure(figsize=(8, 5))

plt.scatter(
    df["G1"],
    df["G3"],
    alpha=0.6
)

plt.title("First Period Grade (G1) vs Final Grade (G3)")
plt.xlabel("G1")
plt.ylabel("G3")

plt.savefig(
    "reports/g1_vs_g3.png",
    bbox_inches="tight"
)

plt.close()


# G2 vs G3

plt.figure(figsize=(8, 5))

plt.scatter(
    df["G2"],
    df["G3"],
    alpha=0.6
)

plt.title("Second Period Grade (G2) vs Final Grade (G3)")
plt.xlabel("G2")
plt.ylabel("G3")

plt.savefig(
    "reports/g2_vs_g3.png",
    bbox_inches="tight"
)

plt.close()

print("\nG1 vs G3 graph saved to reports/g1_vs_g3.png")
print("G2 vs G3 graph saved to reports/g2_vs_g3.png")

# ============================================================
# 16.. END MESSAGE
# ============================================================

print("\n========================================")
print("DATA INSPECTION COMPLETED SUCCESSFULLY")
print("========================================")

# ============================================================
# 18. FAILURES VS FINAL GRADE
# ============================================================

print("\n----- FAILURES VS FINAL GRADE -----")

print(
    df.groupby("failures")["G3"].agg(
        ["count", "mean", "median", "min", "max"]
    )
)

plt.figure(figsize=(8, 5))

df.boxplot(
    column="G3",
    by="failures"
)

plt.title("Final Grade (G3) by Number of Failures")
plt.suptitle("")
plt.xlabel("Number of Previous Failures")
plt.ylabel("Final Grade (G3)")

plt.savefig(
    "reports/g3_by_failures.png",
    bbox_inches="tight"
)

plt.close()

print(
    "\nFailures vs final grade graph saved to "
    "reports/g3_by_failures.png"
)
# ============================================================
# 19. GENDER VS FINAL GRADE
# ============================================================

print("\n----- GENDER VS FINAL GRADE -----")

print(
    df.groupby("sex")["G3"].agg(
        ["count", "mean", "median", "min", "max"]
    )
)

plt.figure(figsize=(8, 5))

df.boxplot(
    column="G3",
    by="sex"
)

plt.title("Final Grade (G3) by Gender")
plt.suptitle("")
plt.xlabel("Gender")
plt.ylabel("Final Grade (G3)")

plt.savefig(
    "reports/g3_by_gender.png",
    bbox_inches="tight"
)

plt.close()

print(
    "\nGender vs final grade graph saved to "
    "reports/g3_by_gender.png"
)
# ============================================================
# 20. SCHOOL VS FINAL GRADE
# ============================================================

print("\n----- SCHOOL VS FINAL GRADE -----")

print(
    df.groupby("school")["G3"].agg(
        ["count", "mean", "median", "min", "max"]
    )
)

plt.figure(figsize=(8, 5))

df.boxplot(
    column="G3",
    by="school"
)

plt.title("Final Grade (G3) by School")
plt.suptitle("")
plt.xlabel("School")
plt.ylabel("Final Grade (G3)")

plt.savefig(
    "reports/g3_by_school.png",
    bbox_inches="tight"
)

plt.close()

print(
    "\nSchool vs final grade graph saved to "
    "reports/g3_by_school.png"
)