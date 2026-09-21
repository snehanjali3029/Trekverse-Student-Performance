# Data Dictionary

## Dataset Information

**Dataset:** Student Performance - Mathematics  
**Source:** UCI Machine Learning Repository  
**Dataset ID:** 320  
**Dataset File:** student-mat  
**Number of Records:** 395  
**Number of Features:** 33 columns  
**Target Variable:** G3 (final grade)  
**License:** CC BY 4.0  

The dataset contains information about students' demographic characteristics,
family background, school-related information, study habits, social activities,
absences, and grades.

The project uses the Mathematics (`student-mat`) dataset.

---

## Feature Dictionary

| Feature | Description | Data Type | Values / Range |
|---|---|---|---|
| school | Student's school | Categorical | GP, MS |
| sex | Student's gender recorded in the dataset | Categorical | F, M |
| age | Student's age | Integer | 15–22 |
| address | Type of home address | Categorical | U, R |
| famsize | Family size | Categorical | LE3, GT3 |
| Pstatus | Parents' cohabitation status | Categorical | T, A |
| Medu | Mother's education level | Integer | 0–4 |
| Fedu | Father's education level | Integer | 0–4 |
| Mjob | Mother's job | Categorical | teacher, health, services, at_home, other |
| Fjob | Father's job | Categorical | teacher, health, services, at_home, other |
| reason | Reason for choosing the school | Categorical | home, reputation, course, other |
| guardian | Student's guardian | Categorical | mother, father, other |
| traveltime | Travel time from home to school | Integer | 1–4 |
| studytime | Weekly study time | Integer | 1–4 |
| failures | Number of past class failures | Integer | 0–4 |
| schoolsup | Extra educational support from school | Categorical | yes, no |
| famsup | Extra educational support from family | Categorical | yes, no |
| paid | Extra paid classes within the course subject | Categorical | yes, no |
| activities | Participation in extracurricular activities | Categorical | yes, no |
| nursery | Attended nursery school | Categorical | yes, no |
| higher | Wants to take higher education | Categorical | yes, no |
| internet | Internet access at home | Categorical | yes, no |
| romantic | Has a romantic relationship | Categorical | yes, no |
| famrel | Quality of family relationships | Integer | 1–5 |
| freetime | Free time after school | Integer | 1–5 |
| goout | Going out with friends | Integer | 1–5 |
| Dalc | Workday alcohol consumption | Integer | 1–5 |
| Walc | Weekend alcohol consumption | Integer | 1–5 |
| health | Current health status | Integer | 1–5 |
| absences | Number of school absences | Integer | 0–93 |
| G1 | First-period grade | Integer | 0–20 |
| G2 | Second-period grade | Integer | 0–20 |
| G3 | Final grade | Integer | 0–20 |

---

## Encoded Feature Details

### Education Level

`Medu` and `Fedu` use the following scale:

| Value | Meaning |
|---|---|
| 0 | None |
| 1 | Primary education (4th grade) |
| 2 | 5th to 9th grade |
| 3 | Secondary education |
| 4 | Higher education |

### Travel Time

| Value | Meaning |
|---|---|
| 1 | <15 minutes |
| 2 | 15–30 minutes |
| 3 | 30 minutes–1 hour |
| 4 | >1 hour |

### Study Time

| Value | Meaning |
|---|---|
| 1 | <2 hours |
| 2 | 2–5 hours |
| 3 | 5–10 hours |
| 4 | >10 hours |

### Family Relationship, Free Time, Going Out, Alcohol Consumption and Health

These variables use a scale from 1 to 5.

Generally:

- 1 = lowest level
- 5 = highest level

The exact interpretation depends on the individual variable.

---

## Target Variable

### G3 - Final Grade

`G3` represents the student's final Mathematics grade.

- Minimum: 0
- Maximum: 20
- Mean: approximately 10.42

The project uses `G3` as the **regression target**.

The model predicts a continuous final score on a 0–20 scale.

---

## Features Excluded From Model Training

The following columns are not used as input features in the final prediction model:

### G3

`G3` is the target variable, so it must not be provided to the model during prediction.

### G1 and G2

`G1` and `G2` are earlier-period grades.

They are excluded because they have a strong relationship with the final grade (`G3`) and could make the prediction task easier in a way that does not match the project's goal of predicting performance using student characteristics, engagement, and study-related information.

Therefore, the model uses:

- Demographic information
- Family background
- School information
- Study habits
- Social/engagement information
- Absence information

---

## Data Quality Notes

The Mathematics dataset contains:

- **395 records**
- **33 columns**
- **No missing values**
- **No duplicate records**
- Values were checked against expected ranges.
- No invalid categorical values were found.
- Some numerical values were identified as statistical outliers using the IQR method.

The identified outliers were retained because they represent valid observations in the original dataset rather than obvious data-entry errors.

For example, some students have relatively high absence counts. These observations were not automatically removed.

---

## Data Privacy

The dataset does not contain directly identifying information such as:

- Student names
- Phone numbers
- Email addresses
- Home addresses
- Student identification numbers

The project therefore avoids using personally identifiable information in the prediction system.

---

## Dataset Limitations

1. The dataset represents students from secondary education and should not automatically be assumed to represent all educational populations.

2. The data comes from specific schools and educational contexts, so model performance may differ when applied to other institutions or populations.

3. The dataset contains sensitive or potentially sensitive characteristics such as family background, health-related self-reporting, and alcohol-consumption variables. These should be handled carefully.

4. Associations found in the data should not be interpreted as proof of causation.

5. The model predictions are intended as decision-support information and should not be treated as definitive judgments about students.

6. The model should be validated on new data before being used in a different educational environment.

---

## Source

UCI Machine Learning Repository:

**Student Performance Dataset**

Dataset ID: 320

DOI: 10.24432/C5TG7T

License: CC BY 4.0