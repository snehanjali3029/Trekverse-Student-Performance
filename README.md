# Intelligent Student Performance Prediction and Personalized Learning Recommendation System

## Project Overview

This project is an AI/ML-based system that predicts student academic performance and provides personalized learning recommendations.

The system uses student demographic information, family background, study habits, engagement information, and school attendance data to predict the student's final academic score.

The project also uses Explainable AI (XAI) techniques to provide insight into the factors influencing individual predictions.

---

## Objectives

The main objectives of the project are:

- Predict student final academic performance.
- Identify factors used by the machine learning model.
- Provide personalized learning recommendations.
- Compare multiple machine learning algorithms.
- Evaluate model performance using appropriate metrics.
- Explain predictions using SHAP.
- Analyze student engagement patterns using clustering.
- Provide an interactive web application using Streamlit.
- Test the system using automated unit tests and robustness tests.

---

## Dataset

The project uses the **Student Performance Dataset** from the UCI Machine Learning Repository.

**Dataset:** Student Performance  
**Course:** Mathematics (`student-mat`)  
**Dataset ID:** 320  
**Records:** 395  
**Columns:** 33  
**Target:** G3 (Final Grade)  
**Grade Range:** 0–20  
**License:** CC BY 4.0

The dataset contains information about:

- Student demographics
- Family background
- School information
- Study habits
- Social activities
- Absences
- Previous grades

### Dataset Source

UCI Machine Learning Repository:

https://archive.ics.uci.edu/dataset/320/student%2Bperformance

---

## Machine Learning Approach

The project performs regression to predict the student's final grade (`G3`).

The following input categories are used:

- Demographic information
- Family information
- School information
- Study habits
- Student engagement
- Absence information

`G1`, `G2`, and `G3` are excluded from the prediction inputs.

`G3` is the target variable, while `G1` and `G2` are excluded because they are earlier grades that have a strong relationship with the final grade.

---

## Data Preprocessing

The preprocessing pipeline includes:

- Data validation
- Missing-value checking
- Duplicate checking
- Range validation
- Outlier analysis
- Numerical feature handling
- Categorical feature encoding
- Train/test splitting

Categorical variables are converted using One-Hot Encoding.

The dataset is divided into:

- 80% training data
- 20% test data

A fixed random state is used to improve reproducibility.

---

## Machine Learning Models

The following models were evaluated:

1. Dummy Regression Baseline
2. Linear Regression
3. Decision Tree Regression
4. Random Forest Regression
5. Gradient Boosting Regression
6. Extra Trees Regression

Random Forest was further tuned using GridSearchCV.

### Evaluation Metrics

The models were evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score
- 5-fold Cross-Validation

---

## Model Performance

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Dummy Baseline | 3.646 | 4.550 | -0.010 |
| Linear Regression | 3.395 | 4.196 | 0.141 |
| Decision Tree | 3.595 | 4.784 | -0.116 |
| Random Forest | 2.998 | 3.795 | 0.298 |
| Gradient Boosting | 3.113 | 3.928 | 0.248 |
| Extra Trees | 3.313 | 4.264 | 0.113 |

The tuned Random Forest model was used for the final prediction application.

### Tuned Random Forest

Parameters:

- `n_estimators = 50`
- `max_depth = 10`
- `min_samples_split = 2`
- `random_state = 42`

Tuned model test results:

- MAE: approximately **3.06**
- RMSE: approximately **3.83**
- R²: approximately **0.285**

These results are based on the held-out test set and should not be interpreted as guaranteed performance on new educational populations.

---

## Explainable AI

SHAP (SHapley Additive exPlanations) is used to explain model predictions.

The project provides:

- Global feature importance
- SHAP summary visualization
- Individual student explanations

Important model features include:

- Absences
- Previous failures
- Health
- Going out
- Age
- Study time

Feature importance represents model behavior and should not be interpreted as proof of causation.

---

## Personalized Recommendations

The system generates rule-based recommendations using:

- Absence level
- Previous failures
- Study time
- Predicted score

Examples include recommendations related to:

- Attendance
- Study schedule
- Reviewing previously unsuccessful subjects
- Additional academic support
- Continued practice

The recommendations are intended as decision-support suggestions and are not guarantees of student outcomes.

---

## Student Engagement Clustering

K-Means clustering is used to explore student engagement patterns.

The clustering uses:

- Study time
- Free time
- Going out
- Absences
- Extracurricular activities

Different values of K were evaluated using silhouette scores.

K=2 was selected for the exploratory clustering analysis because it produced the highest silhouette score among the tested values.

The clustering results are exploratory and should not be interpreted as fixed student classifications.

---

## Streamlit Application

The project includes an interactive Streamlit web application.

The application provides:

- Student input form
- Performance prediction
- Personalized recommendations
- Individual SHAP explanation
- Model performance dashboard
- Input validation
- High-absence warning

### Run the application

Activate the virtual environment and run:

```bash
streamlit run app/app.py