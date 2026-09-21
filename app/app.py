import streamlit as st
import pandas as pd
import joblib
import sys
import os
import shap

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.recommendations import generate_recommendations

# -----------------------------
# Load saved model and preprocessor
# -----------------------------
model = joblib.load("models/random_forest_model.pkl")
preprocessor = joblib.load("models/preprocessor.pkl")


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)


# -----------------------------
# Title
# -----------------------------
st.title("🎓 Student Performance Prediction System")

st.write(
    "Enter student details below to predict the final academic score "
    "and receive personalized learning recommendations."
)


# -----------------------------
# Student input form
# -----------------------------
st.header("Student Information")

with st.form("student_form"):

    col1, col2, col3 = st.columns(3)

    with col1:
        school = st.selectbox("School", ["GP", "MS"])
        sex = st.selectbox("Gender", ["F", "M"])
        age = st.number_input("Age", min_value=15, max_value=22, value=17)
        address = st.selectbox("Address", ["U", "R"])
        famsize = st.selectbox("Family Size", ["GT3", "LE3"])

    with col2:
        Pstatus = st.selectbox("Parent Status", ["T", "A"])
        Medu = st.number_input("Mother Education", 0, 4, 2)
        Fedu = st.number_input("Father Education", 0, 4, 2)
        Mjob = st.selectbox(
            "Mother Job",
            ["teacher", "health", "services", "at_home", "other"]
        )
        Fjob = st.selectbox(
            "Father Job",
            ["teacher", "health", "services", "at_home", "other"]
        )

    with col3:
        reason = st.selectbox(
            "Reason for Choosing School",
            ["course", "home", "reputation", "other"]
        )
        guardian = st.selectbox(
            "Guardian",
            ["mother", "father", "other"]
        )
        traveltime = st.number_input("Travel Time", 1, 4, 2)
        studytime = st.number_input("Study Time", 1, 4, 2)
        failures = st.number_input("Previous Failures", 0, 4, 0)

    st.subheader("Student Lifestyle and Support")

    col4, col5, col6 = st.columns(3)

    with col4:
        famrel = st.number_input("Family Relationship", 1, 5, 4)
        freetime = st.number_input("Free Time", 1, 5, 3)
        goout = st.number_input("Going Out", 1, 5, 3)
        Dalc = st.number_input("Workday Alcohol Consumption", 1, 5, 1)
        Walc = st.number_input("Weekend Alcohol Consumption", 1, 5, 1)

    with col5:
        health = st.number_input("Health Status", 1, 5, 3)
        absences = st.number_input("Number of Absences", 0, 93, 5)
        schoolsup = st.selectbox("School Support", ["yes", "no"])
        famsup = st.selectbox("Family Support", ["yes", "no"])
        paid = st.selectbox("Extra Paid Classes", ["yes", "no"])

    with col6:
        activities = st.selectbox("Extra-curricular Activities", ["yes", "no"])
        nursery = st.selectbox("Attended Nursery School", ["yes", "no"])
        higher = st.selectbox("Wants Higher Education", ["yes", "no"])
        internet = st.selectbox("Internet Access", ["yes", "no"])
        romantic = st.selectbox("Romantic Relationship", ["yes", "no"])

    submit = st.form_submit_button("🔮 Predict Final Grade")


# -----------------------------
# Prediction
# -----------------------------
if submit:

    student_data = {
        "school": school,
        "sex": sex,
        "age": age,
        "address": address,
        "famsize": famsize,
        "Pstatus": Pstatus,
        "Medu": Medu,
        "Fedu": Fedu,
        "Mjob": Mjob,
        "Fjob": Fjob,
        "reason": reason,
        "guardian": guardian,
        "traveltime": traveltime,
        "studytime": studytime,
        "failures": failures,
        "schoolsup": schoolsup,
        "famsup": famsup,
        "paid": paid,
        "activities": activities,
        "nursery": nursery,
        "higher": higher,
        "internet": internet,
        "romantic": romantic,
        "famrel": famrel,
        "freetime": freetime,
        "goout": goout,
        "Dalc": Dalc,
        "Walc": Walc,
        "health": health,
        "absences": absences
    }

    input_df = pd.DataFrame([student_data])

    # Transform input using saved preprocessing pipeline
    processed_input = preprocessor.transform(input_df)

    # Predict final grade
    prediction = model.predict(processed_input)[0]
        # -----------------------------
    # Prediction validation
    # -----------------------------
    if prediction < 0:
        prediction = 0
    elif prediction > 20:
        prediction = 20
        if absences > 20:
            st.warning(
            "The entered absence count is relatively high. "
            "Please verify that the value is correct."
        )

    st.success("Prediction completed successfully!")

    st.header("📊 Prediction Result")

    st.metric(
        label="Predicted Final Grade (G3)",
        value=f"{prediction:.2f} / 20"
    )
        # -----------------------------
    # Personalized Recommendations
    # -----------------------------
    if absences <= 5:
        absence_level = "Low"
    elif absences <= 10:
        absence_level = "Medium"
    else:
        absence_level = "High"

    recommendation_student = {
        "absence_level": absence_level,
        "failures": failures,
        "studytime": studytime
    }

    recommendations = generate_recommendations(
        recommendation_student,
        prediction
    )

    st.header("📚 Personalized Learning Recommendations")

    for recommendation in recommendations:
        st.info(recommendation)

            # -----------------------------
    # Explainable AI - SHAP
    # -----------------------------
    st.header("🔍 Why did the model make this prediction?")

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(processed_input)

    feature_names = preprocessor.get_feature_names_out()

    shap_values_for_student = shap_values[0]

    explanation_df = pd.DataFrame({
        "Feature": feature_names,
        "SHAP Value": shap_values_for_student
    })

    explanation_df["Absolute Impact"] = explanation_df["SHAP Value"].abs()

    explanation_df = explanation_df.sort_values(
        by="Absolute Impact",
        ascending=False
    )

    st.write(
        "The following features had the largest influence on "
        "this individual prediction."
    )

    top_features = explanation_df.head(5)

    for _, row in top_features.iterrows():

        feature = row["Feature"]
        impact = row["SHAP Value"]

        if impact > 0:
            st.success(
                f"**{feature}** → increased the predicted score "
                f"by approximately {impact:.2f}"
            )
        else:
            st.warning(
                f"**{feature}** → decreased the predicted score "
                f"by approximately {abs(impact):.2f}"
            )
            # -----------------------------
# Model Performance Dashboard
# -----------------------------
st.header("📈 Model Performance Dashboard")

st.write(
    "The following metrics were obtained from the held-out test set "
    "used during model evaluation."
)

performance_data = pd.DataFrame({
    "Model": [
        "Dummy Baseline",
        "Linear Regression",
        "Decision Tree",
        "Random Forest",
        "Gradient Boosting",
        "Extra Trees"
    ],
   "MAE": [
    3.646,
    3.395,
    3.595,
    2.998,
    3.113,
    3.313
],
    "RMSE": [
    4.550,
    4.196,
    4.784,
    3.795,
    3.928,
    4.264
],
   "R²": [
    -0.010,
    0.141,
    -0.116,
    0.298,
    0.248,
    0.113
]
})

st.dataframe(
    performance_data,
    width="stretch",
    hide_index=True
)

st.subheader("Tuned Random Forest Test Performance")

metric1, metric2, metric3 = st.columns(3)

with metric1:
    st.metric("MAE", "3.06")

with metric2:
    st.metric("RMSE", "3.83")

with metric3:
    st.metric("R²", "0.285")

st.caption(
    "MAE measures average absolute prediction error, RMSE gives greater "
    "weight to larger errors, and R² represents the proportion of variance "
    "explained by the model on the test set."
)