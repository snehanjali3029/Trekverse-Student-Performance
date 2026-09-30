import streamlit as st
import pandas as pd
import joblib
import sys
import os
import shap

# -----------------------------
# Project path
# -----------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from src.recommendations import generate_recommendations

# -----------------------------
# Load saved model and preprocessor
# -----------------------------
model = joblib.load(os.path.join(PROJECT_ROOT, "models", "random_forest_model.pkl"))
preprocessor = joblib.load(os.path.join(PROJECT_ROOT, "models", "preprocessor.pkl"))

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Intelligent Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------------
# Custom UI Styling
# -----------------------------
st.markdown("""
<style>

html, body {
    scroll-behavior: smooth;
}

/* Main underwater background */
.stApp {
    background:
        radial-gradient(circle at 15% 15%, rgba(56, 189, 248, 0.22), transparent 24%),
        radial-gradient(circle at 85% 70%, rgba(99, 102, 241, 0.20), transparent 30%),
        linear-gradient(180deg, #dff6ff 0%, #eaf8ff 35%, #eef2ff 70%, #f8fafc 100%);
}

/* Main container */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
    position: relative;
    z-index: 10;
}

/* Hero */
.hero {
    background: linear-gradient(135deg, #4f46e5, #7c3aed, #0891b2);
    padding: 35px;
    border-radius: 24px;
    color: white;
    margin-bottom: 30px;
    box-shadow: 0 12px 35px rgba(79, 70, 229, 0.25);
}

.hero h1 {
    color: white;
    font-size: 42px;
    margin-bottom: 10px;
}

.hero p {
    color: #eef2ff;
    font-size: 18px;
    margin-bottom: 0;
}

/* Section cards */
.section-card {
    background: rgba(255, 255, 255, 0.92);
    padding: 25px;
    border-radius: 20px;
    margin: 20px 0;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.07);
    border: 1px solid #e5e7eb;
}

/* Prediction card */
.prediction-card {
    background: linear-gradient(135deg, #eef2ff, #f5f3ff);
    padding: 30px;
    border-radius: 22px;
    text-align: center;
    border: 2px solid #c7d2fe;
    box-shadow: 0 8px 25px rgba(79, 70, 229, 0.12);
    margin: 20px 0;
}

.prediction-title {
    font-size: 18px;
    color: #4b5563;
    margin-bottom: 5px;
}

.prediction-score {
    font-size: 52px;
    font-weight: 800;
    color: #4f46e5;
}

.prediction-subtitle {
    font-size: 15px;
    color: #6b7280;
}

/* Recommendation cards */
.recommendation-card {
    background: linear-gradient(135deg, #ecfeff, #eff6ff);
    padding: 18px 22px;
    border-radius: 16px;
    margin: 12px 0;
    border-left: 5px solid #0891b2;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.05);
}

/* SHAP cards */
.shap-positive {
    background: #ecfdf5;
    border-left: 5px solid #10b981;
    padding: 15px 20px;
    border-radius: 14px;
    margin: 10px 0;
}

.shap-negative {
    background: #fff7ed;
    border-left: 5px solid #f97316;
    padding: 15px 20px;
    border-radius: 14px;
    margin: 10px 0;
}

/* Buttons */
.stButton > button,
.stFormSubmitButton > button {
    width: 100%;
    border-radius: 12px;
    border: none;
    padding: 12px 20px;
    font-size: 17px;
    font-weight: 700;
    background: linear-gradient(90deg, #4f46e5, #7c3aed);
    color: white;
    box-shadow: 0 5px 15px rgba(79, 70, 229, 0.25);
    transition: all 0.2s ease;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(79, 70, 229, 0.35);
}

/* Input widgets */
div[data-baseweb="select"] > div {
    border-radius: 10px;
}

/* Metrics */
div[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.92);
    padding: 18px;
    border-radius: 16px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.05);
}

/* ============================= */
/* Minimal underwater bubble */
/* One subtle bubble only */
/* ============================= */

.stApp::before {
    content: "•";
    position: fixed;
    left: 24vw;
    top: 520px;
    font-size: 18px;
    color: rgba(255, 255, 255, 0.75);
    text-shadow: 0 0 6px rgba(120, 190, 255, 0.45);
    pointer-events: none;
    z-index: 1;
}

/* ============================= */
/* SINGLE PROFESSIONAL SCROLL FISH */
/* Fish is INSIDE the Streamlit scroll container. */
/* The anonymous scroll() timeline follows the real page scroll. */
/* ============================= */

.fish {
    position: sticky;
    top: 360px;
    display: block;
    width: max-content;
    margin-top: -70px;
    margin-left: 12vw;
    margin-bottom: -58px;

    font-size: 58px;
    line-height: 1;
    opacity: 0.22;
    pointer-events: none;
    user-select: none;
    z-index: 1;
    will-change: transform;

    animation: professionalFishScroll 1ms linear both;
    animation-timeline: scroll();
    animation-range: 0% 100%;
}

@keyframes professionalFishScroll {
    0% {
        transform: translateX(0) translateY(0) scaleX(1);
    }
    25% {
        transform: translateX(12vw) translateY(18px) scaleX(1);
    }
    50% {
        transform: translateX(26vw) translateY(-10px) scaleX(1);
    }
    75% {
        transform: translateX(40vw) translateY(12px) scaleX(-1);
    }
    100% {
        transform: translateX(54vw) translateY(-5px) scaleX(-1);
    }
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Title / Hero
# -----------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🎓 Intelligent Student Performance Predictor</h1>
        <p>
            🤖 AI-powered academic prediction
            • 🧠 Explainable AI
            • 📚 Personalized Learning Recommendations
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Single background fish
# -----------------------------
st.markdown(
    """
    <div class="fish" aria-hidden="true">🐠</div>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Student input form
# -----------------------------
st.header("🌊 Student Information")

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

    st.subheader("🐚 Student Lifestyle and Support")

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
    prediction = float(model.predict(processed_input)[0])

    # Keep prediction within valid G3 range
    prediction = max(0, min(20, prediction))

    # Warning for high absences
    if absences > 20:
        st.warning(
            "The entered absence count is relatively high. "
            "Please verify that the value is correct."
        )

    st.success("Prediction completed successfully!")

    # -----------------------------
    # Prediction Result
    # -----------------------------
    st.header("📊 Prediction Result")

    st.markdown(
        f"""
        <div class="prediction-card">
            <div class="prediction-title">Predicted Final Grade (G3)</div>
            <div class="prediction-score">{prediction:.2f} / 20</div>
            <div class="prediction-subtitle">
                AI prediction based on the student's academic and lifestyle information.
            </div>
        </div>
        """,
        unsafe_allow_html=True
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
        st.markdown(
            f"""
            <div class="recommendation-card">
                💡 {recommendation}
            </div>
            """,
            unsafe_allow_html=True
        )

    # -----------------------------
    # Explainable AI - SHAP
    # -----------------------------
    st.header("🔍 Why did the model make this prediction?")

    st.write(
        "SHAP shows which input features had the largest influence on "
        "this individual prediction. These are model explanations, not causal conclusions."
    )

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

    top_features = explanation_df.head(5)

    for _, row in top_features.iterrows():

        feature = row["Feature"]
        impact = row["SHAP Value"]

        if impact > 0:
            st.success(
                f"**{feature}** → increased the predicted score "
                f"by approximately {impact:.2f}"
            )
        elif impact < 0:
            st.warning(
                f"**{feature}** → decreased the predicted score "
                f"by approximately {abs(impact):.2f}"
            )
        else:
            st.info(
                f"**{feature}** → had approximately zero influence "
                f"on this prediction."
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

# -----------------------------
# Footer
# -----------------------------
st.markdown(
    """
    <br>
    <div style="
        text-align:center;
        padding:25px;
        margin-top:40px;
        border-radius:20px;
        background:rgba(255,255,255,0.75);
        color:#475569;
    ">
        🌊 <b>AI Learning Ocean</b><br>
        Intelligent Student Performance Prediction & Personalized Learning
        <br><br>
        <small>Decision-support system • Predictions are not guaranteed outcomes.</small>
    </div>
    """,
    unsafe_allow_html=True
)
