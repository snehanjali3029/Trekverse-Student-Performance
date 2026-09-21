def generate_recommendations(student_data, predicted_score):
    """
    Generate personalized learning recommendations
    using student information and predicted final score.
    """

    recommendations = []

    # --------------------------------------------------
    # 1. Absence-based recommendation
    # --------------------------------------------------

    if student_data["absence_level"] == "High":
        recommendations.append(
            "Focus on improving regular attendance and participation "
            "in classes."
        )

    elif student_data["absence_level"] == "Medium":
        recommendations.append(
            "Try to maintain more consistent attendance and class participation."
        )


    # --------------------------------------------------
    # 2. Previous failures
    # --------------------------------------------------

    if student_data["failures"] > 0:
        recommendations.append(
            "Review topics from previously unsuccessful subjects "
            "and practice them regularly."
        )


    # --------------------------------------------------
    # 3. Study time
    # --------------------------------------------------

    if student_data["studytime"] <= 2:
        recommendations.append(
            "Increase dedicated study time and follow a consistent "
            "daily study schedule."
        )


    # --------------------------------------------------
    # 4. Predicted score
    # --------------------------------------------------

    if predicted_score < 8:
        recommendations.append(
            "Consider additional academic support and focus on "
            "strengthening foundational concepts."
        )

    elif predicted_score < 12:
        recommendations.append(
            "Continue regular practice and focus on improving "
            "understanding of difficult topics."
        )

    else:
        recommendations.append(
            "Maintain the current learning routine and continue "
            "practicing to strengthen academic performance."
        )


    # --------------------------------------------------
    # 5. If no specific recommendation was generated
    # --------------------------------------------------

    if not recommendations:
        recommendations.append(
            "Continue the current study routine and maintain "
            "consistent academic engagement."
        )

    return recommendations


# ======================================================
# TEST THE RECOMMENDATION SYSTEM
# ======================================================

sample_student = {
    "absence_level": "High",
    "failures": 1,
    "studytime": 1
}

predicted_score = 7.5


recommendations = generate_recommendations(
    sample_student,
    predicted_score
)


print("----- PREDICTED SCORE -----")
print(predicted_score)


print("\n----- PERSONALIZED RECOMMENDATIONS -----")

for recommendation in recommendations:
    print("-", recommendation)