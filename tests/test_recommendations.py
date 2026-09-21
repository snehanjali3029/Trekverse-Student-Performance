from src.recommendations import generate_recommendations


def test_high_absence_recommendation():
    student_data = {
        "absence_level": "High",
        "failures": 0,
        "studytime": 3
    }

    recommendations = generate_recommendations(
        student_data,
        predicted_score=10
    )

    assert any("attendance" in recommendation.lower()
               for recommendation in recommendations)


def test_previous_failure_recommendation():
    student_data = {
        "absence_level": "Low",
        "failures": 1,
        "studytime": 3
    }

    recommendations = generate_recommendations(
        student_data,
        predicted_score=10
    )

    assert any("previously unsuccessful" in recommendation.lower()
               for recommendation in recommendations)


def test_low_studytime_recommendation():
    student_data = {
        "absence_level": "Low",
        "failures": 0,
        "studytime": 1
    }

    recommendations = generate_recommendations(
        student_data,
        predicted_score=10
    )

    assert any("study time" in recommendation.lower()
               for recommendation in recommendations)


def test_low_score_recommendation():
    student_data = {
        "absence_level": "Low",
        "failures": 0,
        "studytime": 3
    }

    recommendations = generate_recommendations(
        student_data,
        predicted_score=6
    )

    assert any("academic support" in recommendation.lower()
               for recommendation in recommendations)