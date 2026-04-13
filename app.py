from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

model = joblib.load("model/student_model.pkl")
scaler = joblib.load("model/scaler.pkl")

@app.route("/")
def home():
    return render_template("index.html")


def get_grade(cgpa):
    if cgpa >= 9:
        return "O (Outstanding)"
    elif cgpa >= 8:
        return "A+"
    elif cgpa >= 7:
        return "A"
    elif cgpa >= 6:
        return "B+"
    elif cgpa >= 5:
        return "B"
    else:
        return "C"


@app.route("/predict", methods=["POST"])
def predict():

    attendance = float(request.form["attendance"])
    study_hours = float(request.form["study_hours"])
    study_sessions = float(request.form["study_sessions"])
    previous_sgpa = float(request.form["previous_sgpa"])
    social_media = float(request.form["social_media"])
    consultancy = int(request.form["consultancy"])
    cocurricular = int(request.form["cocurricular"])
    backlog = int(request.form["backlog"])

    # ✅ FIX 1: First semester handling (realistic baseline)
    if previous_sgpa == 0:
        previous_sgpa = 5.5
        first_sem_message = "First semester student - prediction based on behavior"
    else:
        first_sem_message = "Prediction based on past performance and behavior"

    # ✅ Handle extreme values (but DO NOT override real input like attendance)
    study_hours = min(study_hours, 8)
    study_sessions = min(study_sessions, 6)

    # Prepare features
    features = np.array([[
        attendance,
        study_hours,
        study_sessions,
        previous_sgpa,
        social_media,
        consultancy,
        cocurricular
    ]])

    scaled = scaler.transform(features)

    # Model prediction
    predicted_sgpa = round(model.predict(scaled)[0], 2)

    # ✅ FIX 2: Behavior penalty logic (IMPORTANT)
    if attendance < 40:
        predicted_sgpa -= 1.5

    if study_hours < 1:
        predicted_sgpa -= 1.0

    if study_sessions <= 1:
        predicted_sgpa -= 0.5

    # Keep SGPA in valid range
    predicted_sgpa = max(0, min(10, predicted_sgpa))
    predicted_sgpa = round(predicted_sgpa, 2)

    # Backlog handling
    if backlog == 1:
        display_sgpa = predicted_sgpa
        display_cgpa = "Not Available (Backlog)"
        backlog_message = "Backlogs detected. CGPA cannot be calculated."
        grade = "N/A"
        performance = "Cannot evaluate due to backlog"
    else:
        display_sgpa = predicted_sgpa
        display_cgpa = predicted_sgpa
        backlog_message = "No backlogs. Keep performing well!"
        grade = get_grade(predicted_sgpa)

        # ✅ FIX 3: Improved performance logic
        if attendance < 40 or study_hours == 0:
            performance = "At Risk - Poor academic behavior"
        elif predicted_sgpa >= 8.5:
            performance = "Excellent performance!"
        elif predicted_sgpa >= 7:
            performance = "Good performance"
        else:
            performance = "Needs improvement"

    # Suggestions
    suggestions = []

    if backlog == 1:
        suggestions.append("Clear backlogs immediately")

    if attendance < 40:
        suggestions.append("Critical: Increase attendance immediately")

    elif attendance < 75:
        suggestions.append("Improve attendance")

    if study_hours == 0:
        suggestions.append("Start studying daily (minimum 2 hours)")
    elif study_hours < 3:
        suggestions.append("Increase study hours")

    if study_sessions <= 1:
        suggestions.append("Increase study sessions per week")

    if abs(study_hours - social_media) <= 1:
        suggestions.append("Balance study and social media time")

    if social_media > 4:
        suggestions.append("Reduce social media usage")

    if consultancy == 0:
        suggestions.append("Attend consultancy sessions")

    if cocurricular == 0:
        suggestions.append("Participate in co-curricular activities")

    if not suggestions:
        suggestions.append("Maintain current performance")

    return render_template(
        "result.html",
        sgpa=display_sgpa,
        cgpa=display_cgpa,
        grade=grade,
        performance=performance,
        backlog_message=backlog_message,
        suggestions=suggestions,
        first_sem_message=first_sem_message
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
