from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("model.pkl")
feature_columns = joblib.load("feature_columns.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    gender = request.form["gender"]
    age = float(request.form["age"])
    occupation = request.form["occupation"]
    sleep_duration = float(request.form["sleep_duration"])
    quality_sleep = float(request.form["quality_sleep"])
    physical_activity = float(request.form["physical_activity"])
    stress_level = float(request.form["stress_level"])
    bmi_category = request.form["bmi_category"]
    heart_rate = float(request.form["heart_rate"])
    daily_steps = float(request.form["daily_steps"])
    systolic = float(request.form["systolic"])
    diastolic = float(request.form["diastolic"])

    input_data = pd.DataFrame({
        "Gender": [gender],
        "Age": [age],
        "Occupation": [occupation],
        "Sleep Duration": [sleep_duration],
        "Quality of Sleep": [quality_sleep],
        "Physical Activity Level": [physical_activity],
        "Stress Level": [stress_level],
        "BMI Category": [bmi_category],
        "Heart Rate": [heart_rate],
        "Daily Steps": [daily_steps],
        "Systolic": [systolic],
        "Diastolic": [diastolic]
    })

    input_data = pd.get_dummies(input_data)

    input_data = input_data.reindex(
        columns=feature_columns,
        fill_value=0
    )

    prediction = model.predict(input_data)[0]

    print()
    print("--------------------------------")
    print("PREDICTED CONDITION:", prediction)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_data)[0]

        print("PREDICTION PROBABILITIES:")

        for condition, probability in zip(model.classes_, probabilities):
            print(condition, ":", round(probability * 100, 2), "%")

    print("--------------------------------")
    print()

    descriptions = {

        "None": {
            "title": "No Sleep Disorder Predicted",
            "description": "Your entered health and lifestyle factors do not indicate a major sleep disorder according to the model.",
            "solution": "Maintain a regular sleep schedule, get enough sleep, stay physically active, and manage stress."
        },

        "Insomnia": {
            "title": "Insomnia",
            "description": "Insomnia is a sleep problem where a person may have difficulty falling asleep, staying asleep, or getting enough restful sleep.",
            "solution": "Maintain a regular sleep schedule, avoid caffeine and screens close to bedtime, and keep your sleeping environment comfortable. If the problem continues, consult a healthcare professional."
        },

        "Sleep Apnea": {
            "title": "Sleep Apnea",
            "description": "Sleep apnea is a condition where breathing repeatedly stops and starts during sleep. It can cause poor-quality sleep and daytime tiredness.",
            "solution": "Maintain a healthy sleep schedule and discuss persistent symptoms such as loud snoring or breathing pauses with a healthcare professional."
        }
    }

    result = descriptions.get(prediction, {
        "title": prediction,
        "description": "The model has predicted this sleep condition based on the information provided.",
        "solution": "Maintain healthy sleep habits and consult a healthcare professional."
    })

    return render_template(
        "result.html",
        prediction=result["title"],
        description=result["description"],
        solution=result["solution"]
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
