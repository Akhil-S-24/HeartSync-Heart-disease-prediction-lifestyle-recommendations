from flask import Flask, render_template, request
import numpy as np

# ML imports
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

app = Flask(__name__)


def train_synthetic_model(random_state: int = 42) -> Pipeline:

    X, y = make_classification(
        n_samples=4000,
        n_features=13,
        n_informative=10,
        n_redundant=2,
        n_repeated=1,
        n_classes=2,
        class_sep=1.2,
        random_state=random_state,
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=200, solver="lbfgs")),
        ]
    )
    model.fit(X, y)
    return model


model = train_synthetic_model()


def parse_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def parse_int(value: str, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


@app.route("/")
def index():
    return render_template("index.html")


def build_recommendations(form: dict, risk_prob: float) -> list:
    recs = []

    age = parse_int(form.get("age", "0"))
    chol = parse_int(form.get("chol", "0"))
    trestbps = parse_int(form.get("trestbps", "0"))
    thalach = parse_int(form.get("thalach", "0"))
    oldpeak = parse_float(form.get("oldpeak", "0"))
    bmi = parse_float(form.get("bmi", "0"))

    smoker = form.get("smoker", "no")
    alcohol = form.get("alcohol", "none")
    activity = form.get("activity", "low")
    diet = form.get("diet", "average")

    if risk_prob >= 0.5:
        recs.append("Consult a cardiologist for a full risk assessment.")

    if smoker == "yes":
        recs.append("Stop smoking; seek a cessation program and nicotine replacement if needed.")

    if alcohol in {"moderate", "heavy"}:
        recs.append("Reduce alcohol intake; aim for no more than 1 drink/day (women) or 2 (men).")

    if activity == "low":
        recs.append("Increase physical activity to 150 minutes/week of moderate-intensity exercise.")

    if diet in {"poor", "average"}:
        recs.append("Adopt a Mediterranean-style diet: more vegetables, fruits, whole grains, legumes, fish.")

    if bmi >= 27:
        recs.append("Work towards a 5–10% weight loss over 3–6 months through diet and activity.")

    if chol >= 200:
        recs.append("Limit saturated fats and trans fats; consider lipid panel follow-up.")

    if trestbps >= 130 or oldpeak >= 1.0:
        recs.append("Track blood pressure; reduce sodium intake and manage stress.")

    if thalach < 120 and activity != "high":
        recs.append("Gradually build cardio fitness with brisk walks, cycling, or swimming.")

    if age >= 45:
        recs.append("Schedule annual checkups including blood pressure, lipids, and glucose.")

    if not recs:
        recs.append("Maintain current healthy habits and continue regular checkups.")

    return recs


@app.route("/predict", methods=["POST"])
def predict():
    # Model features (13): using common heart dataset-inspired fields
    feature_names = [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
    ]

    values = []
    for name in feature_names:
        raw = request.form.get(name, "0")
        if name in {"oldpeak"}:
            values.append(parse_float(raw))
        else:
            values.append(parse_int(raw))

    X = np.array(values, dtype=float).reshape(1, -1)
    prob = float(model.predict_proba(X)[0, 1])
    label = "High Risk" if prob >= 0.5 else "Low Risk"

    recommendations = build_recommendations(request.form, prob)

    return render_template(
        "result.html",
        probability=round(prob * 100, 2),
        label=label,
        recommendations=recommendations,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)


