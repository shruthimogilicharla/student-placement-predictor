import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="Student Placement Predictor",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Placement Predictor")
st.caption("Placement Prediction & Skill Gap Analyzer")

DATA = "placementdata.csv"

try:
    df = pd.read_csv(DATA)
except Exception as e:
    st.error(f"Could not load dataset: {e}")
    st.stop()

df.columns = [str(c).strip() for c in df.columns]

target = "PlacementStatus"

if target not in df.columns:
    st.error(f"'{target}' column not found.")
    st.write("Available columns:", list(df.columns))
    st.stop()

# Prepare target
y = (
    df[target]
    .astype(str)
    .str.strip()
    .map({
        "Placed": 1,
        "NotPlaced": 0,
        "Not Placed": 0,
        "Yes": 1,
        "No": 0
    })
)

valid = y.notna()
df = df.loc[valid].copy()
y = y.loc[valid].astype(int)

# Remove ID columns
drop_cols = [
    "StudentID",
    "StudentId",
    "student_id",
    "sl_no",
    "sl.no"
]

df = df.drop(
    columns=[c for c in drop_cols if c in df.columns],
    errors="ignore"
)

X = df.drop(columns=[target])

numeric_features = X.select_dtypes(
    include="number"
).columns.tolist()

categorical_features = [
    c for c in X.columns
    if c not in numeric_features
]

preprocessor = ColumnTransformer([
    (
        "num",
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]),
        numeric_features
    ),
    (
        "cat",
        Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]),
        categorical_features
    )
])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "classifier",
        LogisticRegression(
            max_iter=2000,
            random_state=42
        )
    )
])

model.fit(X_train, y_train)

accuracy = accuracy_score(
    y_test,
    model.predict(X_test)
)

st.success("✅ Machine Learning model is ready!")

st.subheader("📚 Academic Profile")

left, right = st.columns(2)

inputs = {}

with left:
    st.write("### Academic Details")

    if "CGPA" in X.columns:
        inputs["CGPA"] = st.number_input(
            "CGPA",
            min_value=0.0,
            max_value=10.0,
            value=8.0,
            step=0.1
        )

    if "AptitudeTestScore" in X.columns:
        inputs["AptitudeTestScore"] = st.number_input(
            "Aptitude Test Score",
            min_value=0.0,
            max_value=100.0,
            value=75.0,
            step=1.0
        )

    if "ApptitudeTestScore" in X.columns:
        inputs["ApptitudeTestScore"] = st.number_input(
            "Aptitude Test Score",
            min_value=0.0,
            max_value=100.0,
            value=75.0,
            step=1.0
        )

    if "SSC_Marks" in X.columns:
        inputs["SSC_Marks"] = st.number_input(
            "SSC Marks",
            min_value=0.0,
            max_value=100.0,
            value=75.0,
            step=1.0
        )

    if "HSC_Marks" in X.columns:
        inputs["HSC_Marks"] = st.number_input(
            "HSC Marks",
            min_value=0.0,
            max_value=100.0,
            value=75.0,
            step=1.0
        )

with right:
    st.write("### Skills & Experience")

    if "Internships" in X.columns:
        inputs["Internships"] = st.number_input(
            "Internships",
            min_value=0,
            max_value=10,
            value=1,
            step=1
        )

    if "Projects" in X.columns:
        inputs["Projects"] = st.number_input(
            "Projects",
            min_value=0,
            max_value=10,
            value=2,
            step=1
        )

    if "Workshops/Certifications" in X.columns:
        inputs["Workshops/Certifications"] = st.number_input(
            "Workshops / Certifications",
            min_value=0,
            max_value=10,
            value=1,
            step=1
        )

    if "WorkshopsCertifications" in X.columns:
        inputs["WorkshopsCertifications"] = st.number_input(
            "Workshops / Certifications",
            min_value=0,
            max_value=10,
            value=1,
            step=1
        )

    if "SoftSkillRating" in X.columns:
        inputs["SoftSkillRating"] = st.number_input(
            "Soft Skill Rating",
            min_value=0.0,
            max_value=5.0,
            value=4.3,
            step=0.1
        )

    if "SoftSkillsRating" in X.columns:
        inputs["SoftSkillsRating"] = st.number_input(
            "Soft Skill Rating",
            min_value=0.0,
            max_value=5.0,
            value=4.3,
            step=0.1
        )

    if "PlacementTraining" in X.columns:
        inputs["PlacementTraining"] = st.selectbox(
            "Placement Training",
            ["Yes", "No"]
        )

    if "ExtracurricularActivities" in X.columns:
        inputs["ExtracurricularActivities"] = st.selectbox(
            "Extra-curricular Activities",
            ["Yes", "No"]
        )

    if "ExtraCurricularActivities" in X.columns:
        inputs["ExtraCurricularActivities"] = st.selectbox(
            "Extra-curricular Activities",
            ["Yes", "No"]
        )

# Fill remaining dataset columns
for column in X.columns:
    if column not in inputs:
        if column in numeric_features:
            inputs[column] = 0.0
        else:
            inputs[column] = "No"

if st.button(
    "🚀 Predict Placement",
    type="primary",
    use_container_width=True
):

    row = pd.DataFrame(
        [inputs],
        columns=X.columns
    )

    prediction = int(model.predict(row)[0])

    probability = float(
        model.predict_proba(row)[0][1]
    )

    a, b, c = st.columns(3)

    a.metric(
        "Prediction",
        "PLACED" if prediction else "NOT PLACED"
    )

    b.metric(
        "Probability",
        f"{probability * 100:.1f}%"
    )

    readiness = (
        "High"
        if probability >= 0.75
        else "Moderate"
        if probability >= 0.50
        else "Needs Improvement"
    )

    c.metric(
        "Readiness",
        readiness
    )

    if prediction:
        st.success(
            "🎉 The model predicts the positive placement class."
        )
    else:
        st.warning(
            "The model predicts the negative placement class."
        )

    st.subheader("🔎 Skill Gap Analyzer")

    gaps = []

    if inputs.get("CGPA", 10) < 7.5:
        gaps.append(
            "Improve academic performance and maintain a stronger CGPA."
        )

    aptitude = inputs.get(
        "AptitudeTestScore",
        inputs.get("ApptitudeTestScore", 100)
    )

    if aptitude < 65:
        gaps.append(
            "Practice quantitative and logical reasoning."
        )

    soft = inputs.get(
        "SoftSkillRating",
        inputs.get("SoftSkillsRating", 5)
    )

    if soft < 3.5:
        gaps.append(
            "Improve communication, presentation and interview skills."
        )

    if inputs.get("Projects", 3) < 2:
        gaps.append(
            "Build at least two strong, documented projects."
        )

    if inputs.get("Internships", 1) < 1:
        gaps.append(
            "Gain practical experience through internships or projects."
        )

    if gaps:
        for gap in gaps:
            st.write("• " + gap)
    else:
        st.success(
            "✅ No major skill gaps detected by the recommendation rules."
        )

st.divider()

st.subheader("📊 Model Evaluation")

st.metric(
    "Test Accuracy",
    f"{accuracy * 100:.2f}%"
)

st.caption(
    "This is an educational machine-learning project. "
    "The prediction is not a guarantee of placement."
)
