from pathlib import Path
import pickle
import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent
MODEL = BASE/"placement_model.pkl"
SCHEMA = BASE/"schema.pkl"
METRICS = BASE/"metrics.csv"

st.set_page_config(page_title="Student Placement Predictor", page_icon="🎓", layout="wide")

if not MODEL.exists() or not SCHEMA.exists():
    st.error("Model is not trained yet. Close this page and double-click START_PROJECT.bat.")
    st.stop()

with open(MODEL, "rb") as f: model = pickle.load(f)
with open(SCHEMA, "rb") as f: schema = pickle.load(f)

st.title("🎓 Student Placement Predictor")
st.caption("Placement Prediction & Skill Gap Analyzer")
st.info("Educational portfolio project using a public placement dataset. The result is a model prediction, not a guarantee of placement.")

cols = set(schema["columns"])
x = {}

def num(label, names, default, lo, hi, step):
    c = next((n for n in names if n in cols), None)
    if c:
        x[c] = st.number_input(label, min_value=lo, max_value=hi, value=default, step=step)

def yn(label, names):
    c = next((n for n in names if n in cols), None)
    if c:
        x[c] = st.selectbox(label, ["Yes","No"])

left, right = st.columns(2)
with left:
    st.subheader("📚 Academic Profile")
    num("CGPA", ["CGPA"], 8.0, 6.0, 10.0, 0.1)
    num("Aptitude Test Score", ["AptitudeTestScore","ApptitudeTestScore"], 75, 0, 100, 1)
    num("SSC Marks", ["SSC_Marks","SSC"], 75, 0, 100, 1)
    num("HSC Marks", ["HSC_Marks","HSC"], 75, 0, 100, 1)

with right:
    st.subheader("💻 Skills & Experience")
    num("Internships", ["Internships"], 1, 0, 5, 1)
    num("Projects", ["Projects"], 2, 0, 10, 1)
    num("Workshops / Certifications", ["Workshops/Certifications","WorkshopsCertifications"], 1, 0, 10, 1)
    num("Soft Skill Rating", ["SoftSkillRating","SoftSkillsRating","SoftSkillrating"], 4.3, 0.0, 5.0, 0.1)  
    yn("Placement Training", ["PlacementTraining"])
    yn("Extra-curricular Activities", ["ExtracurricularActivities","ExtraCurricularActivities"])

# StudentID is excluded during training; fill any unexpected remaining feature safely.
for c in schema["columns"]:
    if c not in x:
        x[c] = 0 if c in schema["numeric_features"] else "No"

if st.button("🚀 Predict Placement", type="primary", use_container_width=True):
    row = pd.DataFrame([x], columns=schema["columns"])
    pred = int(model.predict(row)[0])
    prob = float(model.predict_proba(row)[0,1])

    a,b,c = st.columns(3)
    a.metric("Prediction", "PLACED" if pred else "NOT PLACED")
    b.metric("Model Probability", f"{prob*100:.1f}%")
    c.metric("Readiness", "High" if prob >= .75 else "Moderate" if prob >= .50 else "Needs Improvement")

    if pred:
        st.success("The model predicts the positive placement class.")
    else:
        st.warning("The model predicts the negative placement class.")

    st.subheader("🔎 Skill Gap Analyzer")
    gaps=[]
    if x.get("CGPA",10) < 7.5: gaps.append("Academic performance: focus on maintaining a stronger CGPA.")
    apt=x.get("AptitudeTestScore", x.get("ApptitudeTestScore",100))
    if apt < 65: gaps.append("Aptitude: practice quantitative and logical reasoning.")
    soft=x.get("SoftSkillRating", x.get("SoftSkillsRating", x.get("SoftSkillrating",5)))
    if soft < 3.5: gaps.append("Communication: practice speaking, presentations and interviews.")
    if x.get("Projects",3) < 2: gaps.append("Projects: build at least 2 strong, documented projects.")
    if x.get("Internships",1) < 1: gaps.append("Experience: seek internship or practical experience.")
    if gaps:
        for g in gaps: st.write("• "+g)
    else:
        st.success("No major gaps detected by the project's recommendation rules.")

st.divider()
st.subheader("📊 Model Evaluation")
if METRICS.exists():
    m=pd.read_csv(METRICS)
    st.dataframe(m.style.format({"accuracy":"{:.2%}","precision":"{:.2%}","recall":"{:.2%}","f1":"{:.2%}"}), use_container_width=True, hide_index=True)
