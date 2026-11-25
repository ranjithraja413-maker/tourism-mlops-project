"""
Streamlit app - Wellness Tourism Package purchase prediction
Loads the model that the GitHub Actions pipeline committed to
tourism_project/deployment/, collects customer details, puts them into a
dataframe and predicts whether the customer will buy the package.
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "best_tourism_model_v1.joblib"
META_PATH = APP_DIR / "model_metrics.json"

# Same ordinal encoding as tourism_project/model_building/prep.py
PRODUCT_MAP = {"Basic": 0, "Standard": 1, "Deluxe": 2, "Super Deluxe": 3, "King": 4}
DESIGNATION_MAP = {"Executive": 0, "Manager": 1, "Senior Manager": 2, "AVP": 3, "VP": 4}

st.set_page_config(page_title="Wellness Package Predictor", page_icon="🧳", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def load_meta():
    if META_PATH.exists():
        return json.loads(META_PATH.read_text())
    return {"threshold": 0.5}


if not MODEL_PATH.exists():
    st.error("Model file not found. Run the GitHub Actions pipeline so it commits "
             "`best_tourism_model_v1.joblib` to tourism_project/deployment/.")
    st.stop()

model = load_model()
meta = load_meta()
threshold = float(meta.get("threshold", 0.5))

st.title("🧳 Visit with Us — Wellness Tourism Package Predictor")
st.write("Enter the customer's details and the sales-interaction data to estimate "
         "whether the customer will purchase the **Wellness Tourism Package** before "
         "the sales team contacts them.")

# ------------------------------------------------------------------ inputs
st.subheader("Customer details")
c1, c2, c3 = st.columns(3)
with c1:
    age = st.number_input("Age", min_value=18, max_value=80, value=35)
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital = st.selectbox("Marital status", ["Married", "Single", "Divorced"])
    occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Freelancer"])
    designation = st.selectbox("Designation", list(DESIGNATION_MAP))
with c2:
    income = st.number_input("Monthly income", min_value=1000, max_value=100000, value=23000, step=500)
    city_tier = st.selectbox("City tier", [1, 2, 3])
    passport = st.selectbox("Has a valid passport?", ["No", "Yes"])
    own_car = st.selectbox("Owns a car?", ["No", "Yes"])
    trips = st.number_input("Average trips per year", min_value=1, max_value=25, value=3)
with c3:
    persons = st.number_input("Number of persons visiting", min_value=1, max_value=10, value=3)
    children = st.number_input("Children (<5 yrs) visiting", min_value=0, max_value=5, value=1)
    star = st.selectbox("Preferred property star", [3, 4, 5])

st.subheader("Sales interaction")
d1, d2, d3 = st.columns(3)
with d1:
    contact = st.selectbox("Type of contact", ["Self Enquiry", "Company Invited"])
    product = st.selectbox("Product pitched", list(PRODUCT_MAP))
with d2:
    duration = st.number_input("Duration of pitch (minutes)", min_value=1, max_value=130, value=15)
    followups = st.number_input("Number of follow-ups", min_value=0, max_value=10, value=4)
with d3:
    pitch_score = st.slider("Pitch satisfaction score", 1, 5, 3)

# Save the inputs into a dataframe with the same columns the model was trained on
input_df = pd.DataFrame([{
    "Age": age,
    "TypeofContact": contact,
    "CityTier": city_tier,
    "DurationOfPitch": duration,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": persons,
    "NumberOfFollowups": followups,
    "ProductPitched": PRODUCT_MAP[product],
    "PreferredPropertyStar": star,
    "MaritalStatus": marital,
    "NumberOfTrips": trips,
    "Passport": 1 if passport == "Yes" else 0,
    "PitchSatisfactionScore": pitch_score,
    "OwnCar": 1 if own_car == "Yes" else 0,
    "NumberOfChildrenVisiting": children,
    "Designation": DESIGNATION_MAP[designation],
    "MonthlyIncome": income,
}])

with st.expander("Input sent to the model"):
    st.dataframe(input_df, use_container_width=True)

# ------------------------------------------------------------------ predict
if st.button("Predict", type="primary"):
    prob = float(model.predict_proba(input_df)[0, 1])
    if prob >= threshold:
        st.success(f"✅ Likely to **purchase** the Wellness Package (probability {prob:.1%}). "
                   "Prioritise this customer for outreach.")
    else:
        st.warning(f"❌ Unlikely to purchase (probability {prob:.1%}). "
                   "Consider a lower-cost touchpoint instead of a full sales call.")
    st.progress(min(max(prob, 0.0), 1.0))

# ------------------------------------------------------------------ model card
with st.sidebar:
    st.header("Model")
    st.write(meta.get("algorithm", "XGBoost pipeline"))
    m = meta.get("metrics", {})
    if m:
        st.metric("Test F1", f"{m.get('test_f1', 0):.3f}")
        st.metric("Test recall", f"{m.get('test_recall', 0):.3f}")
        st.metric("Test ROC-AUC", f"{m.get('test_roc_auc', 0):.3f}")
    if "trained_at_utc" in meta:
        st.caption(f"Trained by CI pipeline: {meta['trained_at_utc']} UTC")
    st.caption(f"Decision threshold: {threshold}")
