from pathlib import Path
from datetime import datetime
import random
import joblib
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

HERE = Path(__file__).parent
MODEL_PATH = HERE / "icu_demo_model.joblib"
HISTORY_PATH = HERE / "icu_live_demo_history.csv"
FEATURES = ["heart_rate_bpm", "temperature_c", "spo2_percent", "respiration_rate_bpm"]

st.set_page_config(page_title="ICU Patient Monitor", page_icon="🩺", layout="wide")
st.title("🩺 ICU Wearable Patient-Monitoring Dashboard")
st.caption("Question 3(b) • Educational prototype • Enter values manually or simulate readings")

if not MODEL_PATH.exists():
    st.error("Trained model not found. Open the project folder in the terminal and run: python train_model.py")
    st.stop()

model = joblib.load(MODEL_PATH)

def rule_alerts(row, limits):
    alerts = []
    if row["heart_rate_bpm"] < limits["hr_low"] or row["heart_rate_bpm"] > limits["hr_high"]:
        alerts.append("Heart rate is outside the configured demo range")
    if row["temperature_c"] < limits["temp_low"] or row["temperature_c"] >= limits["temp_high"]:
        alerts.append("Temperature is outside the configured demo range")
    if row["spo2_percent"] < limits["spo2_low"]:
        alerts.append("SpO₂ is below the configured demo limit")
    if row["respiration_rate_bpm"] < limits["rr_low"] or row["respiration_rate_bpm"] > limits["rr_high"]:
        alerts.append("Respiration rate is outside the configured demo range")
    return alerts

def load_history():
    if HISTORY_PATH.exists():
        try:
            return pd.read_csv(HISTORY_PATH)
        except Exception:
            pass
    return pd.DataFrame(columns=["timestamp"] + FEATURES + ["source", "model_result", "rule_alerts"])

def predict_and_save(values, source, limits):
    row = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        **values,
        "source": source,
    }
    x = pd.DataFrame([{feature: row[feature] for feature in FEATURES}])
    prediction = int(model.predict(x)[0])
    row["model_result"] = "Demo alert" if prediction == 1 else "Demo within limits"
    row["rule_alerts"] = "; ".join(rule_alerts(row, limits)) or "None"
    history = pd.concat([load_history(), pd.DataFrame([row])], ignore_index=True)
    history.to_csv(HISTORY_PATH, index=False)
    return row

with st.sidebar:
    st.header("Patient profile")
    patient_name = st.text_input("Patient name / ID", value="Demo ICU Patient")
    st.caption("Use fictional data only.")
    st.divider()
    st.header("Demo alert limits")
    st.caption("These configurable limits are for software demonstration only.")
    limits = {
        "hr_low": st.number_input("Heart rate lower (BPM)", min_value=20, max_value=100, value=50),
        "hr_high": st.number_input("Heart rate upper (BPM)", min_value=60, max_value=220, value=120),
        "temp_low": st.number_input("Temperature lower (°C)", min_value=30.0, max_value=38.0, value=35.5, step=0.1),
        "temp_high": st.number_input("Temperature upper (°C)", min_value=35.0, max_value=43.0, value=38.0, step=0.1),
        "spo2_low": st.number_input("SpO₂ lower (%)", min_value=70, max_value=100, value=92),
        "rr_low": st.number_input("Respiration lower (/min)", min_value=2, max_value=20, value=8),
        "rr_high": st.number_input("Respiration upper (/min)", min_value=10, max_value=60, value=25),
    }

st.subheader("Enter patient vital signs")
st.write("Enter the values below, then click **Analyze readings** to get the model output and threshold alerts.")

with st.form("manual_vitals_form"):
    c1, c2 = st.columns(2)
    with c1:
        heart_rate = st.number_input("Heart rate (BPM)", min_value=20.0, max_value=250.0, value=82.0, step=1.0)
        temperature = st.number_input("Body temperature (°C)", min_value=30.0, max_value=43.0, value=36.9, step=0.1)
    with c2:
        spo2 = st.number_input("Oxygen saturation, SpO₂ (%)", min_value=50.0, max_value=100.0, value=97.0, step=1.0)
        respiration = st.number_input("Respiration rate (breaths/min)", min_value=1.0, max_value=70.0, value=17.0, step=1.0)
    submitted = st.form_submit_button("Analyze readings", type="primary", use_container_width=True)

if submitted:
    values = {
        "heart_rate_bpm": float(heart_rate),
        "temperature_c": float(temperature),
        "spo2_percent": float(spo2),
        "respiration_rate_bpm": float(respiration),
    }
    result = predict_and_save(values, "Manually entered", limits)
    st.session_state["latest_manual_result"] = result

if "latest_manual_result" in st.session_state:
    result = st.session_state["latest_manual_result"]
    st.divider()
    st.subheader("Analysis result")
    st.caption(f"Patient: {patient_name} • Recorded: {result['timestamp']}")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Heart rate", f"{result['heart_rate_bpm']:.0f} BPM")
    m2.metric("Temperature", f"{result['temperature_c']:.1f} °C")
    m3.metric("SpO₂", f"{result['spo2_percent']:.0f}%")
    m4.metric("Respiration", f"{result['respiration_rate_bpm']:.0f}/min")

    if result["rule_alerts"] != "None":
        st.error("THRESHOLD ALERT: " + result["rule_alerts"])
    else:
        st.success("No configured threshold alerts for these entered values.")

    if result["model_result"] == "Demo alert":
        st.warning("Machine-learning output: Demo alert")
    else:
        st.success("Machine-learning output: Demo within limits")

    st.info("The ML output is from a model trained on synthetic demonstration data. It is not a medical diagnosis. Threshold alerts and model predictions are separate.")

st.divider()
with st.expander("Try a simulated example instead"):
    st.write("Use this to test the dashboard without entering values manually.")
    if st.button("Generate and analyze random reading"):
        vals = {
            "heart_rate_bpm": round(random.gauss(82, 16), 1),
            "temperature_c": round(random.gauss(36.9, 0.5), 2),
            "spo2_percent": round(min(100, max(80, random.gauss(96, 3))), 1),
            "respiration_rate_bpm": round(min(40, max(5, random.gauss(17, 4))), 1),
        }
        st.session_state["latest_manual_result"] = predict_and_save(vals, "Simulated", limits)
        st.rerun()
    if st.button("Add high-risk demo example"):
        vals = {
            "heart_rate_bpm": 135.0,
            "temperature_c": 38.5,
            "spo2_percent": 88.0,
            "respiration_rate_bpm": 30.0,
        }
        st.session_state["latest_manual_result"] = predict_and_save(vals, "Simulated alert example", limits)
        st.rerun()

history = load_history()
if not history.empty:
    st.subheader("Reading history")
    chart_data = history.copy()
    chart_data["timestamp"] = pd.to_datetime(chart_data["timestamp"], errors="coerce")
    chart_data = chart_data.dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp")
    t1, t2, t3, t4 = st.tabs(["Heart rate", "Temperature", "SpO₂", "Respiration"])
    with t1:
        st.line_chart(chart_data[["heart_rate_bpm"]], y_label="BPM")
    with t2:
        st.line_chart(chart_data[["temperature_c"]], y_label="°C")
    with t3:
        st.line_chart(chart_data[["spo2_percent"]], y_label="%")
    with t4:
        st.line_chart(chart_data[["respiration_rate_bpm"]], y_label="Breaths/min")
    st.dataframe(history.iloc[::-1], use_container_width=True, hide_index=True)
    st.download_button("Download history CSV", history.to_csv(index=False).encode("utf-8"),
                       "icu_demo_history.csv", "text/csv")
    if st.button("Clear reading history"):
        if HISTORY_PATH.exists():
            HISTORY_PATH.unlink()
        st.session_state.pop("latest_manual_result", None)
        st.rerun()

st.caption("Safety notice: This is an educational software prototype using synthetic training data. It is not validated for real ICU monitoring, diagnosis, treatment, or emergency alerts.")
