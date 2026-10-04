# ICU Wearable Monitoring ML Prototype — Question 3(b)

## Setup (Windows / VS Code)
Open this folder in VS Code, then open Terminal > New Terminal.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Step 1 — Train the model
```powershell
python train_model.py
```
This creates `icu_training_demo.csv` and `icu_demo_model.joblib`, and prints test metrics.

## Step 2 — Run the dashboard
```powershell
python -m streamlit run app.py
```
Open the local URL printed in the terminal, usually http://localhost:8501.

## Step 3 — Demonstrate
- Click **Generate one reading** several times.
- Click **Add alert example** to test the threshold alerts.
- Enable auto-generation in the sidebar to simulate readings every five seconds.
- Review charts/history and download the CSV.

## Dataset note
This starter trains on synthetic data so the workflow works immediately. Its labels are generated from simplified demonstration thresholds, so model metrics do not indicate clinical accuracy. It is not safe for actual patient monitoring.

For a research extension, investigate:
- PhysioNet BIDMC PPG and Respiration: https://physionet.org/content/bidmc/
- MIMIC-IV: https://physionet.org/content/mimiciv/
- MIMIC-IV Waveform Database: https://physionet.org/content/mimic-iv-waveform/

These sources have different formats and access requirements. BIDMC has useful physiological measurements but not all four required features in a ready-made aligned table. MIMIC-IV access requires credentialing/training and a data-use agreement. Inspect the documentation and do not invent missing measurements or labels.

A real-data ML model needs documented labels from an appropriate source and patient-wise train/test splitting. Do not claim a model trained on synthetic labels diagnoses illness.

## Safety
Educational demonstration only. Not a medical device, not clinically validated, and not for real patient care. Real ICU deployment requires validated sensors, clinical validation, privacy/security controls and regulatory review.
