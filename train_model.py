from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

HERE = Path(__file__).parent
DATA = HERE / "icu_training_demo.csv"
MODEL = HERE / "icu_demo_model.joblib"
FEATURES = ["heart_rate_bpm", "temperature_c", "spo2_percent", "respiration_rate_bpm"]

def make_demo_data(n=5000, seed=42):
    rng = np.random.default_rng(seed)
    hr = np.clip(rng.normal(82, 18, n), 35, 180)
    temp = np.clip(rng.normal(36.9, 0.7, n), 34, 41.5)
    spo2 = np.clip(rng.normal(96.5, 3, n), 75, 100)
    resp = np.clip(rng.normal(17, 5, n), 5, 45)
    # Synthetic teaching labels only; not clinical labels.
    label = ((hr < 50) | (hr > 120) | (temp < 35.5) | (temp >= 38) |
             (spo2 < 92) | (resp < 8) | (resp > 25)).astype(int)
    return pd.DataFrame({
        "heart_rate_bpm": hr.round(1),
        "temperature_c": temp.round(2),
        "spo2_percent": spo2.round(1),
        "respiration_rate_bpm": resp.round(1),
        "demo_label": label
    })

def main():
    if DATA.exists():
        df = pd.read_csv(DATA)
        missing = set(FEATURES + ["demo_label"]) - set(df.columns)
        if missing:
            raise ValueError(f"CSV is missing columns: {sorted(missing)}")
        print("Using existing CSV:", DATA.name)
    else:
        df = make_demo_data()
        df.to_csv(DATA, index=False)
        print("Created synthetic teaching dataset:", DATA.name)

    X, y = df[FEATURES], df["demo_label"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(n_estimators=200, random_state=42,
                               class_weight="balanced", min_samples_leaf=3)
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    joblib.dump(model, MODEL)
    print(f"Training rows: {len(X_train)} | Test rows: {len(X_test)}")
    print(f"Test accuracy: {accuracy_score(y_test, pred):.3f}")
    print(classification_report(y_test, pred, target_names=["Demo within limits", "Demo alert"]))
    print("Saved model:", MODEL.name)
    print("WARNING: synthetic labels only; this is not a clinically validated model.")

if __name__ == "__main__":
    main()
