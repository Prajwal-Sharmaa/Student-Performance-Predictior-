"""
model.py — Student Performance Predictor
-----------------------------------------
Trains a Ridge Regression model on student_performance.csv,
evaluates it, and saves the trained model to models/student_model.pkl.

Run standalone:  python model.py
"""

import os
import pickle
import numpy as np
import pandas as pd

# Use only thread-safe, DLL-free sklearn modules (linear models + scaler)
os.environ.setdefault("OMP_NUM_THREADS", "1")

from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "student_performance.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "student_model.pkl")

# Feature columns — must match the input form in app.py
FEATURE_COLS = [
    "study_hours",       # hours per day
    "attendance",        # percentage 0–100
    "previous_score",    # previous exam score 0–100
    "assignment_score",  # assignment score 0–100
    "study_days",        # study days per week 1–7
    "sleep_hours",       # hours per day
    "extracurricular",   # 1 = Yes, 0 = No
    "internet_access",   # 1 = Yes, 0 = No
]
TARGET_COL = "final_score"

# Human-readable display names for charts
FEATURE_LABELS = {
    "study_hours":      "Study Hours/Day",
    "attendance":       "Attendance %",
    "previous_score":   "Previous Score",
    "assignment_score": "Assignment Score",
    "study_days":       "Study Days/Week",
    "sleep_hours":      "Sleep Hours/Day",
    "extracurricular":  "Extracurricular",
    "internet_access":  "Internet Access",
}


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the CSV dataset and validate required columns."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. "
            "Run generate_dataset.py to create it."
        )
    df = pd.read_csv(path)
    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")
    return df


def train_model(df: pd.DataFrame):
    """
    Train a Ridge Regression model.

    Returns
    -------
    model     : fitted Ridge model
    scaler    : fitted StandardScaler
    metrics   : dict with mae, rmse, r2
    coef_dict : dict of {feature: coefficient}
    """
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    # 80/20 train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale features — important for Ridge
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "mae":  round(float(mean_absolute_error(y_test, y_pred)), 2),
        "rmse": round(float(np.sqrt(np.mean((y_test - y_pred) ** 2))), 2),
        "r2":   round(float(r2_score(y_test, y_pred)), 4),
    }

    # Coefficients as feature importance proxy
    coef_dict = dict(zip(FEATURE_COLS, model.coef_.tolist()))

    return model, scaler, metrics, coef_dict


def save_model(model, scaler, path: str = MODEL_PATH):
    """Save the model + scaler together as a tuple."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump((model, scaler), f)
    print(f"Model saved to: {path}")


def load_model(path: str = MODEL_PATH):
    """
    Load the saved (model, scaler) tuple.
    Also trains a fresh model from the CSV if the pkl is missing.
    """
    if not os.path.exists(path):
        print("No saved model found — training a fresh one …")
        df = load_data()
        model, scaler, metrics, _ = train_model(df)
        save_model(model, scaler, path)
        return model, scaler
    with open(path, "rb") as f:
        return pickle.load(f)


def predict(model, scaler, input_dict: dict) -> float:
    """
    Predict a student's final score from a feature dictionary.
    Returns a float clamped to [0, 100].
    """
    row = np.array([[input_dict[col] for col in FEATURE_COLS]])
    row_scaled = scaler.transform(row)
    score = float(model.predict(row_scaled)[0])
    return round(max(0.0, min(100.0, score)), 1)


def get_performance_category(score: float) -> dict:
    """Map a score to a label, emoji, and hex colour."""
    if score >= 80:
        return {"label": "Excellent",          "emoji": "🌟", "colour": "#28a745"}
    elif score >= 65:
        return {"label": "Good",               "emoji": "👍", "colour": "#17a2b8"}
    elif score >= 50:
        return {"label": "Average",            "emoji": "📚", "colour": "#ffc107"}
    else:
        return {"label": "Needs Improvement",  "emoji": "⚠️",  "colour": "#dc3545"}


# ── Standalone execution ───────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading dataset ...")
    df = load_data()
    print(f"  Rows: {len(df)}  |  Columns: {list(df.columns)}")

    print("\nTraining Ridge Regression model ...")
    model, scaler, metrics, coef_dict = train_model(df)

    print("\n-- Evaluation Metrics --")
    print(f"  MAE  : {metrics['mae']}")
    print(f"  RMSE : {metrics['rmse']}")
    print(f"  R2   : {metrics['r2']}")

    print("\n-- Feature Coefficients --")
    for feat, coef in sorted(coef_dict.items(), key=lambda x: -abs(x[1])):
        print(f"  {FEATURE_LABELS[feat]:22s}: {coef:+.4f}")

    save_model(model, scaler)
    print("\nDone!")
