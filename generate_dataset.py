"""
Utility script to regenerate student_performance.csv
Run: python generate_dataset.py
"""
import numpy as np
import pandas as pd

np.random.seed(42)
n = 500

study_hours       = np.round(np.random.uniform(1, 10, n), 1)
attendance        = np.round(np.random.uniform(40, 100, n), 1)
previous_score    = np.round(np.random.uniform(30, 100, n), 1)
assignment_score  = np.round(np.random.uniform(30, 100, n), 1)
study_days        = np.random.randint(1, 8, n)
sleep_hours       = np.round(np.random.uniform(4, 10, n), 1)
extracurricular   = np.random.choice([0, 1], n)   # 1=Yes, 0=No
internet_access   = np.random.choice([0, 1], n, p=[0.2, 0.8])

# Realistic score formula with controlled noise
final_score = (
    0.25 * study_hours * 10 +
    0.20 * attendance * 0.8 +
    0.25 * previous_score * 0.7 +
    0.15 * assignment_score * 0.6 +
    0.05 * study_days * 5 +
    0.03 * sleep_hours * 5 +
    0.04 * extracurricular * 8 +
    0.03 * internet_access * 6 +
    np.random.normal(0, 5, n)
)
final_score = np.clip(np.round(final_score, 1), 0, 100)

df = pd.DataFrame({
    "study_hours":      study_hours,
    "attendance":       attendance,
    "previous_score":   previous_score,
    "assignment_score": assignment_score,
    "study_days":       study_days,
    "sleep_hours":      sleep_hours,
    "extracurricular":  extracurricular,
    "internet_access":  internet_access,
    "final_score":      final_score,
})

df.to_csv("student_performance.csv", index=False)
print(f"Dataset saved: {len(df)} rows, columns: {list(df.columns)}")
