"""Quick smoke-test for model.py — run with: python test_model.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import model as ml

# 1. Load data
df = ml.load_data()
assert len(df) > 0, "Dataset is empty"
print("1. load_data OK — rows:", len(df))

# 2. Train
m, sc, metrics, coefs = ml.train_model(df)
assert metrics["r2"] > 0.3, "R2 suspiciously low"
print("2. train_model OK — R2:", metrics["r2"])

# 3. Save + load
ml.save_model(m, sc)
m2, sc2 = ml.load_model()
print("3. save/load_model OK")

# 4. Predict
inp = {
    "study_hours": 6.0, "attendance": 90.0, "previous_score": 72.0,
    "assignment_score": 75.0, "study_days": 6, "sleep_hours": 7.5,
    "extracurricular": 1, "internet_access": 1,
}
score = ml.predict(m2, sc2, inp)
assert 0 <= score <= 100, "Score out of range"
print("4. predict OK — score:", score)

# 5. Category
for s, expected in [(85, "Excellent"), (70, "Good"), (55, "Average"), (40, "Needs Improvement")]:
    cat = ml.get_performance_category(s)
    assert cat["label"] == expected, f"Wrong category for {s}: {cat['label']}"
print("5. get_performance_category OK")

print("\nAll tests PASSED")
