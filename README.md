# Student Performance Predictor

A beginner-friendly Machine Learning web application that predicts a student's expected final exam score based on their study habits, attendance, and other academic factors.

Built with **Python**, **Streamlit**, **Scikit-learn**, **Pandas**, **NumPy**, and **Plotly**.

---

## Features

| Feature | Description |
|---|---|
| Smart Prediction | Enter study details and receive an instant predicted score |
| Performance Category | Automatic classification: Excellent / Good / Average / Needs Improvement |
| Score Gauge | Visual gauge chart showing the predicted score |
| Data Visualisation | Interactive scatter plots, histograms, pie charts, and a correlation heatmap |
| Feature Importance | Bar chart of Ridge Regression coefficients |
| Personalised Tips | Actionable study recommendations based on your inputs |
| Model Metrics | MAE, RMSE, and R² score displayed on the Model Info page |
| Input Validation | Friendly error messages for invalid inputs |

---

## Technologies Used

- **Python 3.8+** — core language
- **Streamlit** — web application framework
- **Scikit-learn** — Ridge Regression model
- **Pandas** — data loading and manipulation
- **NumPy** — numerical operations
- **Plotly** — interactive charts

---

## Project Structure

```
student-performance-predictor/
│
├── app.py                    # Main Streamlit application
├── model.py                  # ML model: training, saving, prediction helpers
├── student_performance.csv   # Sample dataset (500 students)
├── generate_dataset.py       # Script to regenerate the dataset
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── models/
    └── student_model.pkl     # Saved Ridge Regression model (auto-created)
```

---

## Installation

### 1. Clone or download the project

```bash
git clone <your-repo-url>
cd student-performance-predictor
```

### 2. (Recommended) Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## How to Run

### Step 1 — Train the model (first time only)

```bash
python model.py
```

This will:
- Load `student_performance.csv`
- Train a Ridge Regression model
- Print evaluation metrics (MAE, RMSE, R²)
- Save the model to `models/student_model.pkl`

> **Note:** If you skip this step the app will train the model automatically on first launch.

### Step 2 — Launch the web application

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## How the ML Model Works

### Algorithm: Ridge Regression

Ridge Regression is a form of Linear Regression with L2 regularisation. It works by finding a straight-line (hyper-plane) relationship between the input features and the target score.

**Why Ridge instead of plain Linear Regression?**
- Prevents overfitting when features are correlated (e.g. study hours and study days are related)
- The regularisation penalty (alpha = 1.0) shrinks large coefficients, making the model more robust
- Still fully interpretable — each coefficient directly shows the effect of that feature

### Input Features

| Feature | Typical Range | Description |
|---|---|---|
| study_hours | 1 – 10 | Hours studied per day |
| attendance | 40 – 100 | Attendance percentage |
| previous_score | 30 – 100 | Previous exam score |
| assignment_score | 30 – 100 | Average assignment score |
| study_days | 1 – 7 | Days per week dedicated to studying |
| sleep_hours | 4 – 10 | Hours of sleep per night |
| extracurricular | 0 or 1 | Participates in activities (1 = Yes) |
| internet_access | 0 or 1 | Has internet at home (1 = Yes) |

### Training Process

1. Load 500 student records from `student_performance.csv`
2. Split into 80% training / 20% test sets (random seed = 42)
3. Scale all features using `StandardScaler` (zero mean, unit variance)
4. Train `Ridge(alpha=1.0)` on the training set
5. Evaluate on the test set → MAE ≈ 3.5, R² ≈ 0.68
6. Save `(model, scaler)` tuple to `models/student_model.pkl`

### Performance Categories

| Score Range | Category |
|---|---|
| 80 – 100 | Excellent |
| 65 – 79 | Good |
| 50 – 64 | Average |
| 0 – 49 | Needs Improvement |

---

## Example Prediction

| Input | Value |
|---|---|
| Study Hours/Day | 6.0 |
| Attendance | 90% |
| Previous Score | 72 |
| Assignment Score | 75 |
| Study Days/Week | 6 |
| Sleep Hours | 7.5 |
| Extracurricular | Yes |
| Internet Access | Yes |

**Predicted Score: ~78 → Category: Good**

---

## Regenerating the Dataset

If you want a fresh dataset:

```bash
python generate_dataset.py
```

This creates a new `student_performance.csv` with 500 synthetic student records using NumPy's random generator (seed = 42 for reproducibility).

---

## Future Improvements

- [ ] Add support for uploading a custom CSV dataset
- [ ] Compare multiple algorithms (Linear, Ridge, Lasso, Gradient Boosting)
- [ ] Add a student history tracker to monitor progress over time
- [ ] Export prediction results as a PDF report
- [ ] Add multi-language support
- [ ] Deploy to Streamlit Community Cloud

---

## Notes

- All data is synthetic and generated for educational purposes only
- No paid APIs or external services are required — works fully offline
- The model retrains automatically if `models/student_model.pkl` is missing

---

*College Minor Project — Python + Machine Learning*
