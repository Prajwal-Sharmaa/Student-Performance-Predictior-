"""
app.py — Student Performance Predictor
----------------------------------------
A beginner-friendly Streamlit web application that predicts a student's
expected final exam score using a Ridge Regression model.

Run with:  streamlit run app.py
"""

import os
import sys

# ── Ensure the project folder is on the path so model.py is importable ────────
APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

import model as ml   # model.py in the same folder

# ── Page configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Card-style containers */
.metric-card {
    background: #f7f9fc;
    border: 1px solid #e1e4e8;
    border-radius: 10px;
    padding: 20px 24px;
    margin-bottom: 12px;
}
.score-display {
    font-size: 3.5rem;
    font-weight: 800;
    text-align: center;
    margin: 8px 0;
}
.category-badge {
    font-size: 1.3rem;
    font-weight: 600;
    text-align: center;
    padding: 6px 18px;
    border-radius: 20px;
    display: inline-block;
}
.info-box {
    background: #eaf4ff;
    border-left: 4px solid #3b82d4;
    padding: 12px 16px;
    border-radius: 6px;
    margin-bottom: 10px;
}
.tip-box {
    background: #fff8e1;
    border-left: 4px solid #ffc107;
    padding: 12px 16px;
    border-radius: 6px;
    margin-bottom: 8px;
}
.success-box {
    background: #e8f5e9;
    border-left: 4px solid #28a745;
    padding: 12px 16px;
    border-radius: 6px;
    margin-bottom: 8px;
}
h1 { color: #1a1a2e; }
</style>
""", unsafe_allow_html=True)


# ── Cached helpers ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading ML model ...")
def get_model():
    """Load (or train + save) the Ridge Regression model once per session."""
    return ml.load_model()


@st.cache_data(show_spinner="Loading dataset ...")
def get_data():
    """Load the student dataset once per session."""
    return ml.load_data()


# ── Sidebar navigation ─────────────────────────────────────────────────────────

def sidebar():
    st.sidebar.image(
        "https://img.icons8.com/color/96/graduation-cap.png",
        width=80,
    )
    st.sidebar.title("Student Performance Predictor")
    st.sidebar.markdown("---")

    pages = {
        "🏠  Home":               "Home",
        "🔮  Predict Performance": "Predict",
        "📊  Data Visualisation":  "Visualise",
        "🤖  Model Info":          "Model",
    }
    choice = st.sidebar.radio("Navigate", list(pages.keys()), label_visibility="collapsed")

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<small style='color:#888;'>College Minor Project | Python + Streamlit</small>",
        unsafe_allow_html=True,
    )
    return pages[choice]


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ══════════════════════════════════════════════════════════════════════════════

def page_home():
    st.title("🎓 Student Performance Predictor")
    st.markdown(
        "**Predict your expected final exam score** based on your study habits, "
        "attendance, previous marks, and more — powered by Machine Learning."
    )
    st.markdown("---")

    # Feature highlight cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🔮 Smart Prediction</h3>
            <p>Enter your study details and get an instant predicted score
            along with a performance category.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Visual Insights</h3>
            <p>Explore interactive charts showing how different factors
            influence student performance.</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>💡 Personalised Tips</h3>
            <p>Receive actionable recommendations to help you improve
            based on your individual inputs.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("How It Works")
    steps_col1, steps_col2 = st.columns([1, 1])
    with steps_col1:
        st.markdown("""
        1. **Enter your details** in the *Predict Performance* page
        2. The ML model analyses your study patterns
        3. You receive a **predicted score** and category
        4. Personalised **study tips** are shown instantly
        """)
    with steps_col2:
        st.markdown("""
        **Factors considered:**
        - Study hours per day
        - Attendance percentage
        - Previous exam score
        - Assignment score
        - Study days per week
        - Sleep hours
        - Extracurricular activity
        - Internet access
        """)

    st.markdown("---")
    # Quick dataset stats
    try:
        df = get_data()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Dataset Size", f"{len(df)} students")
        c2.metric("Avg Final Score", f"{df['final_score'].mean():.1f}")
        c3.metric("Avg Study Hours", f"{df['study_hours'].mean():.1f} h/day")
        c4.metric("Avg Attendance", f"{df['attendance'].mean():.1f}%")
    except Exception:
        st.info("Dataset stats will appear once the CSV is loaded.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PREDICT
# ══════════════════════════════════════════════════════════════════════════════

def validate_inputs(study_hours, attendance, prev_score, assign_score,
                    study_days, sleep_hours):
    """Return a list of error strings; empty list means all inputs are valid."""
    errors = []
    if study_hours < 0:
        errors.append("Study hours cannot be negative.")
    if study_hours > 24:
        errors.append("Study hours per day cannot exceed 24.")
    if not (0 <= attendance <= 100):
        errors.append("Attendance must be between 0 and 100.")
    if not (0 <= prev_score <= 100):
        errors.append("Previous exam score must be between 0 and 100.")
    if not (0 <= assign_score <= 100):
        errors.append("Assignment score must be between 0 and 100.")
    if not (1 <= study_days <= 7):
        errors.append("Study days per week must be between 1 and 7.")
    if sleep_hours < 2:
        errors.append("Sleep hours seem too low. Please enter a realistic value (minimum 2).")
    if sleep_hours > 16:
        errors.append("Sleep hours seem too high. Please enter a realistic value (maximum 16).")
    return errors


def get_recommendations(study_hours, attendance, prev_score, assign_score,
                         study_days, sleep_hours, predicted_score):
    """Build a list of (tip_type, message) recommendation tuples."""
    tips = []

    if attendance < 75:
        tips.append(("warning",
            "Your attendance is below 75%. Regular attendance is strongly linked to "
            "better academic performance. Try to attend all classes consistently."))
    elif attendance >= 90:
        tips.append(("success",
            "Excellent attendance! Staying regular in class is one of the best habits "
            "for academic success."))

    if study_hours < 2:
        tips.append(("warning",
            "You are studying less than 2 hours per day. Try setting a daily study "
            "schedule with at least 2-3 focused hours."))
    elif study_hours >= 6:
        tips.append(("success",
            "Great effort on study hours! Make sure to take short breaks to keep "
            "your focus sharp."))

    if prev_score < 50:
        tips.append(("warning",
            "Your previous score was below 50. Identify weak topics, revisit past "
            "papers, and consider asking your teacher for extra help."))

    if assign_score < 60:
        tips.append(("warning",
            "Your assignment score is below 60. Completing assignments helps "
            "reinforce concepts — try not to skip them."))

    if sleep_hours < 6:
        tips.append(("warning",
            "You are sleeping fewer than 6 hours. Adequate sleep (7-9 hours) is "
            "essential for memory consolidation and concentration. "
            "Consider maintaining a regular sleep routine."))

    if study_days < 4:
        tips.append(("warning",
            "Studying only a few days a week can create large gaps. Aim for at "
            "least 5 consistent study days per week for steady progress."))

    if predicted_score >= 80:
        tips.append(("success",
            "Fantastic! Your predicted score is excellent. Keep up the good work "
            "and stay consistent!"))
    elif predicted_score >= 65:
        tips.append(("info",
            "Good job! A few more focused study sessions can push your score "
            "even higher."))
    elif predicted_score >= 50:
        tips.append(("info",
            "You are on track but there is room for improvement. Focus on the "
            "areas flagged above and you will see progress."))
    else:
        tips.append(("warning",
            "Your predicted score needs improvement. Don't be discouraged — small "
            "consistent changes in study habits can make a big difference."))

    return tips


def page_predict():
    st.title("🔮 Predict Student Performance")
    st.markdown("Fill in the details below and click **Predict Performance** to see your predicted score.")
    st.markdown("---")

    # ── Input form ────────────────────────────────────────────────────────────
    with st.form("prediction_form"):
        st.subheader("Student Information")
        student_name = st.text_input("Student Name", placeholder="e.g. Rahul Sharma")

        st.subheader("Study Habits")
        col1, col2 = st.columns(2)
        with col1:
            study_hours = st.number_input(
                "Study Hours per Day", min_value=0.0, max_value=24.0,
                value=4.0, step=0.5,
                help="How many hours do you study daily?"
            )
            attendance = st.number_input(
                "Attendance (%)", min_value=0.0, max_value=100.0,
                value=80.0, step=1.0,
                help="Your class attendance percentage"
            )
            prev_score = st.number_input(
                "Previous Exam Score (0-100)", min_value=0.0, max_value=100.0,
                value=65.0, step=1.0,
                help="Your score in the last exam"
            )
            assign_score = st.number_input(
                "Assignment Score (0-100)", min_value=0.0, max_value=100.0,
                value=70.0, step=1.0,
                help="Your average assignment score"
            )
        with col2:
            study_days = st.slider(
                "Study Days per Week", min_value=1, max_value=7, value=5,
                help="How many days per week do you study?"
            )
            sleep_hours = st.number_input(
                "Sleep Hours per Day", min_value=0.0, max_value=24.0,
                value=7.0, step=0.5,
                help="Average hours of sleep per night"
            )
            extracurricular = st.selectbox(
                "Extracurricular Activity", ["Yes", "No"],
                help="Do you participate in sports, clubs, or other activities?"
            )
            internet_access = st.selectbox(
                "Internet Access at Home", ["Yes", "No"],
                help="Do you have reliable internet access for studying?"
            )

        submitted = st.form_submit_button(
            "🔮 Predict Performance",
            use_container_width=True,
            type="primary",
        )

    # ── Process form submission ───────────────────────────────────────────────
    if submitted:
        # Validate
        errors = validate_inputs(
            study_hours, attendance, prev_score, assign_score, study_days, sleep_hours
        )
        if errors:
            for err in errors:
                st.error(f"❌ {err}")
            return

        # Build feature dict
        input_dict = {
            "study_hours":      float(study_hours),
            "attendance":       float(attendance),
            "previous_score":   float(prev_score),
            "assignment_score": float(assign_score),
            "study_days":       int(study_days),
            "sleep_hours":      float(sleep_hours),
            "extracurricular":  1 if extracurricular == "Yes" else 0,
            "internet_access":  1 if internet_access == "Yes" else 0,
        }

        # Predict
        try:
            model, scaler = get_model()
            predicted_score = ml.predict(model, scaler, input_dict)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            return

        category = ml.get_performance_category(predicted_score)

        st.markdown("---")
        # ── Result display ────────────────────────────────────────────────────
        name_label = student_name.strip() if student_name.strip() else "Student"
        st.subheader(f"Results for {name_label}")

        res_col1, res_col2, res_col3 = st.columns([1.2, 1, 1])

        with res_col1:
            st.markdown(
                f"""
                <div class="metric-card" style="text-align:center;">
                    <div style="font-size:1rem;color:#666;">Predicted Final Score</div>
                    <div class="score-display" style="color:{category['colour']};">{predicted_score}</div>
                    <div style="font-size:0.85rem;color:#999;">out of 100</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with res_col2:
            st.markdown(
                f"""
                <div class="metric-card" style="text-align:center;">
                    <div style="font-size:1rem;color:#666;">Performance Category</div>
                    <div style="font-size:2.5rem;margin:8px 0;">{category['emoji']}</div>
                    <div class="category-badge" style="background:{category['colour']}20;color:{category['colour']};">
                        {category['label']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with res_col3:
            # Score gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=predicted_score,
                domain={"x": [0, 1], "y": [0, 1]},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1},
                    "bar": {"color": category["colour"]},
                    "steps": [
                        {"range": [0, 50],  "color": "#fde8e8"},
                        {"range": [50, 65], "color": "#fff8e1"},
                        {"range": [65, 80], "color": "#e3f2fd"},
                        {"range": [80, 100],"color": "#e8f5e9"},
                    ],
                    "threshold": {
                        "line": {"color": "#333", "width": 3},
                        "thickness": 0.75,
                        "value": predicted_score,
                    },
                },
                title={"text": "Score Gauge"},
            ))
            fig_gauge.update_layout(height=200, margin=dict(t=30, b=10, l=10, r=10))
            st.plotly_chart(fig_gauge, use_container_width=True)

        # ── Score progress bar ────────────────────────────────────────────────
        st.markdown("**Score Progress**")
        st.progress(int(predicted_score))

        # ── Insights and recommendations ──────────────────────────────────────
        st.markdown("---")
        st.subheader("💡 Personalised Insights & Recommendations")

        tips = get_recommendations(
            study_hours, attendance, prev_score, assign_score,
            study_days, sleep_hours, predicted_score
        )

        for tip_type, message in tips:
            if tip_type == "success":
                st.markdown(
                    f'<div class="success-box">✅ {message}</div>',
                    unsafe_allow_html=True,
                )
            elif tip_type == "warning":
                st.markdown(
                    f'<div class="tip-box">💡 {message}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="info-box">ℹ️ {message}</div>',
                    unsafe_allow_html=True,
                )

        # ── Input summary ────────────────────────────────────────────────────
        st.markdown("---")
        with st.expander("View Input Summary"):
            summary_df = pd.DataFrame({
                "Feature": [
                    "Study Hours/Day", "Attendance %", "Previous Score",
                    "Assignment Score", "Study Days/Week", "Sleep Hours/Day",
                    "Extracurricular", "Internet Access",
                ],
                "Your Value": [
                    study_hours, f"{attendance}%", prev_score,
                    assign_score, study_days, sleep_hours,
                    extracurricular, internet_access,
                ],
            })
            st.dataframe(summary_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — DATA VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def page_visualise():
    st.title("📊 Data Visualisation")
    st.markdown("Explore how different factors relate to student performance in our dataset.")
    st.markdown("---")

    try:
        df = get_data()
    except Exception as e:
        st.error(f"Could not load dataset: {e}")
        return

    # Add a readable performance column
    df["Performance"] = df["final_score"].apply(
        lambda s: ml.get_performance_category(s)["label"]
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Study Hours vs Score",
        "Attendance vs Score",
        "Previous Score vs Final",
        "Score Distribution",
        "Feature Importance",
    ])

    with tab1:
        fig = px.scatter(
            df, x="study_hours", y="final_score",
            color="Performance",
            color_discrete_map={
                "Excellent": "#28a745", "Good": "#17a2b8",
                "Average": "#ffc107", "Needs Improvement": "#dc3545",
            },
            labels={"study_hours": "Study Hours per Day", "final_score": "Final Score"},
            title="Study Hours per Day vs Final Score",
            trendline="ols",
            opacity=0.7,
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Each dot represents one student. The trendline shows the general relationship.")

    with tab2:
        fig = px.scatter(
            df, x="attendance", y="final_score",
            color="Performance",
            color_discrete_map={
                "Excellent": "#28a745", "Good": "#17a2b8",
                "Average": "#ffc107", "Needs Improvement": "#dc3545",
            },
            labels={"attendance": "Attendance (%)", "final_score": "Final Score"},
            title="Attendance % vs Final Score",
            trendline="ols",
            opacity=0.7,
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        fig = px.scatter(
            df, x="previous_score", y="final_score",
            color="Performance",
            color_discrete_map={
                "Excellent": "#28a745", "Good": "#17a2b8",
                "Average": "#ffc107", "Needs Improvement": "#dc3545",
            },
            labels={"previous_score": "Previous Score", "final_score": "Final Score"},
            title="Previous Exam Score vs Final Score",
            trendline="ols",
            opacity=0.7,
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        col_a, col_b = st.columns(2)
        with col_a:
            fig_hist = px.histogram(
                df, x="final_score", nbins=20,
                labels={"final_score": "Final Score"},
                title="Distribution of Final Scores",
                color_discrete_sequence=["#3b82d4"],
            )
            fig_hist.update_layout(height=380)
            st.plotly_chart(fig_hist, use_container_width=True)
        with col_b:
            cat_counts = df["Performance"].value_counts().reset_index()
            cat_counts.columns = ["Category", "Count"]
            fig_pie = px.pie(
                cat_counts, names="Category", values="Count",
                title="Performance Category Breakdown",
                color="Category",
                color_discrete_map={
                    "Excellent": "#28a745", "Good": "#17a2b8",
                    "Average": "#ffc107", "Needs Improvement": "#dc3545",
                },
                hole=0.4,
            )
            fig_pie.update_layout(height=380)
            st.plotly_chart(fig_pie, use_container_width=True)

    with tab5:
        # Use model coefficients as a feature importance proxy
        try:
            model, scaler = get_model()
            coef_dict = dict(zip(ml.FEATURE_COLS, model.coef_.tolist()))
            feat_df = pd.DataFrame({
                "Feature":    [ml.FEATURE_LABELS[f] for f in ml.FEATURE_COLS],
                "Coefficient": [abs(coef_dict[f]) for f in ml.FEATURE_COLS],
            }).sort_values("Coefficient", ascending=True)

            fig_bar = px.bar(
                feat_df, x="Coefficient", y="Feature",
                orientation="h",
                title="Feature Importance (Ridge Regression Coefficients)",
                labels={"Coefficient": "Absolute Coefficient Value"},
                color="Coefficient",
                color_continuous_scale="Blues",
            )
            fig_bar.update_layout(height=420, showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)
            st.caption(
                "Larger coefficients indicate features with stronger influence "
                "on the predicted score."
            )
        except Exception as e:
            st.warning(f"Feature importance chart unavailable: {e}")

    # Correlation heatmap
    st.markdown("---")
    st.subheader("Correlation Heatmap")
    num_cols = ml.FEATURE_COLS + ["final_score"]
    corr = df[num_cols].corr().round(2)
    labels = [ml.FEATURE_LABELS.get(c, c) for c in num_cols]
    fig_heat = go.Figure(go.Heatmap(
        z=corr.values,
        x=labels,
        y=labels,
        colorscale="RdBu",
        zmid=0,
        text=corr.values,
        texttemplate="%{text}",
        showscale=True,
    ))
    fig_heat.update_layout(title="Feature Correlation Matrix", height=500)
    st.plotly_chart(fig_heat, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MODEL INFO
# ══════════════════════════════════════════════════════════════════════════════

def page_model():
    st.title("🤖 Machine Learning Model")
    st.markdown("Details about the model used to power the predictions.")
    st.markdown("---")

    try:
        df    = get_data()
        model, scaler = get_model()
        _, _, metrics, coef_dict = ml.train_model(df)
    except Exception as e:
        st.error(f"Could not load model/data: {e}")
        return

    # Metrics cards
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE (Mean Absolute Error)", f"{metrics['mae']}", help="Average prediction error in score points")
    c2.metric("RMSE",                      f"{metrics['rmse']}", help="Root Mean Squared Error")
    c3.metric("R\u00b2 Score",             f"{metrics['r2']}",  help="1.0 = perfect, 0.0 = baseline")

    st.markdown("---")

    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("About the Model")
        st.markdown("""
        **Algorithm:** Ridge Regression (Linear Regression with L2 regularisation)

        **Why Ridge?**
        - Simple, fast, and interpretable
        - Works very well for datasets with correlated features
        - Prevents overfitting via regularisation (alpha = 1.0)
        - No compiled extensions required — runs everywhere

        **Training Details:**
        - Dataset: 500 synthetic student records
        - Train / Test split: 80% / 20%
        - Features scaled with StandardScaler before training
        - Model saved to `models/student_model.pkl`
        """)
    with col_r:
        st.subheader("Features Used")
        feat_info = pd.DataFrame({
            "Feature":     list(ml.FEATURE_LABELS.values()),
            "Column Name": list(ml.FEATURE_LABELS.keys()),
            "Type":        ["Numeric"] * 6 + ["Binary (0/1)"] * 2,
        })
        st.dataframe(feat_info, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Model Coefficients")
    coef_df = pd.DataFrame({
        "Feature":     [ml.FEATURE_LABELS[f] for f in ml.FEATURE_COLS],
        "Coefficient": [round(coef_dict[f], 4) for f in ml.FEATURE_COLS],
    }).sort_values("Coefficient", ascending=False)

    fig = px.bar(
        coef_df, x="Feature", y="Coefficient",
        title="Ridge Regression Coefficients",
        color="Coefficient",
        color_continuous_scale="RdBu",
        text="Coefficient",
    )
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_layout(height=420, xaxis_tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Positive coefficients push the predicted score higher; negative ones lower it.")

    # Dataset preview
    st.markdown("---")
    st.subheader("Dataset Preview (first 10 rows)")
    st.dataframe(df.head(10), use_container_width=True)

    col_stats1, col_stats2 = st.columns(2)
    with col_stats1:
        st.markdown(f"**Total students:** {len(df)}")
        st.markdown(f"**Features:** {len(ml.FEATURE_COLS)}")
    with col_stats2:
        st.markdown(f"**Target column:** `final_score`")
        st.markdown(f"**Score range:** {df['final_score'].min():.1f} – {df['final_score'].max():.1f}")

    with st.expander("Full Dataset Statistics"):
        st.dataframe(df[ml.FEATURE_COLS + ["final_score"]].describe().round(2), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    page = sidebar()

    if page == "Home":
        page_home()
    elif page == "Predict":
        page_predict()
    elif page == "Visualise":
        page_visualise()
    elif page == "Model":
        page_model()


if __name__ == "__main__":
    main()
