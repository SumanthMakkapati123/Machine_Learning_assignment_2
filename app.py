
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report

# Page Config (Must be first)
st.set_page_config(
    page_title="Football Predictor", 
    page_icon="⚽", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------
# HELPER FUNCTIONS
# -----------------
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    else:
        # Load default combined data
        import glob
        files = glob.glob("clean_data/*.csv")
        dfs = []
        for file in files:
            try:
                dfs.append(pd.read_csv(file, on_bad_lines='skip'))
            except:
                pass
        if dfs:
            return pd.concat(dfs, ignore_index=True)
        return None

@st.cache_resource
def load_model(name):
    filename = f"models/{name.replace(' ', '_').lower()}.pkl"
    if os.path.exists(filename):
        return joblib.load(filename)
    else:
        return None

# Load Label Encoder
if os.path.exists('models/label_encoder.pkl'):
    le = joblib.load('models/label_encoder.pkl')
else:
    st.error("Label Encoder not found. Please train models first.")
    st.stop()

# -----------------
# SIDEBAR UI
# -----------------
with st.sidebar:
    st.title("⚙️ Configuration")
    
    st.markdown("### Model")
    model_names = [
        "Logistic Regression",
        "Decision Tree",
        "kNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    ]
    selected_model_name = st.selectbox("Select Classifier", model_names, index=0)
    
    st.markdown("### Data")
    uploaded_file = st.file_uploader("Upload Test Data (CSV)", type=["csv"])
    
    st.divider()
    st.info(f"Loaded Model: **{selected_model_name}**")

# -----------------
# MAIN CONTENT
# -----------------

# Header with User Name
st.title("⚽ Football Match Result Predictor")
st.markdown("---")

# Load Data
data = load_data(uploaded_file)
model = load_model(selected_model_name)

if model is None:
    st.error(f"Model '{selected_model_name}' not found. Please run `train_models.py` first.")
    st.stop()

if data is None:
    st.warning("No data found. Please run `normalize_data.py` or upload a file.")
    st.stop()

# Define feature columns and mapping for display
feature_map = {
    'HS': 'Home Shots',
    'AS': 'Away Shots',
    'HST': 'Home Shots on Target',
    'AST': 'Away Shots on Target',
    'HF': 'Home Fouls',
    'AF': 'Away Fouls',
    'HC': 'Home Corners',
    'AC': 'Away Corners',
    'HY': 'Home Yellow Cards',
    'AY': 'Away Yellow Cards',
    'HR': 'Home Red Cards',
    'AR': 'Away Red Cards',
    'FTR': 'Full Time Result',
    'Date': 'Match Date',
    'HomeTeam': 'Home Team',
    'AwayTeam': 'Away Team'
}

feature_cols = ['HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR']
target_col = 'FTR'

# --- SECTION 1: PREDICTION ---
st.header("1. Predict Match Outcome")

col_input, col_result = st.columns([2, 1])

with col_input:
    st.markdown("#### Match Statistics")
    with st.expander("Expand to Adjust Stats", expanded=True):
        st.markdown("**Shooting**")
        c1, c2 = st.columns(2)
        hs = c1.slider("Home Shots", 0, 30, 12, key="hs")
        as_ = c2.slider("Away Shots", 0, 30, 10, key="as")
        c3, c4 = st.columns(2)
        hst = c3.slider("Home On Target", 0, 20, 5, key="hst")
        ast = c4.slider("Away On Target", 0, 20, 4, key="ast")
    
        st.divider()
        
        st.markdown("**Set Pieces & Fouls**")
        c1, c2, c3, c4 = st.columns(4)
        hc = c1.number_input("Home Corners", 0, 20, 6)
        ac = c2.number_input("Away Corners", 0, 20, 4)
        hf = c3.number_input("Home Fouls", 0, 30, 11)
        af = c4.number_input("Away Fouls", 0, 30, 12)
        
        st.divider()

        st.markdown("**Discipline (Cards)**")
        c1, c2, c3, c4 = st.columns(4)
        hy = c1.number_input("Home Yellow", 0, 10, 2)
        ay = c2.number_input("Away Yellow", 0, 10, 2)
        hr = c3.number_input("Home Red", 0, 5, 0)
        ar = c4.number_input("Away Red", 0, 5, 0)

    # Prepare input vector
    input_data = pd.DataFrame([[hs, as_, hst, ast, hf, af, hc, ac, hy, ay, hr, ar]], columns=feature_cols)

with col_result:
    st.markdown("#### Prediction Result")
    # Auto-predict on change
    prediction_idx = model.predict(input_data)[0]
    prediction_label = le.inverse_transform([prediction_idx])[0]
    
    result_map = {'H': 'Home Win', 'A': 'Away Win', 'D': 'Draw'}
    full_result = result_map.get(prediction_label, prediction_label)
    
    # Display Result Card
    st.markdown(f"""
    <div style="background-color: #e0f2fe; padding: 20px; border-radius: 10px; text-align: center; border: 2px solid #0284c7;">
        <h2 style="color: #0369a1; margin:0;">{full_result}</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Probability
    probs = model.predict_proba(input_data)[0]
    st.markdown("##### Winning Probability")
    
    classes = ['Away', 'Draw', 'Home']
    # Ensure correct mapping based on le.classes_
    # Usually le.classes_ are sorted: ['A', 'D', 'H']
    
    prob_df = pd.DataFrame({
        "Outcome": classes,
        "Probability": [probs[0], probs[1], probs[2]]
    })
    st.bar_chart(prob_df.set_index("Outcome"))

st.divider()

# --- SECTION 2: EVALUATION ---
st.header("2. Model Evaluation")

if st.button("Run Evaluation on Current Dataset", type="primary"):
    with st.spinner("Calculating metrics..."):
        # Filter valid data
        eval_data = data.dropna(subset=feature_cols + [target_col])
        
        if len(eval_data) > 0:
            X_eval = eval_data[feature_cols]
            y_true = eval_data[target_col]
            y_true_encoded = le.transform(y_true)
            
            y_pred = model.predict(X_eval)
            y_prob = model.predict_proba(X_eval)
            
            acc = accuracy_score(y_true_encoded, y_pred)
            auc = roc_auc_score(y_true_encoded, y_prob, multi_class='ovr')
            f1 = f1_score(y_true_encoded, y_pred, average='weighted')
            prec = precision_score(y_true_encoded, y_pred, average='weighted')
            rec = recall_score(y_true_encoded, y_pred, average='weighted')
            mcc = matthews_corrcoef(y_true_encoded, y_pred)
            
            # Metrics Row 1
            col1, col2, col3 = st.columns(3)
            col1.metric("Accuracy", f"{acc:.2%}")
            col2.metric("AUC Score", f"{auc:.3f}")
            col3.metric("F1 Score", f"{f1:.3f}")
            
            # Metrics Row 2
            col4, col5, col6 = st.columns(3)
            col4.metric("Precision", f"{prec:.3f}")
            col5.metric("Recall", f"{rec:.3f}")
            col6.metric("MCC", f"{mcc:.3f}")
            
            st.markdown("##### Confusion Matrix")
            # Center the plot
            col_left, col_mid, col_right = st.columns([1, 2, 1])
            with col_mid:
                cm = confusion_matrix(y_true_encoded, y_pred)
                fig, ax = plt.subplots(figsize=(5, 3))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                            xticklabels=['Away', 'Draw', 'Home'], 
                            yticklabels=['Away', 'Draw', 'Home'])
                plt.ylabel('Actual Result')
                plt.xlabel('Predicted Result')
                st.pyplot(fig)
            
        else:
            st.error("Insufficient data for evaluation.")

st.divider()

# --- SECTION 3: DATASET ---
st.header("3. Dataset Explorer")

with st.expander("Show Raw Data Preview"):
    # Rename columns for display
    display_df = data.copy()
    display_df.rename(columns=feature_map, inplace=True)
    
    # Filter columns to show relevant ones first
    cols_to_show = ['Match Date', 'Home Team', 'Away Team', 'Full Time Result'] + \
                   [c for c in display_df.columns if c not in ['Match Date', 'Home Team', 'Away Team', 'Full Time Result']]
    
    st.dataframe(display_df[cols_to_show], use_container_width=True, height=400)
