import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler  # Only for loading, but not re-fitting

# ────────────────────────────────────────────────
# Page config
# ────────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="centered"
)

# ────────────────────────────────────────────────
# Load model and scaler
# ────────────────────────────────────────────────
@st.cache_resource
def load_predictor():
    with open('diabetes_model.pkl', 'rb') as f:
        loaded = pickle.load(f)
    return loaded

loaded = load_predictor()
model = loaded['model']
scaler = loaded['scaler']

# ────────────────────────────────────────────────
# Title & description
# ────────────────────────────────────────────────
st.title("🩺 Diabetes Risk Predictor")
st.markdown("""
This app uses a **Random Forest** model trained on the Pima Indians Diabetes dataset  
to estimate diabetes risk based on 8 clinical features. Model accuracy: ~75%.
""")

st.info("Note: This is **not** medical advice — for educational use only.")

# ────────────────────────────────────────────────
# Input fields (matching Pima columns)
# ────────────────────────────────────────────────
st.subheader("Enter health measurements")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", 0, 20, 1)
    glucose = st.number_input("Glucose (mg/dL)", 0, 200, 120)
    blood_pressure = st.number_input("Blood Pressure (mm Hg)", 0, 150, 70)
    skin_thickness = st.number_input("Skin Thickness (mm)", 0, 100, 20)

with col2:
    insulin = st.number_input("Insulin (mu U/ml)", 0, 900, 80)
    bmi = st.number_input("BMI (kg/m²)", 0.0, 70.0, 25.0, step=0.1)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.47, step=0.01)
    age = st.number_input("Age (years)", 21, 90, 30)

# ────────────────────────────────────────────────
# Prepare input DataFrame
# ────────────────────────────────────────────────
input_data = np.array([
    pregnancies, glucose, blood_pressure, skin_thickness,
    insulin, bmi, dpf, age
]).reshape(1, -1)

df_input = pd.DataFrame(input_data, columns=[
    'PREGNANCIES', 'GLUCOSE', 'BLOODPRESSURE', 'SKINTHICKNESS',
    'INSULIN', 'BMI', 'DIABETESPEDIGREEFUNCTION', 'AGE'
])

# Step 1: Handle zeros as missing (replace with NaN, then impute with training medians)
cols_with_zeros = ['GLUCOSE', 'BLOODPRESSURE', 'SKINTHICKNESS', 'INSULIN', 'BMI']
df_input[cols_with_zeros] = df_input[cols_with_zeros].replace(0, np.nan)

# Impute with known medians from Pima dataset (since no imputer saved in pkl)
medians = {
    'GLUCOSE': 117,
    'BLOODPRESSURE': 72,
    'SKINTHICKNESS': 29,
    'INSULIN': 125,
    'BMI': 32.3
}
df_input = df_input.fillna(medians)

# Step 2: Feature engineering (replicate notebook logic)
df_input['NEW_AGE_CAT'] = (df_input['AGE'] >= 50).astype(int)  # 1 for senior, 0 otherwise

df_input['NEW_INSULIN_SCORE'] = ((df_input['INSULIN'] >= 16) & (df_input['INSULIN'] <= 166)).astype(int)  # 1 for Normal, 0 for Abnormal

df_input['NEW_GLUCOSE*INSULIN'] = df_input['GLUCOSE'] * df_input['INSULIN']
df_input['NEW_GLUCOSE*PREGNANCIES'] = df_input['GLUCOSE'] * df_input['PREGNANCIES']

# Binary flags for BMI and Glucose
df_input['NEW_BMI_Normal'] = ((df_input['BMI'] >= 18.5) & (df_input['BMI'] < 25)).astype(int)
df_input['NEW_BMI_Prediabetes'] = ((df_input['BMI'] >= 25) & (df_input['BMI'] < 30)).astype(int)
df_input['NEW_BMI_Diabetes'] = (df_input['BMI'] >= 30).astype(int)

df_input['NEW_GLUCOSE_Prediabetes'] = ((df_input['GLUCOSE'] >= 100) & (df_input['GLUCOSE'] < 126)).astype(int)
df_input['NEW_GLUCOSE_Diabetes'] = (df_input['GLUCOSE'] >= 126).astype(int)

# Binary columns for selected NEW_AGE_BMI_NOM categories (matching final features)
df_input['NEW_AGE_BMI_NOM_obesesenior'] = ((df_input['BMI'] >= 30) & (df_input['AGE'] >= 50)).astype(int)
df_input['NEW_AGE_BMI_NOM_underweightmature'] = ((df_input['BMI'] < 18.5) & (df_input['AGE'] >= 21) & (df_input['AGE'] < 50)).astype(int)

# Binary columns for selected NEW_AGE_GLUCOSE_NOM categories (matching final features)
df_input['NEW_AGE_GLUCOSE_NOM_hiddensenior'] = ((df_input['GLUCOSE'] >= 100) & (df_input['GLUCOSE'] <= 125) & (df_input['AGE'] >= 50)).astype(int)
df_input['NEW_AGE_GLUCOSE_NOM_highmature'] = ((df_input['GLUCOSE'] > 125) & (df_input['AGE'] >= 21) & (df_input['AGE'] < 50)).astype(int)
df_input['NEW_AGE_GLUCOSE_NOM_highsenior'] = ((df_input['GLUCOSE'] > 125) & (df_input['AGE'] >= 50)).astype(int)
df_input['NEW_AGE_GLUCOSE_NOM_lowmature'] = ((df_input['GLUCOSE'] < 70) & (df_input['AGE'] >= 21) & (df_input['AGE'] < 50)).astype(int)
df_input['NEW_AGE_GLUCOSE_NOM_lowsenior'] = ((df_input['GLUCOSE'] < 70) & (df_input['AGE'] >= 50)).astype(int)
df_input['NEW_AGE_GLUCOSE_NOM_normalmature'] = ((df_input['GLUCOSE'] >= 70) & (df_input['GLUCOSE'] < 100) & (df_input['AGE'] >= 21) & (df_input['AGE'] < 50)).astype(int)
df_input['NEW_AGE_GLUCOSE_NOM_normalsenior'] = ((df_input['GLUCOSE'] >= 70) & (df_input['GLUCOSE'] < 100) & (df_input['AGE'] >= 50)).astype(int)

# Step 3: Scale numerical columns (using loaded scaler)
# Define the numerical columns that were actually scaled (without OUTCOME)
# Define the exact columns the scaler was fitted on (including OUTCOME in training)
# But we only need the order of the features we actually have
# ────────────────────────────────────────────────
# Scaling workaround – use positional order, ignore names
# ────────────────────────────────────────────────

# These are the 10 numerical columns the scaler should have been fitted on (excluding OUTCOME)
real_num_cols = [
    'PREGNANCIES', 'GLUCOSE', 'BLOODPRESSURE', 'SKINTHICKNESS', 'INSULIN',
    'BMI', 'DIABETESPEDIGREEFUNCTION', 'AGE',
    'NEW_GLUCOSE*INSULIN', 'NEW_GLUCOSE*PREGNANCIES'
]

# Extract only these columns as numpy array (in a fixed order)
X_to_scale = df_input[real_num_cols].to_numpy()  # shape (1, 10)

# The scaler was fitted on 11 columns (10 + OUTCOME), so it expects arrays with 11 columns
# Add a dummy column for OUTCOME (values don't matter – scaler just needs the shape)
dummy_outcome = np.zeros((X_to_scale.shape[0], 1))  # shape (1, 1)
X_for_scaler = np.hstack([X_to_scale, dummy_outcome])  # now shape (1, 11)

# Transform using the array directly
scaled_array = scaler.transform(X_for_scaler)  # scaler accepts array without name check

# Take only the first 10 columns of scaled result (ignore dummy)
scaled_real = scaled_array[:, :10]

# Put back into df_input
df_input[real_num_cols] = scaled_real
# Step 4: Select exact final features in order
final_columns = [
    'PREGNANCIES', 'GLUCOSE', 'BLOODPRESSURE', 'SKINTHICKNESS', 'INSULIN', 'BMI', 
    'DIABETESPEDIGREEFUNCTION', 'AGE', 'NEW_AGE_CAT', 'NEW_INSULIN_SCORE', 
    'NEW_GLUCOSE*INSULIN', 'NEW_GLUCOSE*PREGNANCIES', 
    'NEW_AGE_BMI_NOM_obesesenior', 'NEW_AGE_BMI_NOM_underweightmature', 
    'NEW_AGE_GLUCOSE_NOM_hiddensenior', 'NEW_AGE_GLUCOSE_NOM_highmature', 
    'NEW_AGE_GLUCOSE_NOM_highsenior', 'NEW_AGE_GLUCOSE_NOM_lowmature', 
    'NEW_AGE_GLUCOSE_NOM_lowsenior', 'NEW_AGE_GLUCOSE_NOM_normalmature', 
    'NEW_AGE_GLUCOSE_NOM_normalsenior', 'NEW_BMI_Normal', 'NEW_BMI_Prediabetes', 
    'NEW_BMI_Diabetes', 'NEW_GLUCOSE_Prediabetes', 'NEW_GLUCOSE_Diabetes'
]

# Ensure all columns exist (add missing with 0 if needed, though should be all created)
for col in final_columns:
    if col not in df_input.columns:
        df_input[col] = 0

df_final = df_input[final_columns]
input_for_model = df_final.to_numpy()  # Shape: (1, 26)

# ────────────────────────────────────────────────
# Predict button
# ────────────────────────────────────────────────
if st.button("Predict Diabetes Risk", type="primary"):
    try:
        with st.spinner("Analyzing..."):
            # Debug (remove later if not needed)
            st.write("Debug → Final input shape:", input_for_model.shape)
            st.write("Debug → Model expects features:", model.n_features_in_)

            prediction = model.predict(input_for_model)[0]
            prob = model.predict_proba(input_for_model)[0][1]

        st.divider()

        if prediction == 1:
            st.error(f"**High risk of diabetes** (probability: {prob:.1%})")
            st.markdown("⚠️ Consider consulting a doctor soon.")
        else:
            st.success(f"**Low risk of diabetes** (probability: {prob:.1%})")
            st.markdown("🎉 Keep maintaining a healthy lifestyle!")

        st.progress(prob)

    except Exception as e:
        st.error("Prediction failed!")
        st.exception(e)

# ────────────────────────────────────────────────
# Footer
# ────────────────────────────────────────────────
st.divider()
st.caption("Model accuracy ~75% | Built with Streamlit & Random Forest")