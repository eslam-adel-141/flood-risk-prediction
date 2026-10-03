import pandas as pd
import streamlit as st

from src.config import FEATURE_NAMES
from src.predictor import get_model_info, predict_flood_probability, risk_level

st.set_page_config(page_title="Flood Risk Predictor", page_icon="🌊", layout="wide")

st.title("🌊 Flood Probability Prediction")
st.markdown("Predict flood risk based on environmental and socio-economic factors.")

st.sidebar.header("⚙️ Input Features")
st.sidebar.markdown("Adjust each factor (0 = minimal, 20 = extreme)")

input_dict = {}
col1, col2 = st.sidebar.columns(2)
for i, feat in enumerate(FEATURE_NAMES):
    col = col1 if i % 2 == 0 else col2
    input_dict[feat] = col.slider(feat, 0, 20, 8)

if st.sidebar.button("🔍 Predict Flood Risk", use_container_width=True):
    prob = predict_flood_probability(input_dict)
    level = risk_level(prob)

    c1, c2 = st.columns(2)
    c1.metric("Predicted Flood Probability", f"{prob:.3f}")
    c2.metric("Risk Level", level)
    st.progress(min(prob, 1.0))

    st.subheader("📊 Input Summary")
    st.dataframe(pd.DataFrame([input_dict]).T.rename(columns={0: "Value"}))
else:
    st.info("👈 Set the factors in the sidebar and click Predict.")

with st.expander("ℹ️ Model info"):
    st.json(get_model_info())

st.markdown("---")
st.caption("Model trained on the Flood Prediction Dataset (50,000 samples).")
