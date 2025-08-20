import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load
from sklearn.preprocessing import StandardScaler

# Load trained model and scaler
model = load("models/ids_model.pkl")
scaler = load("models/scaler.pkl")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("data/Data.csv")
    return df

df = load_data()

st.title("🔐 Intrusion Detection System Dashboard")

st.markdown("### 📊 Dataset Overview")
st.write(df.head())

# Correlation Heatmap
st.markdown("### 🔥 Feature Correlation Heatmap")
corr = df.corr(numeric_only=True)
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(corr, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# Bar Chart for Attack/Normal distribution
st.markdown("### 🚨 Class Distribution")
if 'Label' in df.columns:
    st.bar_chart(df['Label'].value_counts())
else:
    st.warning("⚠️ 'Label' column not found in dataset!")

# Make predictions on a random sample
st.markdown("### 🤖 Predict Random Sample")
if st.button("Predict Sample"):
    sample = df.drop('Label', axis=1).sample(1) if 'Label' in df.columns else df.sample(1)
    scaled = scaler.transform(sample)
    prediction = model.predict(scaled)[0]
    st.write("Prediction:", "🔒 **Attack**" if prediction == 1 else "✅ **Normal**")
