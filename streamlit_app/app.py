"""
NeuroState — EEG Alpha/Theta Mental State Classifier
Run with:
    streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.signal_processing import bandpower, butter_bandpass_filter, compute_spectrogram
from scipy import signal

st.set_page_config(page_title="NeuroState", layout="wide")

# ----------------------------------------------------------
# Title + Description
# ----------------------------------------------------------
st.title("🧠 NeuroState — EEG Alpha/Theta Mental State Classifier")
st.markdown("""
This mini-BCI demo:
- Loads or generates an EEG signal  
- Computes alpha, theta, and delta bandpower  
- Uses the theta/alpha ratio to classify mental state  
- Shows filtered signals and a spectrogram  
""")

# ----------------------------------------------------------
# Sidebar Inputs
# ----------------------------------------------------------
st.sidebar.header("Data Input")
uploaded_file = st.sidebar.file_uploader("Upload EEG CSV (single column of samples)", type=["csv"])
fs = st.sidebar.number_input("Sampling rate (Hz)", value=256.0, step=1.0)

window_sec = st.sidebar.slider("Welch window length (seconds)", 0.5, 5.0, 2.0)

st.sidebar.header("Synthetic Signal Settings")
duration = st.sidebar.number_input("Synthetic duration (s)", value=10)
mix_alpha = st.sidebar.slider("Alpha amplitude", 0.5, 2.0, 1.0)
mix_theta = st.sidebar.slider("Theta amplitude", 0.0, 2.0, 0.6)

st.sidebar.header("Classification Thresholds")
thresh_focused = st.sidebar.number_input("Focused if ratio <", value=0.5)
thresh_relaxed = st.sidebar.number_input("Relaxed if ratio <", value=1.5)

# ----------------------------------------------------------
# Load or Generate Data
# ----------------------------------------------------------
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if df.shape[1] == 1:
        sig = df.iloc[:, 0].values
    elif "signal" in df.columns:
        sig = df["signal"].values
    else:
        sig = df.select_dtypes(include=[np.number]).iloc[:, 0].values
    st.success("Loaded uploaded EEG file.")
else:
    st.info("No file uploaded — generating synthetic EEG.")
    t = np.arange(0, duration, 1 / fs)
    rng = np.random.default_rng(seed=42)
    sig = (
        mix_alpha * np.sin(2 * np.pi * 10 * t) +
        mix_theta * np.sin(2 * np.pi * 6 * t) +
        0.5 * rng.normal(size=len(t))
    )

sig = np.asarray(sig, dtype=float)
time_axis = np.arange(len(sig)) / fs

# ----------------------------------------------------------
# Raw Signal Display
# ----------------------------------------------------------
st.subheader("Raw EEG Signal")
fig1, ax1 = plt.subplots(figsize=(10, 3))
ax1.plot(time_axis, sig)
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Amplitude")
st.pyplot(fig1)

# ----------------------------------------------------------
# Filtered Bands
# ----------------------------------------------------------
st.subheader("Bandpass-Filtered Signals")
col1, col2 = st.columns(2)

with col1:
    alpha_sig = butter_bandpass_filter(sig, 8, 12, fs)
    fig_a, ax_a = plt.subplots(figsize=(6, 2))
    ax_a.plot(time_axis, alpha_sig)
    ax_a.set_title("Alpha (8–12 Hz)")
    st.pyplot(fig_a)

with col2:
    theta_sig = butter_bandpass_filter(sig, 4, 7, fs)
    fig_t, ax_t = plt.subplots(figsize=(6, 2))
    ax_t.plot(time_axis, theta_sig)
    ax_t.set_title("Theta (4–7 Hz)")
    st.pyplot(fig_t)

# ----------------------------------------------------------
# Bandpower Computation
# ----------------------------------------------------------
delta_bp = bandpower(sig, fs, (1, 4), window_sec=window_sec)
theta_bp = bandpower(sig, fs, (4, 7), window_sec=window_sec)
alpha_bp = bandpower(sig, fs, (8, 12), window_sec=window_sec)

st.subheader("Bandpower (absolute)")
fig2, ax2 = plt.subplots(figsize=(6, 3))
ax2.bar(["Delta (1–4)", "Theta (4–7)", "Alpha (8–12)"], [delta_bp, theta_bp, alpha_bp])
ax2.set_ylabel("Power")
st.pyplot(fig2)

# ----------------------------------------------------------
# Classification
# ----------------------------------------------------------
ratio = theta_bp / alpha_bp if alpha_bp > 0 else np.inf

if ratio < thresh_focused:
    state = "Focused"
elif ratio < thresh_relaxed:
    state = "Relaxed"
else:
    state = "Drowsy / Low Arousal"

st.subheader("Classified Mental State")
st.metric("Theta/Alpha Ratio", f"{ratio:.3f}")
st.markdown(f"### **State: {state}**")

# ----------------------------------------------------------
# Spectrogram
# ----------------------------------------------------------
st.subheader("Spectrogram")
f, t_spec, Sxx = compute_spectrogram(sig, fs, nperseg=int(window_sec * fs))
fig3, ax3 = plt.subplots(figsize=(10, 3))
ax3.pcolormesh(t_spec, f, 10 * np.log10(Sxx + 1e-12), shading="gouraud")
ax3.set_ylim(0, 40)
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Frequency (Hz)")
st.pyplot(fig3)
