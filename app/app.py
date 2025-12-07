"""
app.py - NeuroState: EEG Alpha/Theta Mental State Classifier (Streamlit)
Run: streamlit run app/app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.signal_processing import bandpower, butter_bandpass_filter, compute_spectrogram
from scipy import signal

st.set_page_config(page_title="NeuroState", layout="wide")

st.title("NeuroState — EEG Alpha/Theta Mental State Classifier")
st.markdown("An educational mini-BCI that computes alpha and theta bandpowers and classifies a mental state.")

# Sidebar: inputs
st.sidebar.header("Data input & settings")
uploaded_file = st.sidebar.file_uploader("Upload EEG CSV (single column of samples) or leave empty to use synthetic", type=["csv"])
fs = st.sidebar.number_input("Sampling rate (Hz)", value=256.0, min_value=20.0, step=1.0)
window_sec = st.sidebar.slider("Welch window (seconds)", 0.5, 5.0, 2.0)

st.sidebar.markdown("### Classification thresholds (theta/alpha ratio)")
thresh_focused = st.sidebar.number_input("Threshold for 'Focused' (ratio < )", value=0.5, step=0.1)
thresh_relaxed = st.sidebar.number_input("Threshold for 'Relaxed' (ratio between )", value=1.5, step=0.1)

# Load or generate data
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    # Accept either single column or column named 'signal'
    if df.shape[1] == 1:
        sig = df.iloc[:, 0].values
    elif 'signal' in df.columns:
        sig = df['signal'].values
    else:
        # take first numeric column
        sig = df.select_dtypes(include=[np.number]).iloc[:, 0].values
    st.success("Loaded uploaded EEG file.")
else:
    st.info("No file uploaded — generating synthetic EEG (alpha + theta mix + noise).")
    duration = st.sidebar.number_input("Synthetic duration (s)", value=10)
    t = np.arange(0, duration, 1/fs)
    # create two sinusoids: theta (6 Hz) and alpha (10 Hz), mix ratio adjustable
    mix_alpha = st.sidebar.slider("Synthetic alpha amplitude", 0.5, 2.0, 1.0)
    mix_theta = st.sidebar.slider("Synthetic theta amplitude", 0.0, 2.0, 0.6)
    rng = np.random.default_rng(seed=42)
    sig = (mix_alpha * np.sin(2*np.pi*10*t) +
           mix_theta * np.sin(2*np.pi*6*t) +
           0.5 * rng.normal(size=len(t)))

# Ensure 1D numpy array
sig = np.asarray(sig, dtype=float)

# Show raw signal
st.subheader("Raw EEG signal")
fig1, ax1 = plt.subplots(figsize=(10, 3))
time_axis = np.arange(len(sig)) / fs
ax1.plot(time_axis, sig)
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Amplitude (a.u.)")
st.pyplot(fig1)

# Compute filtered bands for display (optional)
st.subheader("Bandpass filtered views (for demonstration)")
col1, col2 = st.columns(2)
with col1:
    alpha_sig = butter_bandpass_filter(sig, 8, 12, fs)
    figa, axa = plt.subplots(figsize=(6,2))
    axa.plot(time_axis, alpha_sig)
    axa.set_title("Alpha (8–12 Hz)")
    st.pyplot(figa)
with col2:
    theta_sig = butter_bandpass_filter(sig, 4, 7, fs)
    figt, axt = plt.subplots(figsize=(6,2))
    axt.plot(time_axis, theta_sig)
    axt.set_title("Theta (4–7 Hz)")
    st.pyplot(figt)

# Compute bandpowers
alpha_bp = bandpower(sig, fs, (8, 12), window_sec=window_sec)
theta_bp = bandpower(sig, fs, (4, 7), window_sec=window_sec)
delta_bp = bandpower(sig, fs, (1, 4), window_sec=window_sec)

# Show bandpower bar chart
st.subheader("Bandpower (absolute)")
fig2, ax2 = plt.subplots(figsize=(6,3))
ax2.bar(['Delta (1-4)','Theta (4-7)','Alpha (8-12)'], [delta_bp, theta_bp, alpha_bp])
ax2.set_ylabel("Power (a.u.)")
st.pyplot(fig2)

# Classification via theta/alpha ratio
ratio = (theta_bp / alpha_bp) if alpha_bp > 0 else np.inf
if ratio < thresh_focused:
    state = "Focused"
elif ratio < thresh_relaxed:
    state = "Relaxed"
else:
    state = "Drowsy / Low Arousal"

st.subheader("Classified mental state")
st.metric("Theta / Alpha ratio", f"{ratio:.3f}", delta=None)
st.markdown(f"**Inferred state:** **{state}**")

st.markdown("---")
st.subheader("Spectrogram (full signal)")
f, t_spec, Sxx = compute_spectrogram(sig, fs, nperseg=int(window_sec*fs))
fig3, ax3 = plt.subplots(figsize=(10,3))
ax3.pcolormesh(t_spec, f, 10*np.log10(Sxx+1e-12), shading='gouraud')
ax3.set_ylabel('Frequency [Hz]')
ax3.set_xlabel('Time [sec]')
ax3.set_ylim(0, 40)
st.pyplot(fig3)

st.markdown("### Notes")
st.write("""
- This demo uses Welch's method to compute bandpower and a simple ratio (Theta/Alpha) for classification.
- Thresholds are adjustable in the sidebar. For real data, you'd calibrate thresholds using labelled samples.
- For reproducibility, provide the same sample rate and known data format in README.
""")
