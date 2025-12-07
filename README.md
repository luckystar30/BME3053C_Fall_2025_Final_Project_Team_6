# NeuroState — EEG Alpha/Theta Mental State Classifier

A mini-BCI educational app that computes alpha and theta bandpower from EEG signals and classifies mental state.

## Biomedical Context

This app is designed for neuroscience students, BCI enthusiasts, and educators to explore EEG signals and understand how alpha and theta rhythms relate to mental state (focused, relaxed, drowsy). It provides an interactive way to visualize EEG dynamics and basic mental state classification.

## Quick Start Instructions

### Opening the Repository in GitHub Codespaces

1. Open your GitHub repository page.
2. Click the **Code** button → **Open with Codespaces** → **New codespace**.
3. Wait for Codespaces to launch and load the repository environment.

### Running the Application

1. Ensure Python dependencies are installed:

pip install streamlit numpy scipy pandas matplotlib
cd streamlit_app
streamlit run app.py

## Usage Guide

[Step-by-step explanation with screenshots or text]

- **Step 1:Upload or Generate EEG Data** You can upload a CSV file containing a single-column EEG signal, if no file is uploaded, the app generates a synthetic EEG (alpha + theta sine waves with noise). Sidebar controls allow you to adjust sampling rate, synthetic duration, and alpha/theta amplitudes
- **Step 2:Adjust Analysis Parameters** Use the Welch window slider to set the segment length for bandpower calculations. Set classification thresholds for the theta/alpha ratio to define “Focused” and “Relaxed” states.
- **Step 3:Visualize Signals and Results** Raw EEG plot: Shows the signal over time. Bandpass-filtered plots: Displays alpha (8–12 Hz) and theta (4–7 Hz) components separately. Bandpower bar chart: Shows absolute power for delta, theta, and alpha bands. Mental state classification: Displays the theta/alpha ratio and inferred mental state (Focused, Relaxed, or Drowsy). Spectrogram: Visualizes how EEG power is distributed across frequencies over time, showing dynamic changes in alpha and theta activity.
- Note: All major code blocks were developed using GitHub Copilot (GPT-5.1 Codex in Agent mode). The code was manually reviewed, tested, and documented to ensure accuracy and reproducibility.

## Data Description (optional)


### Data Source

EEG signals are either synthetic or user-provided CSV files.

Synthetic signals combine alpha (10 Hz) and theta (6 Hz) sine waves with Gaussian noise to simulate EEG activity.


## Project Structure

streamlit_app/
├── app.py                    # Main Streamlit application
├── utils/                    # Signal processing functions
│   ├── __init__.py
│   └── signal_processing.py  # Functions for bandpower, filtering, spectrogram

app.py: Handles the UI, data loading, plotting, and mental state classification.
utils/signal_processing.py: Contains core DSP functions (bandpower, butter_bandpass_filter, compute_spectrogram).

Synthetic EEG generation occurs in app.py if no CSV is uploaded.

