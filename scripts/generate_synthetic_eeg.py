import numpy as np
import pandas as pd

def generate_eeg_dataframe(duration=10, fs=256, alpha_amp=1.0, theta_amp=0.6):
    """
    Generate synthetic EEG signal containing alpha and theta waves.
    Returns a DataFrame with time and signal columns.
    """
    t = np.arange(0, duration, 1/fs)
    sig = (alpha_amp * np.sin(2*np.pi*10*t) +
           theta_amp * np.sin(2*np.pi*6*t) +
           0.5 * np.random.normal(size=len(t)))

    return pd.DataFrame({"time_s": t, "signal": sig})


# Optional: keep your original file-writing function
def generate(duration=10, fs=256, alpha_amp=1.0, theta_amp=0.6, out_file="data/example_eeg.csv"):
    df = generate_eeg_dataframe(duration, fs, alpha_amp, theta_amp)
    df.to_csv(out_file, index=False)
    print(f"Wrote {out_file} ({len(df)} samples)")


if __name__ == "__main__":
    generate()
