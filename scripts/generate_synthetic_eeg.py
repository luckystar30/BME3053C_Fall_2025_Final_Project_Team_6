import numpy as np
import pandas as pd

def generate(duration=10, fs=256, alpha_amp=1.0, theta_amp=0.6, out_file="data/example_eeg.csv"):
    t = np.arange(0, duration, 1/fs)
    sig = (alpha_amp * np.sin(2*np.pi*10*t) +
           theta_amp * np.sin(2*np.pi*6*t) +
           0.5 * np.random.normal(size=len(t)))
    df = pd.DataFrame({"signal": sig})
    df.to_csv(out_file, index=False)
    print(f"Wrote {out_file} ({len(sig)} samples)")

if __name__ == "__main__":
    generate()
