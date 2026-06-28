import os
import sys
import pandas as pd

# Redirect stdout to a log file
log_file = open("download_log.txt", "w", encoding="utf-8")
sys.stdout = log_file
sys.stderr = log_file

path = r"C:\Users\mishr\.cache\kagglehub\datasets\rxydenxd\indian-railways-delay-dataset\versions\1"

print("--- Inspecting files in dataset ---")
for fname in sorted(os.listdir(path)):
    fpath = os.path.join(path, fname)
    if fname.endswith(".csv"):
        print(f"\nFile: {fname}")
        df_chunk = pd.read_csv(fpath, nrows=5)
        print("Columns:", df_chunk.columns.tolist())
        
        # Let's count rows
        num_rows = 0
        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            for _ in f:
                num_rows += 1
        print("Total rows:", num_rows - 1)
        
        df_sample = pd.read_csv(fpath, nrows=10)
        print("Sample rows:")
        print(df_sample)
        
        if fname == 'combined_delay.csv':
            # Check delay distribution
            df_delay = pd.read_csv(fpath, nrows=200000)
            print("\nNull delay count in first 200k:", df_delay['delay'].isnull().sum())
            non_null_delays = df_delay['delay'].dropna()
            print("Non-null delay stats (in first 200k):")
            print(non_null_delays.describe())
            print("Some non-zero delay sample values:")
            print(df_delay[df_delay['delay'] > 0].head(10))

log_file.close()
