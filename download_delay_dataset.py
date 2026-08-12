import os
import sys
import pandas as pd
import kagglehub


# ============================================================
# 1. Download dataset from Kaggle
# ============================================================

print("Downloading / locating Kaggle dataset...")

dataset_path = kagglehub.dataset_download(
    "rxydenxd/indian-railways-delay-dataset"
)

print(f"Dataset path: {dataset_path}")


# ============================================================
# 2. Log output
# ============================================================

log_file = open("download_log.txt", "w", encoding="utf-8")

original_stdout = sys.stdout
original_stderr = sys.stderr

sys.stdout = log_file
sys.stderr = log_file


try:

    print("====================================================")
    print("INDIAN RAILWAYS DATASET INSPECTION")
    print("====================================================")

    # ========================================================
    # 3. Inspect all CSV files
    # ========================================================

    for fname in sorted(os.listdir(dataset_path)):

        if not fname.endswith(".csv"):
            continue

        fpath = os.path.join(dataset_path, fname)

        print(f"\n\n{'=' * 60}")
        print(f"File: {fname}")
        print(f"{'=' * 60}")

        # ----------------------------------------------------
        # Read only a few rows to inspect columns
        # ----------------------------------------------------

        df_sample = pd.read_csv(fpath, nrows=10)

        print("\nColumns:")
        print(df_sample.columns.tolist())

        # ----------------------------------------------------
        # Count total rows
        # ----------------------------------------------------

        num_rows = 0

        with open(
            fpath,
            "r",
            encoding="utf-8",
            errors="ignore"
        ) as f:

            for _ in f:
                num_rows += 1

        total_rows = num_rows - 1

        print(f"\nTotal rows: {total_rows}")

        # ----------------------------------------------------
        # Sample data
        # ----------------------------------------------------

        print("\nSample rows:")
        print(df_sample)

        # ====================================================
        # 4. Special inspection for combined_delay.csv
        # ====================================================

        if fname == "combined_delay.csv":

            print("\n--- Delay Analysis ---")

            # IMPORTANT:
            # Don't load the entire 1M+ row dataset just for
            # inspection. Read only a chunk.

            df_delay = pd.read_csv(
                fpath,
                nrows=200_000
            )

            print(
                "\nRows inspected for delay analysis:",
                len(df_delay)
            )

            # ------------------------------------------------
            # Null values
            # ------------------------------------------------

            null_delay_count = df_delay["delay"].isnull().sum()

            print(
                "\nNull delay count in first 200k:",
                null_delay_count
            )

            # ------------------------------------------------
            # Delay statistics
            # ------------------------------------------------

            non_null_delays = df_delay["delay"].dropna()

            print("\nNon-null delay statistics:")
            print(non_null_delays.describe())

            # ------------------------------------------------
            # Non-zero delay examples
            # ------------------------------------------------

            print("\nSome non-zero delay examples:")

            print(
                df_delay[
                    df_delay["delay"] > 0
                ].head(10)
            )


finally:

    # ========================================================
    # 5. Restore stdout / stderr
    # ========================================================

    sys.stdout = original_stdout
    sys.stderr = original_stderr

    log_file.close()


print("Dataset inspection completed.")
print("Check download_log.txt for detailed output.")
