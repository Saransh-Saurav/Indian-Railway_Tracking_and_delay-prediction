import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error

# Force UTF-8 output on Windows
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None


# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH = r"C:\Users\mishr\.cache\kagglehub\datasets\rxydenxd\indian-railways-delay-dataset\versions\1"
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "model")
SAMPLE_ROWS = 500_000   # limit to avoid OOM on a laptop

os.makedirs(MODEL_DIR, exist_ok=True)


def load_data():
    print("Loading combined_delay.csv …")
    delay_df  = pd.read_csv(os.path.join(DATA_PATH, "combined_delay.csv"),   nrows=SAMPLE_ROWS)
    print("Loading combined_schedule.csv …")
    sched_df  = pd.read_csv(os.path.join(DATA_PATH, "combined_schedule.csv"))
    print("Loading train_details.csv …")
    train_df  = pd.read_csv(os.path.join(DATA_PATH, "train_details.csv"))
    print("Loading station_full_names.csv …")
    stat_df   = pd.read_csv(os.path.join(DATA_PATH, "station_full_names.csv"))
    return delay_df, sched_df, train_df, stat_df


def time_to_hour(t):
    """Convert 'HH:MM:SS' or 'HH:MM' to hour float. Returns NaN on error."""
    try:
        parts = str(t).split(":")
        return int(parts[0]) + int(parts[1]) / 60
    except Exception:
        return float("nan")


def preprocess(delay_df, sched_df, train_df, stat_df):
    print("Preprocessing …")

    # ── Drop rows with null delay ────────────────────────────────────────────
    df = delay_df.dropna(subset=["delay"]).copy()

    # ── Parse date features ──────────────────────────────────────────────────
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["day_of_week"] = df["date"].dt.dayofweek   # 0=Mon … 6=Sun
    df["month"]       = df["date"].dt.month

    # ── Merge schedule for distance & scheduled time ─────────────────────────
    sched_df = sched_df[["train_no", "station_name", "station_no",
                          "distance_from_origin", "arrival_time"]].copy()
    sched_df["scheduled_hour"] = sched_df["arrival_time"].apply(time_to_hour)
    sched_df["train_no"] = sched_df["train_no"].astype(str)
    df["train_no"]       = df["train_no"].astype(str)
    df = df.merge(
        sched_df[["train_no", "station_name", "station_no",
                  "distance_from_origin", "scheduled_hour"]],
        on=["train_no", "station_name", "station_no"],
        how="left"
    )

    # ── Merge train type ─────────────────────────────────────────────────────
    train_df["train_no"] = train_df["train_no"].astype(str)
    df = df.merge(train_df[["train_no", "type_code"]], on="train_no", how="left")

    # ── Merge station zone ───────────────────────────────────────────────────
    df = df.merge(
        stat_df[["station_name", "station_zone"]],
        on="station_name", how="left"
    )

    # ── Encode categoricals ──────────────────────────────────────────────────
    le_type  = LabelEncoder()
    le_zone  = LabelEncoder()
    le_train = LabelEncoder()

    df["type_code"]    = df["type_code"].fillna("UNKNOWN")
    df["station_zone"] = df["station_zone"].fillna("UNKNOWN")

    df["type_code_enc"]    = le_type.fit_transform(df["type_code"])
    df["station_zone_enc"] = le_zone.fit_transform(df["station_zone"])
    df["train_no_enc"]     = le_train.fit_transform(df["train_no"])

    # ── Drop remaining nulls in feature columns ──────────────────────────────
    features = ["train_no_enc", "station_no", "distance_from_origin",
                "scheduled_hour", "day_of_week", "month",
                "type_code_enc", "station_zone_enc"]
    df = df.dropna(subset=features)

    # ── Clip extreme outlier delays (keep ±300 min) ──────────────────────────
    df = df[(df["delay"] >= -60) & (df["delay"] <= 300)]

    X = df[features].values
    y = df["delay"].values

    # ── Save encoders and feature metadata ───────────────────────────────────
    joblib.dump(le_type,  os.path.join(MODEL_DIR, "le_type.joblib"))
    joblib.dump(le_zone,  os.path.join(MODEL_DIR, "le_zone.joblib"))
    joblib.dump(le_train, os.path.join(MODEL_DIR, "le_train.joblib"))

    # Save mapping dicts for inference
    type_map   = {str(k): int(v) for k, v in zip(le_type.classes_,  le_type.transform(le_type.classes_))}
    zone_map   = {str(k): int(v) for k, v in zip(le_zone.classes_,  le_zone.transform(le_zone.classes_))}
    train_map  = {str(k): int(v) for k, v in zip(le_train.classes_, le_train.transform(le_train.classes_))}

    # Also: average delay per train for fast lookup
    train_avg = df.groupby("train_no")["delay"].mean().to_dict()
    stat_avg  = df.groupby("station_name")["delay"].mean().to_dict()

    # Save schedule lookup (train_no -> {station_name -> {dist, hour}})
    sched_lookup = {}
    for _, row in sched_df.iterrows():
        t = str(row["train_no"])
        s = str(row["station_name"])
        sched_lookup.setdefault(t, {})[s] = {
            "station_no": int(row["station_no"]),
            "distance":   float(row["distance_from_origin"]) if not pd.isna(row["distance_from_origin"]) else 0,
            "sched_hour": float(row["scheduled_hour"])       if not pd.isna(row["scheduled_hour"])       else 12,
        }

    meta = {
        "features":    features,
        "type_map":    type_map,
        "zone_map":    zone_map,
        "train_map":   train_map,
        "train_avg":   {k: float(v) for k, v in train_avg.items()},
        "stat_avg":    {k: float(v) for k, v in stat_avg.items()},
        "sched_lookup": sched_lookup,
    }
    with open(os.path.join(MODEL_DIR, "meta.json"), "w") as f:
        json.dump(meta, f)

    print(f"Dataset size after cleaning: {len(X):,} rows")
    return X, y, meta


def train(X, y):
    print("Splitting into train/test …")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.15, random_state=42)

    print("Training GradientBoostingRegressor …  (may take a few minutes)")
    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.8,
        random_state=42,
        verbose=1,
    )
    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_te)
    mae    = mean_absolute_error(y_te, y_pred)
    print(f"\nMAE = {mae:.1f} minutes on test set")
    return model


def main():
    delay_df, sched_df, train_df, stat_df = load_data()
    X, y, meta = preprocess(delay_df, sched_df, train_df, stat_df)
    model = train(X, y)
    model_path = os.path.join(MODEL_DIR, "delay_model.joblib")
    joblib.dump(model, model_path)
    print(f"\nModel saved -> {model_path}")
    print(f"Meta  saved -> {os.path.join(MODEL_DIR, 'meta.json')}")


if __name__ == "__main__":
    main()
