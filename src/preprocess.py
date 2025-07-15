# src/preprocess.py

import os
import pandas as pd

def prepare_data():
    """
    Load raw CSVs, aggregate monthly unemployment into quarterly,
    align with quarterly GDP growth, and return (X, y).
    """
    # Paths (relative to project root)
    raw_dir = os.path.join(os.path.dirname(__file__), os.pardir, "data", "raw")
    unemp_csv = os.path.abspath(os.path.join(raw_dir, "unemployment.csv"))
    gdp_csv   = os.path.abspath(os.path.join(raw_dir, "gdp_growth.csv"))

    # Load CSVs
    unemp = pd.read_csv(unemp_csv, parse_dates=["date"])
    gdp   = pd.read_csv(gdp_csv, parse_dates=["date"])

    # --- 1) Aggregate unemployment into quarterly averages ---
    unemp["quarter"] = unemp["date"].dt.to_period("Q")
    unemp_q = (
        unemp
        .groupby("quarter")["value"]
        .mean()
        .reset_index()
        .rename(columns={"value": "avg_unemployment"})
    )

    # --- 2) Prepare GDP growth series by quarter ---
    gdp["quarter"] = gdp["date"].dt.to_period("Q")
    gdp_q = (
        gdp[["quarter", "value"]]
        .rename(columns={"value": "gdp_growth"})
    )

    # --- 3) Merge on quarter and drop any NaNs ---
    data = pd.merge(unemp_q, gdp_q, on="quarter").dropna()

    # --- 4) Build feature matrix X and target y ---
    X = data[["avg_unemployment"]].astype(float)
    y = data["gdp_growth"].astype(float)

    print(f"Prepared data: {len(data)} quarters  →  X.shape={X.shape}, y.shape={y.shape}")
    return X, y

# Allow standalone testing
if __name__ == "__main__":
    prepare_data()
