#!/usr/bin/env python3
# run.py - Orchestrates data fetching, preprocessing, and modeling

from src.fetch_data import download_data
from src.preprocess import prepare_data
from src.model import train_and_evaluate

def main():
    print("🔄 Starting Macroeconomy Signals pipeline…")

    # 1) Fetch raw data from FRED
    download_data()

    # 2) Preprocess into features X and target y
    X, y = prepare_data()

    # 3) Train the regression model and evaluate
    train_and_evaluate(X, y)

    print("✅ Pipeline complete.")

if __name__ == "__main__":
    main()
