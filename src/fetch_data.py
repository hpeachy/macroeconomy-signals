# src/fetch_data.py

import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("FRED_API_KEY")
BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

def download_data():
    """
    Download unemployment and GDP-growth series from FRED
    and save them as CSVs under data/raw/.
    """
    os.makedirs("data/raw", exist_ok=True)

    series_map = {
        "UNRATE": "unemployment.csv",                # Monthly unemployment rate
        "A191RL1Q225SBEA": "gdp_growth.csv"          # Quarterly real GDP % change
    }

    for series_id, filename in series_map.items():
        params = {
            "series_id": series_id,
            "api_key": API_KEY,
            "file_type": "json"
        }
        resp = requests.get(BASE_URL, params=params)
        resp.raise_for_status()
        observations = resp.json().get("observations", [])

        # Convert to DataFrame and write out
        df = pd.DataFrame(observations)
        df.to_csv(os.path.join("data/raw", filename), index=False)
        print(f"Downloaded {series_id} → data/raw/{filename}")

if __name__ == "__main__":
    download_data()
