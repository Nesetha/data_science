"""Inspect the extracted FETAL_PLANES_DB directory without assuming its CSV schema."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "raw"

def inspect():
    csvs = list(DATA_DIR.rglob("*.csv"))
    images = [p for p in DATA_DIR.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    print(f"CSV files found: {len(csvs)}")
    print(f"Image files found: {len(images)}")
    for path in csvs:
        print(f"\n--- {path.relative_to(ROOT)} ---")
        try:
            df = pd.read_csv(path, sep=None, engine="python")
            print("Shape:", df.shape)
            print("Columns:", list(df.columns))
            print(df.head(5).to_string(index=False))
        except Exception as exc:
            print("Could not parse:", exc)
    if not csvs:
        print("\nNo CSV found. Inspect the extracted folder and metadata format manually.")

if __name__ == "__main__":
    inspect()
