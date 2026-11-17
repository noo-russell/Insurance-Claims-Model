"""Download freMTPL2 (French motor third-party liability) and save it as CSV.

Source: CASdatasets R package (Dutang & Charpentier), mirrored on GitHub.
Two tables: one row per policy (freq) and one row per claim (sev).
"""
from pathlib import Path
import urllib.request

import pyreadr

BASE = "https://raw.githubusercontent.com/dutangc/CASdatasets/master/data/"
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    for name in ("freMTPL2freq", "freMTPL2sev"):
        rda = RAW / f"{name}.rda"
        if not rda.exists():
            print(f"downloading {name} ...")
            urllib.request.urlretrieve(BASE + rda.name, rda)
        df = pyreadr.read_r(str(rda))[name]
        df["IDpol"] = df["IDpol"].astype(str).astype(float).astype("int64")
        df.to_csv(RAW / f"{name}.csv", index=False)
        print(f"{name}: {len(df):,} rows -> {RAW / (name + '.csv')}")


if __name__ == "__main__":
    main()
