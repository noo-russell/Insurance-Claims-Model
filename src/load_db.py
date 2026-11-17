"""Create the SQLite database, load the raw CSVs and build the modelling views."""
from pathlib import Path
import sqlite3

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "insurance.db"
RAW = ROOT / "data" / "raw"

POLICY_COLS = {
    "IDpol": "policy_id", "ClaimNb": "claim_count", "Exposure": "exposure",
    "VehPower": "veh_power", "VehAge": "veh_age", "DrivAge": "driver_age",
    "BonusMalus": "bonus_malus", "VehBrand": "veh_brand", "VehGas": "veh_gas",
    "Area": "area", "Density": "density", "Region": "region",
}


def main():
    con = sqlite3.connect(DB)
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript((ROOT / "sql" / "01_schema.sql").read_text())

    policies = pd.read_csv(RAW / "freMTPL2freq.csv").rename(columns=POLICY_COLS)
    policies["claim_count"] = policies["claim_count"].astype(int)
    policies[list(POLICY_COLS.values())].to_sql("policies", con, if_exists="append", index=False)

    claims = pd.read_csv(RAW / "freMTPL2sev.csv").rename(
        columns={"IDpol": "policy_id", "ClaimAmount": "claim_amount"})
    claims.to_sql("claims", con, if_exists="append", index=False)

    con.executescript((ROOT / "sql" / "04_model_views.sql").read_text())
    con.commit()
    for t in ("policies", "claims"):
        print(t, f"{con.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]:,} rows")
    con.close()
    print("database written to", DB)


if __name__ == "__main__":
    main()
