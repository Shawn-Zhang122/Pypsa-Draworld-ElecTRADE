#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build non-fossil synthetic generators (33-node nodal version)
NO zone layer.
Province & region come ONLY from mapping JSON.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# 1. Canonical output columns
# ============================================================

CANON_COLS = [
    "region_grid","province","node","Resource","technology","cluster","R_ID",
    "COAL","GAS","NUCLEAR","HYDRO","ONWIND","OFFWIND","SOLAR","PUMPED","BATTERY","OTHER",
    "Commit",
    "Existing_Cap_MW","num_units","unmodified_existing_cap_mw","Cap_Size",
    "Var_OM_Cost_per_MWh","Start_Cost_per_MW","Start_Fuel_MMBTU_per_MW",
    "Heat_Rate_MMBTU_per_MWh","Fuel","Min_Power","Eff_Up","Eff_Down",
    "Ramp_Up_Percentage","Ramp_Dn_Percentage",
    "status","start_year","retired_year",
]


# ============================================================
# 2. Defaults
# ============================================================

DEFAULTS = {
    "nuclear":      dict(unit=1000, var_om=2, min_power=0.6, ramp=0.3, fuel="nuclear"),
    "hydro":        dict(unit=300,  var_om=1, min_power=0.0, ramp=1.0, fuel="hydro"),
    "pumped_hydro": dict(unit=300,  var_om=1, min_power=0.0, ramp=1.0, fuel="hydro"),
    "onwind":       dict(unit=200,  var_om=0, min_power=0.0, ramp=1.0, fuel="wind"),
    "offwind":      dict(unit=300,  var_om=0, min_power=0.0, ramp=1.0, fuel="wind"),
    "solar_pv":     dict(unit=100,  var_om=0, min_power=0.0, ramp=1.0, fuel="solar"),
    "battery":      dict(unit=10,   var_om=0, min_power=0.0, ramp=1.0, fuel="battery"),
}

CAP_COL_TO_TECH = {
    "Nuclear": "nuclear",
    "Hydro": "hydro",
    "Onshore Wind": "onwind",
    "Offshore Wind": "offwind",
    "SolarPV": "solar_pv",
    "PumpHydro": "pumped_hydro",
    "Battery": "battery",
}


# ============================================================
# 3. Mapping loader (node only)
# ============================================================

def load_mapping(path: Path) -> pd.DataFrame:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for rg, provs in cfg.items():
        for prov, nodes in provs.items():
            for node in nodes:
                rows.append({
                    "region_grid": rg,
                    "province": prov,
                    "node": node
                })
    df = pd.DataFrame(rows)
    if not df["node"].is_unique:
        raise ValueError("Duplicate nodes in mapping_json")
    return df


# ============================================================
# 4. Read capacity excel
# ============================================================

def read_manual_capacity(path: Path, sheet: str) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=sheet, header=None, engine="openpyxl")
    headers = raw.iloc[1].tolist()
    df = raw.iloc[2:].copy()
    df.columns = headers

    df = df[df["BAs"].notna()].copy()
    df["BAs"] = df["BAs"].astype(str).str.strip()

    for c in CAP_COL_TO_TECH:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    df["Simulation Year"] = pd.to_numeric(df["Simulation Year"], errors="coerce")
    return df


# ============================================================
# 5. Build generators (NO province split)
# ============================================================

def build_generators(df_cap: pd.DataFrame, year: int) -> pd.DataFrame:
    df = df_cap[df_cap["Simulation Year"] == year].copy()
    if df.empty:
        raise ValueError(f"No data for year={year}")

    rows = []

    for _, r in df.iterrows():
        node = r["BAs"]

        for col, tech in CAP_COL_TO_TECH.items():
            cap_gw = float(r[col])
            if cap_gw <= 0:
                continue

            cap_mw_unmod = cap_gw * 1000
            d = DEFAULTS[tech]
            unit = d["unit"]
            n_units = int(max(1, np.ceil(cap_mw_unmod / unit)))
            cap_mw = n_units * unit

            rows.append({
                "node": node,
                "Resource": f"SYN_{tech.upper()}_{node}",
                "technology": tech,
                "cluster": 1,
                "R_ID": f"MANUAL_{tech.upper()}_{node}",

                "COAL": 0,
                "GAS": 0,
                "NUCLEAR": int(tech == "nuclear"),
                "HYDRO": int(tech == "hydro"),
                "ONWIND": int(tech == "onwind"),
                "OFFWIND": int(tech == "offwind"),
                "SOLAR": int(tech == "solar_pv"),
                "PUMPED": int(tech == "pumped_hydro"),
                "BATTERY": int(tech == "battery"),
                "OTHER": 0,

                "Commit": 1,
                "Existing_Cap_MW": cap_mw,
                "unmodified_existing_cap_mw": cap_mw_unmod,
                "num_units": n_units,
                "Cap_Size": unit,

                "Var_OM_Cost_per_MWh": d["var_om"],
                "Start_Cost_per_MW": 0.0,
                "Start_Fuel_MMBTU_per_MW": 0.0,
                "Heat_Rate_MMBTU_per_MWh": 0.0, 
                "Fuel": d["fuel"],
                "Min_Power": d["min_power"],
                "Eff_Up": 1.0,
                "Eff_Down": 1.0,
                "Ramp_Up_Percentage": d["ramp"],
                "Ramp_Dn_Percentage": d["ramp"],

                "status": "synthetic",
                "start_year": np.nan,
                "retired_year": np.nan,
            })

    return pd.DataFrame(rows)


# ============================================================
# 6. CLI
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capacity_xlsx", default="data/raw/Calibration_Capacity_nonfossilfuel_mannualcollection-2025.xlsx")
    ap.add_argument("--mapping_json", default="config/region_province_node_33.json")
    ap.add_argument("--sheet", default="cap0_2025")
    ap.add_argument("--year", type=int, default=2025)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]

    df_cap = read_manual_capacity(root / args.capacity_xlsx, args.sheet)
    mapping = load_mapping(root / args.mapping_json)

    gen = build_generators(df_cap, args.year)

    # merge ONLY on node
    gen = gen.merge(mapping, on="node", how="left")

    missing = gen[gen["region_grid"].isna()][["node"]].drop_duplicates()
    if not missing.empty:
        raise ValueError("Mapping missing for nodes:\n" + missing.to_string(index=False))

    for c in CANON_COLS:
        if c not in gen.columns:
            gen[c] = np.nan

    gen = gen[CANON_COLS]

    out = root / "data/generators/nonfossil_generators_units.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    gen.to_csv(out, index=False)

    print("OK: wrote", out)


if __name__ == "__main__":
    main()
