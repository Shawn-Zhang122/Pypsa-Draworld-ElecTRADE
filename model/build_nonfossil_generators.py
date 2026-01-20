#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_nonfossil_generators.py

Build non-fossil generators + capacity-by-node-tech in the SAME "uniform" format
as your fossil synthetic generators.

Source (manual capacity collection, unit = GW):
  Calibration_Capacity_nonfossilfuel_mannualcollection-2025.xlsx
  sheet: cap0_2025

Technologies used (from your headers):
  Nuclear, Hydro, Onshore Wind, Offshore Wind, SolarPV, PumpHydro, Battery

Key behaviors:
- Treat column "BAs" as your model "node" directly (it already includes split nodes like Hebei_HEBEI_NORTH).
- Convert GW -> MW.
- Create one synthetic row per (node, technology) with unitization:
    num_units = ceil(cap_mw / unit_size_mw), Cap_Size = unit_size_mw, Existing_Cap_MW = num_units*Cap_Size
- HYDRO and PUMPED are separated:
    HYDRO flag only for hydro
    PUMPED flag only for pumped_hydro

Outputs:
- data/generators/nonfossil_generators_units.csv   (same column schema as fossil)
- data/capacities/nonfossil_capacity_by_node_tech.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 0. Canonical output columns (match fossil generator schema)
# ============================================================

CANON_COLS = [
    "region_grid", "province", "Zone", "node",
    "Resource", "technology", "cluster", "R_ID",
    "COAL", "GAS", "NUCLEAR", "HYDRO", "ONWIND", "OFFWIND", "SOLAR", "PUMPED", "BATTERY", "OTHER",
    "Commit",
    "Existing_Cap_MW", "num_units", "unmodified_existing_cap_mw", "Cap_Size",
    "Var_OM_Cost_per_MWh", "Start_Cost_per_MW", "Start_Fuel_MMBTU_per_MW",
    "Heat_Rate_MMBTU_per_MWh", "Fuel", "Min_Power", "Eff_Up", "Eff_Down",
    "Ramp_Up_Percentage", "Ramp_Dn_Percentage",
    "status", "start_year", "retired_year",
]


# ============================================================
# 1. Defaults for synthetic unitization (edit if you have priors)
# ============================================================

DEFAULTS = {
    "nuclear":      dict(unit_size_mw=1000.0, var_om=2.0, start_cost=10.0, min_power=0.60, ramp=0.30, fuel="nuclear"),
    "hydro":        dict(unit_size_mw=300.0,  var_om=1.0, start_cost=0.0,  min_power=0.00, ramp=1.00, fuel="hydro"),
    "pumped_hydro": dict(unit_size_mw=300.0,  var_om=1.0, start_cost=0.0,  min_power=0.00, ramp=1.00, fuel="hydro"),
    "onwind":       dict(unit_size_mw=200.0,  var_om=0.0, start_cost=0.0,  min_power=0.00, ramp=1.00, fuel="wind"),
    "offwind":      dict(unit_size_mw=300.0,  var_om=0.0, start_cost=0.0,  min_power=0.00, ramp=1.00, fuel="wind"),
    "solar_pv":     dict(unit_size_mw=100.0,  var_om=0.0, start_cost=0.0,  min_power=0.00, ramp=1.00, fuel="solar"),
    "battery":      dict(unit_size_mw=10.0,  var_om=0.0, start_cost=0.0,  min_power=0.00, ramp=1.00, fuel="battery"),
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
# 2. Mapping loader (same JSON structure you used in fossil builder)
# ============================================================

def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))

def load_mapping(mapping_json: Path) -> pd.DataFrame:
    cfg = load_json(mapping_json)
    rows = []
    for rg, provs in cfg.items():
        for prov, pdata in provs.items():
            for zone, zdata in pdata["zones"].items():
                for node in zdata["nodes"]:
                    rows.append(dict(region_grid=rg, province=prov, Zone=zone, node=node))
    df = pd.DataFrame(rows)
    if not df["node"].is_unique:
        raise ValueError("Duplicate nodes in mapping_json")
    return df


# ============================================================
# 3. Read manual capacity excel (robust to the 2-row header style)
# ============================================================

def read_manual_capacity_xlsx(path: Path, sheet: str) -> pd.DataFrame:
    """
    The file is in a 2-row header layout:
      row0: metadata
      row1: actual headers
      row2+: data

    Returns a clean table with columns like:
      Simulation Year, BAs, Nuclear, Hydro, Onshore Wind, ...
    """
    raw = pd.read_excel(path, sheet_name=sheet, header=None, engine="openpyxl")
    if raw.shape[0] < 3:
        raise ValueError("Unexpected manual capacity sheet shape; needs >= 3 rows")

    headers = raw.iloc[1].tolist()
    df = raw.iloc[2:].copy()
    df.columns = headers

    # drop blank node rows
    df = df[df["BAs"].notna()].copy()

    # numeric coercion
    for c in df.columns:
        if c in ["Simulation Year", "BAs"]:
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    df["Simulation Year"] = pd.to_numeric(df["Simulation Year"], errors="coerce")
    df["BAs"] = df["BAs"].astype(str).str.strip()
    return df


# ============================================================
# 4. Build synthetic generators from capacity table
# ============================================================

def build_nonfossil_from_capacity(df_cap: pd.DataFrame, target_year: int) -> pd.DataFrame:
    """
    Produce one row per (node, technology).
    Input unit is GW -> output MW.
    """
    df = df_cap.copy()
    df = df[df["Simulation Year"] == target_year].copy()
    if df.empty:
        raise ValueError(f"No rows found for Simulation Year={target_year}")

    rows = []
    for _, r in df.iterrows():
        node = r["BAs"]

        # province guess: substring before "_" if present, else node itself.
        province = node.split("_", 1)[0] if "_" in node else node

        for cap_col, tech in CAP_COL_TO_TECH.items():
            if cap_col not in df.columns:
                raise KeyError(f"Missing capacity column in excel: {cap_col}")

            cap_gw = float(r[cap_col])
            if cap_gw <= 0:
                continue

            cap_mw_unmod = cap_gw * 1000.0
            d = DEFAULTS[tech]
            unit = float(d["unit_size_mw"])
            n = int(max(1, np.ceil(cap_mw_unmod / unit)))
            cap_mw = n * unit

            rows.append({
                "province": province,
                "node": node,

                "Resource": f"NONFOSSIL_{tech.upper()}_SYNTH_{node}",
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
                "num_units": n,
                "Cap_Size": unit,

                "Var_OM_Cost_per_MWh": float(d["var_om"]),
                "Start_Cost_per_MW": float(d["start_cost"]),
                "Start_Fuel_MMBTU_per_MW": 0.0,

                "Heat_Rate_MMBTU_per_MWh": np.nan,
                "Fuel": d["fuel"],
                "Min_Power": float(d["min_power"]),
                "Eff_Up": np.nan,
                "Eff_Down": np.nan,

                "Ramp_Up_Percentage": float(d["ramp"]),
                "Ramp_Dn_Percentage": float(d["ramp"]),

                "status": "synthetic",
                "start_year": np.nan,     # manual file gives only capacity snapshot
                "retired_year": np.nan,
            })

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("All nonfossil capacities are zero after filtering; nothing to export.")
    return out


def attach_mapping(df: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    out = df.merge(mapping, on=["province", "node"], how="left")

    missing = out[out["Zone"].isna()][["province", "node"]].drop_duplicates()
    if not missing.empty:
        raise ValueError(
            "Some (province,node) not found in mapping_json. Missing:\n"
            + missing.to_string(index=False)
        )
    return out


def enforce_canon_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in CANON_COLS:
        if c not in out.columns:
            out[c] = np.nan
    return out[CANON_COLS]


def build_capacity_by_node_tech(df_gen: pd.DataFrame) -> pd.DataFrame:
    cap = (
        df_gen.pivot_table(
            index="node",
            columns="technology",
            values="Existing_Cap_MW",
            aggfunc="sum",
            fill_value=0.0,
        )
        .reset_index()
    )
    return cap


# ============================================================
# 5. CLI
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--capacity_xlsx",
        default="data/raw/Calibration_Capacity_nonfossilfuel_mannualcollection-2025.xlsx",
        help="Manual nonfossil capacity snapshot (GW)",
    )
    ap.add_argument("--sheet", default="cap0_2025")
    ap.add_argument("--mapping_json", default="config/region_province_zone_node_33.json")
    ap.add_argument("--target_year", type=int, default=2025)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]

    df_cap = read_manual_capacity_xlsx(root / args.capacity_xlsx, args.sheet)
    mapping = load_mapping(root / args.mapping_json)

    gen = build_nonfossil_from_capacity(df_cap, target_year=args.target_year)
    gen = attach_mapping(gen, mapping)
    gen = enforce_canon_cols(gen)

    cap = build_capacity_by_node_tech(gen)

    out_g = root / "data/generators"
    out_c = root / "data/capacities"
    out_g.mkdir(parents=True, exist_ok=True)
    out_c.mkdir(parents=True, exist_ok=True)

    gen.to_csv(out_g / "nonfossil_generators_units.csv", index=False)
    cap.to_csv(out_c / "nonfossil_capacity_by_node_tech.csv", index=False)

    print("OK: wrote",
          out_g / "nonfossil_generators_units.csv",
          "and",
          out_c / "nonfossil_capacity_by_node_tech.csv")


if __name__ == "__main__":
    main()

# End of file
