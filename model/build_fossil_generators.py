#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_generators.py

Unified generator builder for China Daily ElecTRADE (33-node DA model).
Strict version: fail-fast, no silent fixes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ============================================================
# 0. Constants
# ============================================================

PROVINCE_NAME_FIX = {
    "Nei Mongol": "Inner Mongolia",
    "Neimenggu": "Inner Mongolia",
    "Xinjiang Uygur": "Xinjiang",
    "Ningxia Hui": "Ningxia",
    "Guangxi Zhuang": "Guangxi",
    "Tibet Autonomous Region": "Tibet",
}

DEFAULT_HEAT_RATE_COAL = {
    "ultra_supercritical_coal": 8.8,
    "supercritical_coal": 9.4,
    "conventional_steam_coal": 10.3,
    "igcc_coal": 8.5,
}

FOSSIL_SYNTH_DEFAULTS = {
    "gas": dict(unit_size=600.0, heat_rate=7.2, var_om=4.0, start_cost=20.0, min_power=0.30, ramp=0.80),
    "oil": dict(unit_size=300.0, heat_rate=10.5, var_om=6.0, start_cost=40.0, min_power=0.40, ramp=0.60),
}

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
# 1. Utilities
# ============================================================

def normalize_str(x) -> str:
    return "" if pd.isna(x) else "".join(str(x).strip().split())

def standardize_province(x: str) -> str:
    return PROVINCE_NAME_FIX.get(str(x), str(x))

def heat_rate_to_efficiency(hr: float) -> float:
    return 3.412 / hr if hr and hr > 0 else np.nan

def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


# ============================================================
# 2. Config loaders
# ============================================================

def load_splits(path: Path) -> dict:
    return load_json(path)

def load_mapping(path: Path) -> pd.DataFrame:
    cfg = load_json(path)
    rows = []
    for rg, provs in cfg.items():
        for prov, pdata in provs.items():
            for zone, zdata in pdata["zones"].items():
                for node in zdata["nodes"]:
                    rows.append(dict(region_grid=rg, province=prov, Zone=zone, node=node))
    df = pd.DataFrame(rows)
    if not df["node"].is_unique:
        raise ValueError("Duplicate nodes in mapping")
    return df


# ============================================================
# 3. GEM loading & validation
# ============================================================

def load_gem(path: Path, sheet="Units") -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
    df.columns = df.columns.str.strip()

    if "Country/Area" not in df.columns:
        raise KeyError("Missing column: Country/Area")

    return df[df["Country/Area"].str.lower() == "china"].copy()


def filter_operating(df: pd.DataFrame, target_date: str) -> pd.DataFrame:
    required = ["Status", "Start year", "Retired year"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"GEM missing columns: {missing}")

    year = pd.Timestamp(target_date).year
    out = df.copy()

    out["Start year"] = pd.to_numeric(out["Start year"], errors="coerce")
    out["Retired year"] = pd.to_numeric(out["Retired year"], errors="coerce")

    return out[
        (out["Status"].str.lower() == "operating") &
        (out["Start year"] <= year) &
        (out["Retired year"].isna() | (out["Retired year"] > year))
    ]


# ============================================================
# 4. Geography
# ============================================================

def assign_node(province: str, prefecture: str, splits: dict) -> str:
    prov = standardize_province(province)
    pref = normalize_str(prefecture)

    if prov in splits:
        for k, plist in splits[prov].items():
            if pref in map(normalize_str, plist):
                return f"{prov}_{k}"
    return prov


def add_geo_fields(df: pd.DataFrame, splits: dict) -> pd.DataFrame:
    # --- province column (coal vs oil/gas schema)
    if "Subnational unit (province, state)" in df.columns:
        prov_col = "Subnational unit (province, state)"
    elif "State/Province" in df.columns:
        prov_col = "State/Province"
    else:
        raise KeyError(
            "Missing province column. Expected one of: "
            "'Subnational unit (province, state)' or 'State/Province'"
        )

    # --- prefecture column (shared)
    pref_col = "Major area (prefecture, district)"
    if pref_col not in df.columns:
        raise KeyError(f"Missing column: {pref_col}")

    out = df.copy()
    out["province"] = out[prov_col].map(standardize_province)
    out["prefecture"] = out[pref_col]

    out["node"] = out.apply(
        lambda r: assign_node(r["province"], r["prefecture"], splits),
        axis=1,
    )
    return out



# ============================================================
# 5. Coal (unit-level)
# ============================================================

def map_coal_tech(x) -> str:
    s = str(x).lower()
    if "ultra" in s and "supercritical" in s:
        return "ultra_supercritical_coal"
    if "supercritical" in s:
        return "supercritical_coal"
    if "igcc" in s:
        return "igcc_coal"
    return "conventional_steam_coal"


def build_coal(df_raw: pd.DataFrame, splits: dict) -> pd.DataFrame:
    df = add_geo_fields(df_raw, splits)

    if "Capacity (MW)" not in df.columns:
        raise KeyError("Missing column: Capacity (MW)")

    df["Existing_Cap_MW"] = pd.to_numeric(df["Capacity (MW)"], errors="coerce")
    df = df[df["Existing_Cap_MW"] > 0]

    df["technology"] = df["Combustion technology"].map(map_coal_tech)
    df["Heat_Rate_MMBTU_per_MWh"] = df["technology"].map(DEFAULT_HEAT_RATE_COAL)
    eff = df["Heat_Rate_MMBTU_per_MWh"].map(heat_rate_to_efficiency)

    return pd.DataFrame({
        "province": df["province"],
        "node": df["node"],
        "Resource": df["Plant name"] + " " + df["Unit name"],
        "technology": df["technology"],
        "cluster": 1,
        "R_ID": df["GEM unit/phase ID"],
        "COAL": 1, "GAS": 0, "OTHER": 0,
        "Commit": 1,
        "Existing_Cap_MW": df["Existing_Cap_MW"],
        "unmodified_existing_cap_mw": df["Existing_Cap_MW"],
        "num_units": 1,
        "Cap_Size": df["Existing_Cap_MW"],
        "Var_OM_Cost_per_MWh": 5.0,
        "Start_Cost_per_MW": 30.0,
        "Start_Fuel_MMBTU_per_MW": 0.0,
        "Heat_Rate_MMBTU_per_MWh": df["Heat_Rate_MMBTU_per_MWh"],
        "Fuel": "coal",
        "Min_Power": 0.40,
        "Eff_Up": eff,
        "Eff_Down": eff,
        "Ramp_Up_Percentage": 0.5,
        "Ramp_Dn_Percentage": 0.5,
        "status": df["Status"],
        "start_year": df["Start year"],
        "retired_year": df["Retired year"],
    })


# ============================================================
# 6. Gas / Oil synthetic
# ============================================================

def classify_fuel(x: str) -> Optional[str]:
    s = str(x).lower()
    if "gas" in s:
        return "gas"
    if "oil" in s or "diesel" in s:
        return "oil"
    return None

def is_split_province(prov: str, splits: dict) -> bool:
    return prov in splits and len(splits[prov]) > 1

def build_synth(df_raw: pd.DataFrame, splits: dict, fuel: str) -> pd.DataFrame:
    p = FOSSIL_SYNTH_DEFAULTS[fuel]

    # --- GEO
    df = add_geo_fields(df_raw, splits)

    # --- REQUIRED COLUMNS
    for col in ["Fuel", "Capacity (MW)", "Turbine/Engine Technology"]:
        if col not in df.columns:
            raise KeyError(f"Missing column: {col}")

    # ============================================================
    # 1. READ-IN CAPACITY (UNMODIFIED)
    # ============================================================
    df["unmodified_existing_cap_mw"] = pd.to_numeric(
        df["Capacity (MW)"], errors="coerce"
    )
    df = df[df["unmodified_existing_cap_mw"] > 0]

    # --- FUEL FILTER
    df = df[df["Fuel"].map(classify_fuel) == fuel]

    # ============================================================
    # 1.5 TECHNOLOGY FILTER (INLINE, NO EXTRA FUNCTION)
    # ============================================================
    tech = (
        df["Turbine/Engine Technology"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df["technology"] = np.where(
        tech == "gas turbine", "gas_turbine",
        np.where(
            tech == "combined cycle", "combined_cycle",
            "unknown_gas"
        )
    )

    df = df[df["technology"].notna()]

    # ============================================================
    # 2. RESOLVE SPLIT PROVINCES BEFORE AGGREGATION
    # ============================================================
    rows = []
    required_cols = df.columns.tolist()

    for _, r in df.iterrows():
        prov = r["province"]
        node = r["node"]
        cap  = r["unmodified_existing_cap_mw"]

        base = {c: r[c] for c in required_cols}

        if is_split_province(prov, splits) and node == prov:
            sub_nodes = list(splits[prov].keys())
            share = cap / len(sub_nodes)

            for sn in sub_nodes:
                rr = base.copy()
                rr["node"] = f"{prov}_{sn}"
                rr["unmodified_existing_cap_mw"] = share
                rows.append(rr)
        else:
            rows.append(base)

    df = pd.DataFrame(rows, columns=required_cols)

    # ============================================================
    # 3. AGGREGATE (NODE × TECHNOLOGY)
    # ============================================================
    agg = (
        df.groupby(
            ["province", "node", "technology"], as_index=False
        )
        .agg(unmodified_existing_cap_mw=("unmodified_existing_cap_mw", "sum"),
             R_ID=("GEM unit ID", "first")
             )
        
    )

    # ============================================================
    # 4. UNITIZATION (MODEL CAPACITY)
    # ============================================================
    agg["num_units"] = np.ceil(
        agg["unmodified_existing_cap_mw"] / p["unit_size"]
    ).astype(int).clip(lower=1)

    agg["Cap_Size"] = p["unit_size"]
    agg["Existing_Cap_MW"] = agg["num_units"] * agg["Cap_Size"]

    eff = heat_rate_to_efficiency(p["heat_rate"])

    # ============================================================
    # 5. OUTPUT
    # ============================================================
    return pd.DataFrame({
        "province": agg["province"],
        "node": agg["node"],

        "Resource": (
            fuel.upper()
            + "_"
            + agg["technology"].str.upper()
            + "_SYNTH_"
            + agg["node"]
        ),

        "technology": agg["technology"],
        "cluster": 1,
        "R_ID": agg["R_ID"],

        "COAL": 0,
        "GAS": int(fuel == "gas"),
        "OTHER": int(fuel == "oil"),

        "Commit": 1,
        "Existing_Cap_MW": agg["Existing_Cap_MW"],
        "unmodified_existing_cap_mw": agg["unmodified_existing_cap_mw"],
        "num_units": agg["num_units"],
        "Cap_Size": agg["Cap_Size"],

        "Var_OM_Cost_per_MWh": p["var_om"],
        "Start_Cost_per_MW": p["start_cost"],
        "Start_Fuel_MMBTU_per_MW": 0.0,

        "Heat_Rate_MMBTU_per_MWh": p["heat_rate"],
        "Fuel": fuel,
        "Min_Power": p["min_power"],
        "Eff_Up": eff,
        "Eff_Down": eff,

        "Ramp_Up_Percentage": p["ramp"],
        "Ramp_Dn_Percentage": p["ramp"],

        "status": "synthetic",
        "start_year": np.nan,
        "retired_year": np.nan,
    })


# ============================================================
# 7. Mapping / output
# ============================================================

def attach_mapping(df: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    out = df.merge(mapping, on=["province","node"], how="left")
    if out["Zone"].isna().any():
        raise ValueError(out[out["Zone"].isna()][["province","node"]])
    return out


def enforce_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in CANON_COLS:
        if c not in out.columns:
            out[c] = np.nan
    return out[CANON_COLS]


# ============================================================
# 8. Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--coal_xlsx",
        default="data/raw/Global-Coal-Plant-Tracker-July-2025.xlsx",
        help="GEM coal tracker"
    )
    ap.add_argument(
        "--gogpt_xlsx",
        default="data/raw/Global-Oil-and-Gas-Plant-Tracker-GOGPT-August-2025.xlsx",
        help="GEM oil & gas tracker"
    )
    ap.add_argument(
        "--splits_json",
        default="config/splits_33nodes.json"
    )
    ap.add_argument(
        "--mapping_json",
        default="config/region_province_node_33.json"
    )
    ap.add_argument(
        "--target_date",
        default="2026-01-01"
    )

    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]

    splits = load_splits(root / args.splits_json)
    mapping = load_mapping(root / args.mapping_json)

    coal = build_coal(
        filter_operating(load_gem(root / args.coal_xlsx), args.target_date),
        splits
    )

    gas = build_synth(
        filter_operating(load_gem(root / args.gogpt_xlsx), args.target_date),
        splits,
        "gas"
    )

    oil = build_synth(
        filter_operating(load_gem(root / args.gogpt_xlsx), args.target_date),
        splits,
        "oil"
    )

    coal = enforce_cols(attach_mapping(coal, mapping))
    gas  = enforce_cols(attach_mapping(gas, mapping))
    oil  = enforce_cols(attach_mapping(oil, mapping))

    all_units = pd.concat([coal, gas, oil], ignore_index=True)

    cap = (
        all_units
        .pivot_table(
            index="node",
            columns="technology",
            values="Existing_Cap_MW",
            aggfunc="sum",
            fill_value=0
        )
        .reset_index()
    )

    out_g = root / "data/generators"
    out_c = root / "data/capacities"
    out_g.mkdir(parents=True, exist_ok=True)
    out_c.mkdir(parents=True, exist_ok=True)

    coal.to_csv(out_g / "coal_units.csv", index=False)
    gas.to_csv(out_g / "gas_units_synth.csv", index=False)
    oil.to_csv(out_g / "oil_units_synth.csv", index=False)
    all_units.to_csv(out_g / "fossil_generators_units.csv", index=False)
    cap.to_csv(out_c / "fossil_capacity_by_node_tech.csv", index=False)

    print("Coal / gas / oil generators built successfully.")


if __name__ == "__main__":
    main()
