"""
Annual economic dispatch (LP), 8760 hours, 33 nodes
- PyPSA + Linopy backend
- Coal marginal cost computed on-the-fly via build_fuel_cost.py
- OFFWIND != ONWIND
- RE profiles already node-specific (no broadcasting)
- Storage modeled as Store + (charge/discharge) Links with fixed durations:
    battery = 2.3h, pumped hydro = 8h
"""

import pandas as pd
import numpy as np
import pypsa

from build_fuel_cost import hourly_index, build_marginal_cost


# =======================
# CONFIG
# =======================
YEAR = 2025

LOAD_CSV   = "data/inputs/Load/Load_NDRC_BAs_China_Draworld_normalised_2025compiled.csv"
EDGES_CSV  = "data/inputs/Network/edges_33nodes_500kVplus_updates_Jan2026.csv"
GEN_CSV    = "data/inputs/Generator/generators_units.csv"

ONWIND_CSV  = "data/inputs/REprofile/ninja_wind_29.0000_120.0000.csv"
OFFWIND_CSV = "data/inputs/REprofile/ninja_wind_29.0000_120.0000.csv"
SOLAR_CSV   = "data/inputs/REprofile/ninja_pv_29.0000_120.0000.csv"

OTHERS_SETTING_CSV = "data/raw/others_setting.csv"

OUT_PRICE = "results/prices_2025_8760.csv"
OUT_FLOW  = "results/flows_2025_8760.csv"


# =======================
# SOLVER (Linopy)
# =======================
solver_options = {
    "Threads": 4,
    "TimeLimit": 28800,
    "Presolve": 2,
    "NumericFocus": 2,
}


# =======================
# STORAGE PARAMETERS (implicit by carrier)
# =======================
STORAGE_TECH = {
    "battery": {"hours": 2.3, "eta_chg": 0.95, "eta_dis": 0.95},
    "pumped":  {"hours": 8.0, "eta_chg": 0.90, "eta_dis": 0.90},
}


# =======================
# CARRIERS (define once)
# =======================
# One electricity carrier for the main grid buses
ELECTRICITY_CARRIER = "electricity"

# Generator carriers that may appear from infer_carrier()
GEN_CARRIERS = {
    "coal", "gas", "nuclear", "hydro",
    "onwind", "offwind", "solar",
    "other",
}

# Network link carriers
NETWORK_CARRIERS = {"ac", "dc"}

# Storage carriers (must match STORAGE_TECH keys)
STORAGE_CARRIERS = set(STORAGE_TECH.keys())

# Storage link carriers used in Store+(charge/discharge) Links
STORAGE_LINK_CARRIERS = (
    {f"{c}_charge" for c in STORAGE_CARRIERS}
    | {f"{c}_discharge" for c in STORAGE_CARRIERS}
)

# All carriers used anywhere in this script
ALL_CARRIERS = (
    {ELECTRICITY_CARRIER}
    | GEN_CARRIERS
    | NETWORK_CARRIERS
    | STORAGE_CARRIERS
    | STORAGE_LINK_CARRIERS
)


# =======================
# BASIC HELPERS
# =======================
def norm(x):
    return str(x).strip()


def read_ts(path, idx, nodes):
    df = pd.read_csv(path)
    t = pd.to_datetime(df["datetime"])
    df = df.drop(columns=["datetime"])
    df.columns = [norm(c) for c in df.columns]
    df.index = t
    df = df.reindex(idx)
    return df[nodes].astype(float)


def infer_carrier(row):
    flags = [
        ("COAL", "coal"),
        ("GAS", "gas"),
        ("NUCLEAR", "nuclear"),
        ("HYDRO", "hydro"),
        ("ONWIND", "onwind"),
        ("OFFWIND", "offwind"),
        ("SOLAR", "solar"),
        ("PUMPED", "pumped"),
        ("BATTERY", "battery"),
        ("OTHER", "other"),
    ]
    for c, v in flags:
        if c in row and pd.notna(row[c]) and float(row[c]) > 0.5:
            return v
    if "Fuel" in row and isinstance(row["Fuel"], str):
        f = row["Fuel"].lower()
        if "coal" in f:
            return "coal"
        if "gas" in f:
            return "gas"
    return "other"


def efficiency(row, carrier):
    if "Eff_Up" in row and pd.notna(row["Eff_Up"]):
        return float(row["Eff_Up"])
    if "Heat_Rate_MMBTU_per_MWh" in row and pd.notna(row["Heat_Rate_MMBTU_per_MWh"]):
        return 3.412 / float(row["Heat_Rate_MMBTU_per_MWh"])
    if carrier == "coal":
        return 0.35
    if carrier == "gas":
        return 0.50
    return 1.0


# =======================
# BUILD NETWORK
# =======================
idx = hourly_index(YEAR)

gen = pd.read_csv(GEN_CSV)
gen["node"] = gen["node"].map(norm)
nodes = sorted(gen["node"].unique())

load = read_ts(LOAD_CSV, idx, nodes)
onwind_cf  = read_ts(ONWIND_CSV,  idx, nodes)
offwind_cf = read_ts(OFFWIND_CSV, idx, nodes)
solar_cf   = read_ts(SOLAR_CSV,   idx, nodes)

edges = pd.read_csv(EDGES_CSV)
edges["from_node"] = edges["from_node"].map(norm)
edges["to_node"]   = edges["to_node"].map(norm)
edges["ac_dc"]     = edges["ac_dc"].str.upper().str.strip()

n = pypsa.Network()
n.set_snapshots(idx)

# -----------------------
# Register carriers ONCE (before adding components that reference carriers)
# -----------------------
for c in sorted(ALL_CARRIERS):
    if c not in n.carriers.index:
        n.add("Carrier", c)

# buses
for b in nodes:
    n.add("Bus", b, carrier=ELECTRICITY_CARRIER)


# loads
for b in nodes:
    n.add("Load", f"load_{b}", bus=b)
n.loads_t.p_set = load.rename(columns={b: f"load_{b}" for b in nodes})
n.loads_t.p_set = n.loads_t.p_set.fillna(0.0)

# -----------------------
# Inter-node connections
# AC and DC both as Links
# but with different directionality
# -----------------------
for i, r in edges.iterrows():
    cap = float(r["capacity_mw"])
    if cap <= 0:
        continue

    acdc = r["ac_dc"].strip().upper()
    name = f"link_{i}_{r['from_node']}_{r['to_node']}"

    if acdc == "DC":
        # DC: single direction (from_node -> to_node)
        n.add(
            "Link",
            name,
            bus0=r["from_node"],
            bus1=r["to_node"],
            p_nom=cap,
            p_min_pu=0.0,
            efficiency=1.0,
            carrier="dc",
        )
    else:
        # AC: bidirectional
        n.add(
            "Link",
            name,
            bus0=r["from_node"],
            bus1=r["to_node"],
            p_nom=cap,
            p_min_pu=-1.0,
            efficiency=1.0,
            carrier="ac",
        )

# -----------------------
# Generators + Storage
# -----------------------
gen = gen.reset_index(drop=False)
gen["carrier"] = gen.apply(infer_carrier, axis=1)

# 1) Add "real" generators (exclude storage carriers)
for _, r in gen.iterrows():
    carrier = r["carrier"]
    if carrier in STORAGE_TECH:
        continue

    p_nom = float(r["Existing_Cap_MW"])
    if p_nom <= 0:
        continue

    eff = efficiency(r, carrier)
    rid = int(r["index"])
    name = f"gen__{rid}"

    p_min_pu = 0.0
    if carrier == "coal" and "Min_Power" in r and pd.notna(r["Min_Power"]):
        p_min_pu = float(r["Min_Power"])

    n.add(
        "Generator",
        name,
        bus=r["node"],
        carrier=carrier,
        p_nom=p_nom,
        efficiency=eff,
        p_min_pu=p_min_pu,
        p_max_pu=1.0
        )

# 2) Add storage as Store + (charge/discharge) Links, implicit by carrier params
for _, r in gen.iterrows():
    carrier = r["carrier"]
    if carrier not in STORAGE_TECH:
        continue

    node = r["node"]
    p_nom = float(r["Existing_Cap_MW"])
    if p_nom <= 0:
        continue

    params = STORAGE_TECH[carrier]
    hours = params["hours"]
    eta_c = params["eta_chg"]
    eta_d = params["eta_dis"]

    e_nom = p_nom * hours

    rid = int(r["index"])
    # storage energy must live on a BUS (Links connect buses, not Stores)
    s_bus   = f"{node}__{carrier}__bus__{rid}"
    s_store = f"{node}__{carrier}__store__{rid}"
    s_chg   = f"{node}__{carrier}__chg__{rid}"
    s_dis   = f"{node}__{carrier}__dis__{rid}"

    # storage bus
    n.add("Bus", s_bus, carrier=carrier)

    # energy store (MWh)
    n.add(
        "Store",
        s_store,
        bus=s_bus,
        e_nom=e_nom,
        e_initial=0.5 * e_nom,
        e_cyclic=True,
        carrier=carrier,
    )

    # charging: grid(node) -> storage bus
    n.add(
        "Link",
        s_chg,
        bus0=node,
        bus1=s_bus,
        p_nom=p_nom,
        p_min_pu=0.0,
        efficiency=eta_c,
        carrier=f"{carrier}_charge",
    )

    # discharging: storage bus -> grid(node)
    n.add(
        "Link",
        s_dis,
        bus0=s_bus,
        bus1=node,
        p_nom=p_nom,
        p_min_pu=0.0,
        efficiency=eta_d,
        carrier=f"{carrier}_discharge",
    )

# -----------------------
# RE availability (node-specific, no broadcasting)
# -----------------------
n.generators_t.p_max_pu = pd.DataFrame(1.0, index=n.snapshots, columns=n.generators.index)

for carrier, cf in {
    "onwind": onwind_cf,
    "offwind": offwind_cf,
    "solar": solar_cf,
    }.items():

    gens = n.generators.index[n.generators.carrier == carrier]
    if len(gens) == 0:
        continue
    mat = np.column_stack([cf[n.generators.at[g, "bus"]].values for g in gens])
    n.generators_t.p_max_pu.loc[:, gens] = mat.clip(0.0, 1.0).fillna(1.0)


# =======================
# MARGINAL COST
# =======================

# ---------- 1) Coal (hourly, node/month specific)
mc_coal = build_marginal_cost(
    year=YEAR,
    others_setting_csv=OTHERS_SETTING_CSV,
    generators_units=gen,
)

mc = pd.DataFrame(
    0.0,
    index=n.snapshots,
    columns=n.generators.index,
    dtype=float
)

# assign coal MC
for rid in mc_coal.columns:
    gname = f"gen__{rid}"
    if gname in mc.columns:
        mc[gname] = mc_coal[rid].values


# ---------- 2) Read Gas & Nuclear prices from others_setting.csv
others = pd.read_csv(OTHERS_SETTING_CSV).set_index("node")

gas_price_by_node = others["Gas_Price (RMB/MWh-e)"]
nuclear_price_by_node = others["Nuclear_Price (RMB/MWh-e)"]


# ---------- 3) Gas (static, node-specific)
for _, r in gen[gen["carrier"] == "gas"].iterrows():
    rid = int(r["index"])
    gname = f"gen__{rid}"
    node = r["node"]

    if gname in mc.columns:
        mc[gname] = float(gas_price_by_node.loc[node])


# ---------- 4) Nuclear (static, node-specific)
for _, r in gen[gen["carrier"] == "nuclear"].iterrows():
    rid = int(r["index"])
    gname = f"gen__{rid}"
    node = r["node"]

    if gname in mc.columns:
        mc[gname] = float(nuclear_price_by_node.loc[node])


# ---------- 5) Assign to network
n.generators_t.marginal_cost = mc


# =======================
# SOLVE (LINOPY)
# =======================
n.optimize(
    solver_name="gurobi",
    solver_options=solver_options,
)


# =======================
# EXPORT
# =======================
n.buses_t.marginal_price.to_csv(OUT_PRICE)

flows = []
if len(n.links):
    f = n.links_t.p0.copy()
    f.columns = [f"LINK::{c}" for c in f.columns]
    flows.append(f)

if flows:
    pd.concat(flows, axis=1).to_csv(OUT_FLOW)
