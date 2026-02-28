# ============================================================
# 48-HOUR ROLLING MARKET SIMULATION (Clean Version)
# ============================================================

import pandas as pd
import numpy as np
import pypsa

pd.options.mode.string_storage = "python"

import os
from pathlib import Path
os.makedirs("docs/out/prices", exist_ok=True)
os.makedirs("docs/out/flows", exist_ok=True)

from build_fuel_cost import build_marginal_cost

# ============================================================
# CONFIG
# ============================================================

YEAR = 2025
HORIZON_HOURS = 48
STEP_HOURS = 24

# 1. Dynamically get the root directory of your project
# Path(__file__) is 'model/rolling_48hours_network.py'
# .parent is 'model/'
# .parent.parent is 'Pypsa-Draworld-ElecTRADE/'
BASE_DIR = Path(__file__).resolve().parent.parent

# 2. Update all paths using the / operator for cross-platform compatibility
LOAD_CSV   = BASE_DIR / "data/inputs/Load/Load_NDRC_BAs_China_Draworld_normalised_2025compiled.csv"
EDGES_CSV  = BASE_DIR / "data/inputs/Network/edges_33nodes_500kVplus_updates_Jan2026.csv"
GEN_CSV    = BASE_DIR / "data/inputs/Generator/generators_units.csv"

ONWIND_CSV  = BASE_DIR / "data/inputs/REprofile/ninja_wind_29.0000_120.0000.csv"
OFFWIND_CSV = BASE_DIR / "data/inputs/REprofile/ninja_wind_29.0000_120.0000.csv"
SOLAR_CSV   = BASE_DIR / "data/inputs/REprofile/ninja_pv_29.0000_120.0000.csv"

# Ensure this matches exactly where you saved the file earlier
HYDRO_CSV   = BASE_DIR / "data/inputs/REprofile/artificial_hydro_China_3400hrs.csv"

OTHERS_SETTING_CSV = BASE_DIR / "data/inputs/Others/others_setting.csv"

# 3. Add a "Safety Check" print for your cron logs
for name, path in [("Load", LOAD_CSV), ("Hydro", HYDRO_CSV), ("Gen", GEN_CSV)]:
    if not path.exists():
        print(f"WARNING: File not found at: {path}")
    else:
        print(f"SUCCESS: {name} path resolved to: {path}")


# ============================================================
# SOLVER OPTIONS
# ============================================================

solver_options = {
    "Threads": 4,
    "TimeLimit": 28800,
    "Method": 2,
    "Crossover": 0,
    "Presolve": 2,
    "DualReductions": 1,
    "NumericFocus": 3,
    "BarConvTol": 1e-6,
    "InfUnbdInfo": 1,
}


# ============================================================
# STORAGE PARAMETERS
# ============================================================

STORAGE_TECH = {
    "battery": {"hours": 2.3, "eta_chg": 0.95, "eta_dis": 0.95},
    "pumped":  {"hours": 8.0, "eta_chg": 0.90, "eta_dis": 0.80},
}


# ============================================================
# CARRIERS
# ============================================================

ELECTRICITY_CARRIER = "electricity"

GEN_CARRIERS = {
    "coal", "gas", "nuclear", "hydro",
    "onwind", "offwind", "solar",
    "other",
}

NETWORK_CARRIERS = {"ac", "dc"}

STORAGE_CARRIERS = set(STORAGE_TECH.keys())

STORAGE_LINK_CARRIERS = (
    {f"{c}_charge" for c in STORAGE_CARRIERS}
    | {f"{c}_discharge" for c in STORAGE_CARRIERS}
)

ALL_CARRIERS = (
    {ELECTRICITY_CARRIER}
    | GEN_CARRIERS
    | NETWORK_CARRIERS
    | STORAGE_CARRIERS
    | STORAGE_LINK_CARRIERS
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def norm(x):
    return str(x).strip()


def read_ts(path, idx, nodes):
    df = pd.read_csv(path)

    if len(df) != len(idx):
        raise ValueError(f"{path}: row count != snapshots")

    df = df.drop(columns=["datetime"])
    df.columns = [str(c).strip() for c in df.columns]
    df.index = idx

    missing = set(nodes) - set(df.columns)
    if missing:
        raise KeyError(f"{path}: missing nodes {missing}")

    if df.isna().any().any():
        raise ValueError(f"{path}: NaNs in data")

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

    carrier = str(carrier).lower()

    if carrier in {"coal", "gas"}:

        if "Eff_Up" in row and pd.notna(row["Eff_Up"]):
            eff = float(row["Eff_Up"])
            if eff > 0:
                return eff

        if "Heat_Rate_MMBTU_per_MWh" in row and pd.notna(row["Heat_Rate_MMBTU_per_MWh"]):
            hr = float(row["Heat_Rate_MMBTU_per_MWh"])
            if hr > 0:
                return 3.412 / hr

        return 0.35 if carrier == "coal" else 0.50

    return 1.0

# ============================================================
# BUILD NETWORK (UNCHANGED LOGIC)
# ============================================================

def build_network(year: int) -> pypsa.Network:

    idx = pd.date_range(
        f"{year}-01-01 00:00:00",
        f"{year}-12-31 23:00:00",
        freq="h"
    )

    n = pypsa.Network()
    n.set_snapshots(idx)

    # =======================
    # READ INPUT FILES
    # =======================

    gen = pd.read_csv(GEN_CSV)
    gen["node"] = gen["node"].map(norm)
    nodes = sorted(gen["node"].unique())

    load = read_ts(LOAD_CSV, idx, nodes)
    onwind_cf  = read_ts(ONWIND_CSV,  idx, nodes)
    offwind_cf = read_ts(OFFWIND_CSV, idx, nodes)
    solar_cf   = read_ts(SOLAR_CSV,   idx, nodes)
    hydro_cf  = read_ts(HYDRO_CSV,   idx, nodes)    

    edges = pd.read_csv(EDGES_CSV)
    edges["from_node"] = edges["from_node"].map(norm)
    edges["to_node"]   = edges["to_node"].map(norm)
    edges["ac_dc"]     = edges["ac_dc"].str.upper().str.strip()

    others = pd.read_csv(OTHERS_SETTING_CSV).set_index("node")
    max_load_mw = others["Max_Load_GW"] * 1e3  # GW → MW

    # =======================
    # REGISTER CARRIERS
    # =======================

    for c in sorted(ALL_CARRIERS):
        if c not in n.carriers.index:
            n.add("Carrier", c)

    if "load_shedding" not in n.carriers.index:
        n.add("Carrier", "load_shedding")

    # =======================
    # BUSES
    # =======================

    for b in nodes:
        n.add("Bus", b, carrier=ELECTRICITY_CARRIER)

    # =======================
    # LOADS
    # =======================

    load_mw = load.mul(max_load_mw, axis=1)

    for b in nodes:
        n.add("Load", f"load_{b}", bus=b)

    n.loads_t.p_set = load_mw.rename(columns={b: f"load_{b}" for b in nodes})
    n.loads_t.p_set = n.loads_t.p_set.fillna(0.0)

    # =======================
    # AC / DC LINKS
    # =======================

    for i, r in edges.iterrows():

        cap = float(r["capacity_mw"])
        if cap <= 0:
            continue

        acdc = r["ac_dc"]
        name = f"link_{i}_{r['from_node']}_{r['to_node']}"

        if acdc == "DC":
            n.add(
                "Link",
                name,
                bus0=r["from_node"],
                bus1=r["to_node"],
                p_nom=cap,
                p_min_pu=0.0,
                efficiency=0.99,
                carrier="dc",
                marginal_cost=0.0,
            )
        else:
            name_fwd = f"{name}__fwd"
            name_rev = f"{name}__rev"

            n.add(
                "Link",
                name_fwd,
                bus0=r["from_node"],
                bus1=r["to_node"],
                p_nom=cap,
                p_min_pu=0.0,
                p_max_pu=1.0,
                efficiency=0.99,
                carrier="ac",
                marginal_cost=0.0,
            )

            n.add(
                "Link",
                name_rev,
                bus0=r["to_node"],
                bus1=r["from_node"],
                p_nom=cap,
                p_min_pu=0.0,
                p_max_pu=1.0,
                efficiency=0.99,
                carrier="ac",
                marginal_cost=0.0,
            )

    # =======================
    # GENERATORS
    # =======================

    gen = gen.reset_index(drop=False)
    gen["carrier"] = gen.apply(infer_carrier, axis=1)

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

        n.add(
            "Generator",
            name,
            bus=r["node"],
            carrier=carrier,
            p_nom=p_nom,
            efficiency=eff,
            p_min_pu=0.0,
            p_max_pu=1.0,
        )

    # =======================
    # LOAD SHEDDING (VoLL)
    # =======================

    VOLL = 1e4

    for bus in n.buses.index:
        n.add(
            "Generator",
            name=f"load_shed_{bus}",
            bus=bus,
            carrier="load_shedding",
            p_nom=1e7,
            p_min_pu=0.0,
            p_max_pu=1.0,
            marginal_cost=VOLL,
            capital_cost=0.0,
            p_nom_extendable=False,
        )

    # =======================
    # STORAGE (Store + Links)
    # =======================

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

        s_bus   = f"{node}__{carrier}__bus__{rid}"
        s_store = f"{node}__{carrier}__store__{rid}"
        s_chg   = f"{node}__{carrier}__chg__{rid}"
        s_dis   = f"{node}__{carrier}__dis__{rid}"

        n.add("Bus", s_bus, carrier=carrier)

        n.add(
            "Store",
            s_store,
            bus=s_bus,
            e_nom=e_nom,
            e_initial=0.5 * e_nom,
            e_cyclic=False,  # REQUIRED for rolling
            carrier=carrier,
        )

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

    # =======================
    # RE AVAILABILITY
    # =======================

    n.generators_t.p_max_pu = pd.DataFrame(
        1.0,
        index=n.snapshots,
        columns=n.generators.index
    )

    for carrier, cf in {
        "onwind": onwind_cf,
        "offwind": offwind_cf,
        "solar": solar_cf,
        "hydro": hydro_cf,
    }.items():

        gens = n.generators.index[n.generators.carrier == carrier]
        if gens.empty:
            continue

        buses = n.generators.loc[gens, "bus"]
        sub = cf[buses.values]
        sub.columns = gens

        if sub.isna().any().any():
            raise ValueError(f"NaNs in p_max_pu for {carrier}")

        n.generators_t.p_max_pu.loc[:, gens] = sub.clip(0.0, 1.0)

    # =======================
    # MARGINAL COST
    # =======================

    mc = pd.DataFrame(
        np.repeat(
            n.generators.marginal_cost.values[None, :],
            len(n.snapshots),
            axis=0
        ),
        index=n.snapshots,
        columns=n.generators.index,
        dtype=float
    )

    gen_coal = gen[gen["carrier"] == "coal"]

    mc_coal = build_marginal_cost(
        year=year,
        others_setting_csv=OTHERS_SETTING_CSV,
        generators_units=gen_coal,
    )

    coal_map = {int(rid): f"gen__{int(rid)}" for rid in gen_coal["index"]}

    for rid, gname in coal_map.items():
        mc[gname] = mc_coal[str(rid)].values

    gas_price_by_node = others["Gas_Price (RMB/MWh-e)"]
    for _, r in gen[gen["carrier"] == "gas"].iterrows():
        mc[f"gen__{int(r['index'])}"] = float(gas_price_by_node.loc[r["node"]])

    nuclear_price_by_node = others["Nuclear_Price (RMB/MWh-e)"]
    for _, r in gen[gen["carrier"] == "nuclear"].iterrows():
        mc[f"gen__{int(r['index'])}"] = float(nuclear_price_by_node.loc[r["node"]])

    n.generators_t.marginal_cost = mc

    # =======================
    # FIX CAPACITIES (NO EXTENDABLES)
    # =======================

    if len(n.links):
        n.links["p_nom_extendable"] = False
        n.links["p_nom_min"] = n.links["p_nom"]
        n.links["p_nom_max"] = n.links["p_nom"]
        n.links["capital_cost"] = 0.0

    n.generators["p_nom_extendable"] = False
    n.generators["capital_cost"] = 0.0
    n.generators["p_nom_0"] = n.generators.p_nom.copy()

    if len(n.stores):
        n.stores["e_nom_extendable"] = False
        n.stores["e_nom_min"] = n.stores["e_nom"]
        n.stores["e_nom_max"] = n.stores["e_nom"]
        n.stores["capital_cost"] = 0.0


    return n


# ============================================================
# ROLLING 48H SOLVER
# ============================================================

def solve_rolling_48h(year: int):

    print("Building full-year network...")
    n_full = build_network(year)

    snapshots_all = n_full.snapshots

    # Initial SOC state
    soc_state = n_full.stores.e_initial.copy()

    price_results = []
    flow_results = []

    start_times = pd.date_range(
        snapshots_all[0],
        snapshots_all[-1],
        freq=f"{STEP_HOURS}h"
    )

    for start in start_times:

        end = start + pd.Timedelta(hours=HORIZON_HOURS - 1)

        horizon = snapshots_all[
            (snapshots_all >= start) &
            (snapshots_all <= end)
        ]

        if len(horizon) == 0:
            break

        publish_hours = min(STEP_HOURS, len(horizon))

        print(f"Solving horizon: {start} → {horizon[-1]}")

        n = n_full.copy()
        n.set_snapshots(horizon)

        n.stores["e_initial"] = soc_state

        n.optimize(
            solver_name="gurobi",
            solver_options=solver_options,
        )

        price_pub = n.buses_t.marginal_price.iloc[:publish_hours]
        # price_results.append(price_pub)
    
        date_str = price_pub.index[0].strftime("%Y-%m-%d")
        price_pub.to_csv(
            f"docs/out/prices/{date_str}.csv",
            float_format="%.2f"
        )
        if len(n.links):
            flow_pub = n.links_t.p0.iloc[:publish_hours]
            flow_pub.to_csv(
                f"docs/out/flows/{date_str}.csv",
                float_format="%.2f"
            )

        #   flow_results.append(flow_pub)

        soc_state = n.stores_t.e.iloc[publish_hours - 1].copy()


    #prices = pd.concat(price_results)
    #flows = pd.concat(flow_results) if flow_results else None
    #prices.to_csv("results/prices_2025_rolling_48h.csv")

    #if flows is not None:
    #    flows.to_csv("results/flows_2025_rolling_48h.csv")

    print("Rolling simulation completed.")

# ============================================================
# MAIN ENTRY
# ============================================================

if __name__ == "__main__":

    MODE = "rolling"  # or "annual"

    if MODE == "annual":
        n = build_network(YEAR)
        n.optimize(
            solver_name="gurobi",
            solver_options=solver_options,
        )
        n.export_to_netcdf("results/China_33nodes_dispatch_2025.nc")

    elif MODE == "rolling":
        solve_rolling_48h(YEAR)


