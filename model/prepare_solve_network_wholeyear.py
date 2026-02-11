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

pd.options.mode.string_storage = "python"

from build_fuel_cost import hourly_index, build_marginal_cost
import validation_before_solving as vbs

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

OTHERS_SETTING_CSV = "data/inputs/Others/others_setting.csv"

OUT_PRICE = "results/prices_2025_8760.csv"
OUT_FLOW  = "results/flows_2025_8760.csv"


# =======================
# SOLVER (Linopy)
# =======================

solver_options = {
    "Threads": 4,
    "TimeLimit": 28800,

    # CORE FIXES
    "Method": 2,          # barrier
    "Crossover": 0,       # keep barrier solution
    "Presolve": 2,        # aggressive presolve
    "DualReductions": 1,  # IMPORTANT: allow cleanup

    # NUMERICS
    "NumericFocus": 3,
    "BarConvTol": 1e-6,

    # SAFETY
    "InfUnbdInfo": 1,
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

    # force canonical hourly index
    if len(df) != len(idx):
        raise ValueError(f"{path}: row count != snapshots")

    df = df.drop(columns=["datetime"])
    df.columns = [str(c).strip() for c in df.columns]
    df.index = idx   # <-- authoritative time index

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

    # -----------------------
    # Fossil / thermal only
    # -----------------------
    if carrier in {"coal", "gas"}:

        # explicit efficiency provided
        if "Eff_Up" in row and pd.notna(row["Eff_Up"]):
            eff = float(row["Eff_Up"])
            if eff > 0:
                return eff

        # heat rate (MMBTU/MWh) → efficiency
        if "Heat_Rate_MMBTU_per_MWh" in row and pd.notna(row["Heat_Rate_MMBTU_per_MWh"]):
            hr = float(row["Heat_Rate_MMBTU_per_MWh"])
            if hr > 0:                     # CRITICAL guard
                return 3.412 / hr

        # fallback defaults
        return 0.35 if carrier == "coal" else 0.50

    # -----------------------
    # Non-thermal generators
    # -----------------------
    # renewables, nuclear, hydro, storage, etc.
    return 1.0


# =======================
# BUILD NETWORK
# =======================
idx = pd.date_range(f"{YEAR}-01-01 00:00:00", f"{YEAR}-12-31 23:00:00", freq="h")

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

others = pd.read_csv(OTHERS_SETTING_CSV)
#others["node"] = others["node"].map(norm)
others = others.set_index("node")

max_load_mw = others["Max_Load_GW"] * 1e3  # GW -> MW

n = pypsa.Network()
n.set_snapshots(idx)
#n.set_investment_periods([YEAR])

# -----------------------
# Register carriers ONCE (before adding components that reference carriers)
# -----------------------
for c in sorted(ALL_CARRIERS):
    if c not in n.carriers.index:
        n.add("Carrier", c)

if "load_shedding" not in n.carriers.index:
    n.add("Carrier", "load_shedding")

# buses
for b in nodes:
    n.add("Bus", b, carrier=ELECTRICITY_CARRIER)

# loads
load_mw = load.mul(max_load_mw, axis=1)
for b in nodes:
    n.add("Load", f"load_{b}", bus=b)
n.loads_t.p_set = load_mw.rename(columns={b: f"load_{b}" for b in nodes})
n.loads_t.p_set = n.loads_t.p_set.fillna(0.0)

print("=== Max load per node (MW) ===")
print(n.loads_t.p_set.max().sort_values(ascending=False))


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
            efficiency=0.99,
            carrier="dc",
            marginal_cost = 0.0 
        )

    else:
    # AC: two unidirectional Links (PyPSA-Eur style)
        name_fwd = f"{name}__fwd"
        name_rev = f"{name}__rev"

        # from_node -> to_node
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
            marginal_cost=0.0,   # SAFE: no signed-flow cost
        )

        # to_node -> from_node
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
    # only applied for UC models, but set here for validation purposes
    # if carrier == "coal" and "Min_Power" in r and pd.notna(r["Min_Power"]):
    #     p_min_pu = float(r["Min_Power"])

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
    
VOLL = 1e4  # RMB/MWh, choose >> any marginal cost

for bus in n.buses.index:
    n.add(
        "Generator",
        name=f"load_shed_{bus}",
        bus=bus,
        carrier="load_shedding",
        p_nom=1e7,              # effectively unlimited
        p_min_pu=0.0,
        p_max_pu=1.0,
        marginal_cost=VOLL,
        capital_cost=0.0,
        p_nom_extendable=False
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
    if gens.empty:
        continue

    buses = n.generators.loc[gens, "bus"]

    sub = cf[buses.values]
    sub.columns = gens

    # fail fast
    if sub.isna().any().any():
        raise ValueError(f"NaNs in p_max_pu for {carrier}")

    n.generators_t.p_max_pu.loc[:, gens] = sub.clip(0.0, 1.0)

# =======================
# MARGINAL COST (EXPLICIT)
# =======================

# 0) start from static marginal_cost (broadcast to all snapshots)
mc = pd.DataFrame(
    np.repeat(n.generators.marginal_cost.values[None, :], len(n.snapshots), axis=0),
    index=n.snapshots,
    columns=n.generators.index,
    dtype=float
)

# -------------------------------------------------
# 1) COAL: hourly, node/month specific
# -------------------------------------------------
gen_coal = gen[gen["carrier"] == "coal"]

mc_coal = build_marginal_cost(
    year=YEAR,
    others_setting_csv=OTHERS_SETTING_CSV,
    generators_units=gen_coal,
)

coal_map = {int(rid): f"gen__{int(rid)}" for rid in gen_coal["index"]}

mc_coal_cols = set(map(int, mc_coal.columns))
missing = set(coal_map) - mc_coal_cols
if missing:
    raise KeyError(f"Missing coal MC for rids: {sorted(missing)}")

for rid, gname in coal_map.items():
    mc[gname] = mc_coal[str(rid)].values

# -------------------------------------------------
# 2) GAS: static, node-specific
# -------------------------------------------------
gas_price_by_node = others["Gas_Price (RMB/MWh-e)"]

for _, r in gen[gen["carrier"] == "gas"].iterrows():
    gname = f"gen__{int(r['index'])}"
    mc[gname] = float(gas_price_by_node.loc[r["node"]])

# -------------------------------------------------
# 3) NUCLEAR: static, node-specific
# -------------------------------------------------
nuclear_price_by_node = others["Nuclear_Price (RMB/MWh-e)"]

for _, r in gen[gen["carrier"] == "nuclear"].iterrows():
    gname = f"gen__{int(r['index'])}"
    mc[gname] = float(nuclear_price_by_node.loc[r["node"]])

# -------------------------------------------------
# 4) ASSIGN TO NETWORK
# -------------------------------------------------
n.generators_t.marginal_cost = mc

# =======================
# SOLVE (LINOPY)
#vbs.validation_before_solving(n)

# --- Links (AC/DC)
n.links["p_nom_extendable"] = False
n.links["p_nom_min"] = n.links["p_nom"]
n.links["p_nom_max"] = n.links["p_nom"]

# --- Generators
# default: no generator is extendable
n.generators["p_nom_extendable"] = False
# only coal generators are extendable, to avoid the "infeasible" issue
mask = n.generators.carrier == "coal"
n.generators.loc[mask, "p_nom_extendable"] = False

# --- Stores
n.stores["e_nom_extendable"] = False
n.stores["e_nom_min"] = n.stores["e_nom"]
n.stores["e_nom_max"] = n.stores["e_nom"]

#Investment cost silienced.
n.links["capital_cost"] = 0.0
n.generators["capital_cost"] = 0.0
n.stores["capital_cost"] = 0.0
n.generators["p_nom_0"] = n.generators.p_nom.copy()
#to be used for faster testing and online display/update every week
#n.set_snapshots(n.snapshots[:168])  # 1 week

n.optimize(
    solver_name="gurobi",
    solver_options=solver_options,
)

# WRITE NETCDF IMMEDIATELY
#prices = n.buses_t.marginal_price.copy()
#n.generators["p_nom_star"] = n.generators.p_nom
#prices.to_csv(OUT_PRICE)
mu = n.model.dual["Store-energy_balance"]
mu_df = mu.to_pandas()

mu_df.index = n.snapshots
mu_df.columns = n.stores.index

n.stores_t["mu_energy"] = mu_df

n.export_to_netcdf("results/China_33nodes_dispatch_2025.nc")

# =======================
# EXPORT
# =======================
# after solve
#del n.links_t
#del n.stores_t
#n.links_t.p0.to_csv(OUT_FLOW)
