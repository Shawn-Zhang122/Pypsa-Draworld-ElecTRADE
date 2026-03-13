# ============================================================
# China-Daily-ElecTRADE: run_daily.py
# Rolling 48h Solve (Publish 24h)
# Optimized for Merit Order Visualization with Dynamic Costs
# ============================================================

import os
import json
from datetime import datetime
import sys
import pytz
import pandas as pd
import numpy as np
from rolling_48hours_network import build_network

# ============================================================
# CONFIG
# ============================================================
DATA_YEAR = 2025
HORIZON_HOURS = 48
PUBLISH_HOURS = 24
SOLVER_NAME = "gurobi"

solver_options = {
    "Threads": 4,
    "Method": 2,
    "Crossover": 0,
    "Presolve": 2,
    "NumericFocus": 3,
}

# Ensure output directories exist
os.makedirs("docs/out/prices", exist_ok=True)
os.makedirs("docs/out/flows", exist_ok=True)
os.makedirs("docs/out/merit", exist_ok=True)

# ============================================================
# Determine Real Publication Day (China Time)
# ============================================================
tz = pytz.timezone("Asia/Shanghai")
today_real = datetime.now(tz).date()
real_delivery_day = today_real + pd.Timedelta(days=1)
date_str = real_delivery_day.strftime("%Y-%m-%d")

print(f"--- Market Run for Delivery Date: {date_str} ---")

# Map Real Day -> DATA_YEAR Day (seasonal replay)
model_delivery_day = pd.Timestamp(DATA_YEAR, real_delivery_day.month, real_delivery_day.day)
if model_delivery_day.month == 2 and model_delivery_day.day == 29:
    model_delivery_day = pd.Timestamp(DATA_YEAR, 2, 28)

# ============================================================
# Build & Setup Network
# ============================================================
n_full = build_network(DATA_YEAR)

start = model_delivery_day
end = start + pd.Timedelta(hours=HORIZON_HOURS - 1)
snapshots = n_full.snapshots[(n_full.snapshots >= start) & (n_full.snapshots <= end)]

if len(snapshots) == 0:
    raise RuntimeError("No snapshots found in DATA_YEAR for selected date.")

n = n_full.copy()
n.set_snapshots(snapshots)

# Carry Storage SOC from previous day
soc_file = "docs/out/last_soc.csv"
if os.path.exists(soc_file):
    soc_state = pd.read_csv(soc_file, index_col=0).squeeze()
    n.stores["e_initial"] = soc_state
    print("Loaded previous SOC state.")
else:
    print("No SOC file found. Using default initial state.")

# ============================================================
# Optimize
# ============================================================
print(f"Optimizing {HORIZON_HOURS}h rolling horizon with {SOLVER_NAME}...")
n.optimize(solver_name=SOLVER_NAME, solver_options=solver_options)

# ============================================================
# Prepare Results for Publication (First 24h)
# ============================================================
real_index = [
    pd.Timestamp(real_delivery_day.year, ts.month, ts.day, ts.hour)
    for ts in n.snapshots[:PUBLISH_HOURS]
]

# 1. Price Export (LMP)
price_pub = n.buses_t.marginal_price.iloc[:PUBLISH_HOURS].copy()
price_pub.index = real_index
price_pub.index.name = "snapshot"
price_path = f"docs/out/prices/{date_str}.csv"
price_pub.to_csv(price_path, float_format="%.2f")

# 2. Flow Export
if len(n.links):
    flow_pub = n.links_t.p0.iloc[:PUBLISH_HOURS].copy()
    flow_pub.index = real_index
    flow_path = f"docs/out/flows/{date_str}.csv"
    flow_pub.to_csv(flow_path, float_format="%.2f")

# 3. Update SOC for next day's run
new_soc = n.stores_t.e.iloc[PUBLISH_HOURS - 1].copy()
new_soc.to_csv(soc_file)

# ============================================================
# Export Merit Order (Supply/Demand) Data
# ============================================================
merit_data = []
print("Extracting nodal merit order data ...")

# Define your 33 main provincial nodes to avoid sub-buses/battery-buses
# If you don't have this list, we filter buses that actually have loads
main_buses = n.loads.bus.unique()

for i, t in enumerate(n.snapshots[:PUBLISH_HOURS]):
    ts_str = real_index[i].strftime("%Y-%m-%d %H:%M")
    
    for bus in main_buses:
        # --- 1. Net Demand calculation (Load + Exports) ---
        # --- 1. Net Demand calculation (Load + Exports + Charging) ---
        
        # A. Physical Load
        loads_at_bus = n.loads[n.loads.bus == bus].index
        load_val = float(n.loads_t.p_set.loc[t, loads_at_bus].sum()) if not loads_at_bus.empty else 0.0
        
        # B. Exports to other provinces (p0 > 0 at bus0)
        exp_links = n.links[(n.links.bus0 == bus) & n.links.carrier.isin(["ac", "dc"])]
        exports = float(n.links_t.p0.loc[t, exp_links.index].sum()) if not exp_links.empty else 0.0
        
        # C. Storage Charging (p0 > 0 at bus0)
        chg_links = n.links[(n.links.bus0 == bus) & (n.links.carrier.str.contains("charge"))]
        charging = float(n.links_t.p0.loc[t, chg_links.index].sum()) if not chg_links.empty else 0.0
        
        # Total Demand that Supply must meet
        net_demand = load_val + exports + charging

        # --- 2. Local Generators ---
        gens = n.generators[n.generators.bus == bus]
        for gen_name, row in gens.iterrows():
            dispatch = float(n.generators_t.p.loc[t, gen_name])
            
            # Dynamic cost check
            if gen_name in n.generators_t.marginal_cost.columns:
                cost = float(n.generators_t.marginal_cost.loc[t, gen_name])
            else:
                cost = float(row.marginal_cost)

            if dispatch > 0.01:
                merit_data.append({
                    "bus": bus, "snapshot": ts_str, "type": "Generator",
                    "label": f"{row.carrier}_{gen_name}",
                    "capacity": dispatch, "cost": cost,
                    "net_demand": net_demand
                })

        # --- 3. Imports via Links ---
        imp_links = n.links[(n.links.bus1 == bus) & n.links.carrier.isin(["ac", "dc"])]
        for link_name, row in imp_links.iterrows():
            source_price = float(n.buses_t.marginal_price.loc[t, row.bus0])
            dest_price = float(n.buses_t.marginal_price.loc[t, row.bus1])
            eff = float(row.efficiency)
            
            base_bid = source_price / eff
            delivered_cap = float(n.links_t.p0.loc[t, link_name] * eff)

            if delivered_cap > 0.01:
                merit_data.append({
                    "bus": bus, "snapshot": ts_str, "type": "Import",
                    "label": f"Import_{row.bus0}",
                    "capacity": delivered_cap, "cost": base_bid,
                    "net_demand": net_demand
                })
                if dest_price > (base_bid + 0.1):
                    merit_data.append({
                        "bus": bus, "snapshot": ts_str, "type": "Congestion",
                        "label": f"Rent_{link_name}",
                        "capacity": 0, "cost": dest_price,
                        "net_demand": net_demand
                    })

        # --- 4. Storage Discharge (Link-based Supply) ---
        # Find discharge links where the destination (bus1) is the current province
        dis_links = n.links[(n.links.bus1 == bus) & (n.links.carrier.str.contains("discharge"))]
        
        for link_name, row in dis_links.iterrows():
            # p0 is the power leaving the storage bus
            p0 = n.links_t.p0.loc[t, link_name]
            eff = float(row.efficiency)
            
            # Power delivered to the province is p0 * efficiency
            # For a discharge link, this value is POSITIVE in PyPSA
            p_delivered = float(p0 * eff)

            if p_delivered > 0.01:
                # Clean up label: "lithium_discharge" -> "lithium"
                clean_label = row.carrier.replace("_discharge", "")
                
                merit_data.append({
                    "bus": bus,
                    "snapshot": ts_str,
                    "type": "Storage",
                    "label": f"Discharge_{clean_label}_{link_name.split('__')[-1]}",
                    "capacity": p_delivered,
                    # Storage is a price-setter, so cost = Nodal LMP
                    "cost": float(n.buses_t.marginal_price.loc[t, bus]),
                    "net_demand": net_demand
                })

            # --- 5. Storage Charge (Link-based Supply), to be added by Shuwei, 2026---

merit_df = pd.DataFrame(merit_data)
merit_path = f"docs/out/merit/{date_str}.csv"
merit_df.to_csv(merit_path, index=False, float_format="%.2f")

# ============================================================
# Update index.json
# ============================================================
update_time_str = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
index_path = "docs/out/index.json"

if os.path.exists(index_path):
    with open(index_path, "r") as f:
        index = json.load(f)
else:
    index = {"price_files": [], "merit_files": [], "soc_file": "last_soc.csv"}

# Ensure keys exist
if "merit_files" not in index: index["merit_files"] = []
if "price_files" not in index: index["price_files"] = []

# Update with newest files
if f"{date_str}.csv" not in index["price_files"]:
    index["price_files"].append(f"{date_str}.csv")
if f"{date_str}.csv" not in index["merit_files"]:
    index["merit_files"].append(f"{date_str}.csv")

index["last_update"] = update_time_str
index["delivery_date"] = date_str
index["price_files"] = sorted(index["price_files"], reverse=True)
index["merit_files"] = sorted(index["merit_files"], reverse=True)

with open(index_path, "w") as f:
    json.dump(index, f, indent=2)

# ============================================================
# Full Network Export (.nc)
# ============================================================
finish_time = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
# Export Network
nc_network_path = f"docs/out/network_{date_str}.nc"
n.export_to_netcdf(nc_network_path)

# Print clean status for cron.log
print(f"\n" + "="*50)
print(f"SUCCESS: Optimization & Export Complete")
print(f"Delivery Date : {date_str}")
print(f"Solved At (Beijing Time)    : {finish_time}")
print(f"Pypsa-Draworld-ElecTRADENetwork Saved : {nc_network_path}")
print("="*50 + "\n")

sys.exit(0)



