# ============================================================
# China-Daily-ElecTRADE
# Rolling 48h Solve (Publish 24h)
# Data Year = 2025
# Publication Year = Real Calendar (e.g. 2026)
# ============================================================

import os
import json
from datetime import datetime
import pytz
import pandas as pd

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

# ============================================================
# Ensure output directories exist
# ============================================================

os.makedirs("docs/out/prices", exist_ok=True)
os.makedirs("docs/out/flows", exist_ok=True)

# ============================================================
# Determine Real Publication Day (China Time)
# ============================================================

tz = pytz.timezone("Asia/Shanghai")
today_real = datetime.now(tz).date()

# Day-ahead publication date (real calendar)
real_delivery_day = today_real + pd.Timedelta(days=1)

print(f"Publishing market results for: {real_delivery_day}")

# ============================================================
# Map Real Day → DATA_YEAR Day (seasonal replay)
# ============================================================

model_delivery_day = pd.Timestamp(
    DATA_YEAR,
    real_delivery_day.month,
    real_delivery_day.day
)

# Handle leap mismatch safely
if model_delivery_day.month == 2 and model_delivery_day.day == 29:
    model_delivery_day = pd.Timestamp(DATA_YEAR, 2, 28)

# ============================================================
# Build Network (DATA YEAR)
# ============================================================

n_full = build_network(DATA_YEAR)

# ============================================================
# Define 48h Rolling Horizon (DATA YEAR)
# ============================================================

start = model_delivery_day
end = start + pd.Timedelta(hours=HORIZON_HOURS - 1)

snapshots = n_full.snapshots[
    (n_full.snapshots >= start) &
    (n_full.snapshots <= end)
]

if len(snapshots) == 0:
    raise RuntimeError("No snapshots found in DATA_YEAR for selected date.")

n = n_full.copy()
n.set_snapshots(snapshots)

# ============================================================
# Carry Storage SOC
# ============================================================

soc_file = "docs/out/last_soc.csv"

if os.path.exists(soc_file):
    soc_state = pd.read_csv(soc_file, index_col=0).squeeze()
    print("Loaded previous SOC state.")
else:
    soc_state = n.stores.e_initial.copy()
    print("Using default initial SOC.")

n.stores["e_initial"] = soc_state

# ============================================================
# Optimize
# ============================================================

print("Optimizing 48h rolling horizon...")
n.optimize(
    solver_name=SOLVER_NAME,
    solver_options=solver_options,
)

# ============================================================
# Publish First 24h (Convert Timestamp to REAL YEAR)
# ============================================================

price_pub = n.buses_t.marginal_price.iloc[:PUBLISH_HOURS]
flow_pub = n.links_t.p0.iloc[:PUBLISH_HOURS] if len(n.links) else None

# Convert DATA_YEAR timestamps → REAL YEAR timestamps
real_index = [
    pd.Timestamp(
        real_delivery_day.year,
        ts.month,
        ts.day,
        ts.hour
    )
    for ts in price_pub.index
]

price_pub.index = real_index

if flow_pub is not None:
    flow_pub.index = real_index

# ============================================================
# Export CSV
# ============================================================

date_str = real_delivery_day.strftime("%Y-%m-%d")

price_path = f"docs/out/prices/{date_str}.csv"
flow_path  = f"docs/out/flows/{date_str}.csv"

price_pub.index = real_index
price_pub.index.name = "snapshot"
price_pub.to_csv(price_path, float_format="%.2f")

if flow_pub is not None:
    flow_pub.to_csv(flow_path, float_format="%.2f")

print(f"Published prices → {price_path}")

# ============================================================
# Save New SOC (still DATA_YEAR internal)
# ============================================================

new_soc = n.stores_t.e.iloc[PUBLISH_HOURS - 1].copy()
new_soc.to_csv(soc_file)

print("Saved updated SOC state.")

# ============================================================
# Update index.json
# ============================================================

index_path = "docs/out/index.json"

if os.path.exists(index_path):
    with open(index_path, "r") as f:
        index = json.load(f)
else:
    index = {"price_files": [], "soc_file": "last_soc.csv"}

filename = f"{date_str}.csv"

if filename not in index["price_files"]:
    index["price_files"].append(filename)

index["price_files"] = sorted(index["price_files"], reverse=True)
index["soc_file"] = "last_soc.csv"

with open(index_path, "w") as f:
    json.dump(index, f, indent=2)

print("Updated index.json")
print("Daily rolling simulation completed.")

# ============================================================
# Export Full PyPSA Network (.nc)
# ============================================================

# Use REAL publication date for naming
date_str = real_delivery_day.strftime("%Y-%m-%d")

nc_network_path = f"docs/out/network_{date_str}.nc"

# This exports:
# - all components
# - solved dispatch
# - marginal prices
# - duals
# - stores state
# - links, generators, constraints, etc.
n.export_to_netcdf(nc_network_path)

print(f"Exported solved network → {nc_network_path}")