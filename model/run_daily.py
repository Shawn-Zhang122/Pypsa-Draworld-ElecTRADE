# ============================================================
# China-Daily-ElecTRADE
# Daily Rolling Market Simulation (48h horizon, 24h publish)
# ============================================================

#The run environment configuration

# Install Miniconda3,https://docs.conda.io/en/latest/miniconda.html
# cd ~/Downloads
# bash Miniconda3-py39_*.sh

#conda create -n pypsa-draworld-electrade python=3.9 -y
#conda activate pypsa-draworld-electrade
# conda install -c conda-forge \
#   numpy=1.24 \
#   scipy=1.10 \
#   pandas=1.5 \
#   xarray \
#   netcdf4 \
#   h5py \
#   pytables \
#   matplotlib \
#   geopandas \
#   shapely \
#   pyproj \
#   tqdm \
#   pyyaml \
#   ipython \
#   jupyterlab -y

# pip install --no-cache-dir \
#   pypsa==0.25.1 \
#   linopy==0.3.13

# pip install gurobipy==10.0.3
# export GRB_LICENSE_FILE=$PWD/solver/gurobi.lic

import os
from datetime import datetime, timedelta
import pytz
import pandas as pd

from rolling_48hours_network import build_network

# ============================================================
# CONFIG
# ============================================================

YEAR = 2025
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
# Determine delivery day (China time)
# ============================================================

# Day-ahead market → publish next day
tz = pytz.timezone("Asia/Shanghai")
today_real = datetime.now(tz).date()

# Map real calendar to simulation year
delivery_day = pd.Timestamp(
    YEAR,
    today_real.month,
    today_real.day
)

# Day-ahead
delivery_day += pd.Timedelta(days=1)

# Handle Dec 31 wrap-around
if delivery_day.year > YEAR:
    delivery_day = pd.Timestamp(YEAR, 1, 1)

print(f"Running day-ahead simulation for delivery date: {delivery_day}")

# ============================================================
# Build network
# ============================================================

n_full = build_network(YEAR)

# ============================================================
# Define rolling horizon
# ============================================================

start = pd.Timestamp(delivery_day)
end = start + pd.Timedelta(hours=HORIZON_HOURS - 1)

snapshots = n_full.snapshots[
    (n_full.snapshots >= start) &
    (n_full.snapshots <= end)
]

if len(snapshots) == 0:
    raise RuntimeError("No snapshots found for selected delivery date.")

n = n_full.copy()
n.set_snapshots(snapshots)

# ============================================================
# Carry storage SOC from previous day if exists
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
# Publish first 24h
# ============================================================

price_pub = n.buses_t.marginal_price.iloc[:PUBLISH_HOURS]
flow_pub = n.links_t.p0.iloc[:PUBLISH_HOURS] if len(n.links) else None

date_str = delivery_day.strftime("%Y-%m-%d")

price_path = f"docs/out/prices/{date_str}.csv"
flow_path  = f"docs/out/flows/{date_str}.csv"

price_pub.to_csv(price_path, float_format="%.2f")

if flow_pub is not None:
    flow_pub.to_csv(flow_path, float_format="%.2f")

print(f"Published prices → {price_path}")

# ============================================================
# Save new SOC (end of published day)
# ============================================================

new_soc = n.stores_t.e.iloc[PUBLISH_HOURS - 1].copy()
new_soc.to_csv(soc_file)

print("Saved updated SOC state.")

print("Daily rolling simulation completed.")

# ============================================================
# Update index.json
# ============================================================
import json
index_path = "docs/out/index.json"

if os.path.exists(index_path):
    with open(index_path, "r") as f:
        index = json.load(f)
else:
    index = {"price_files": [], "soc_file": "last_soc.csv"}

filename = f"{date_str}.csv"

if filename not in index["price_files"]:
    index["price_files"].append(filename)

# sort newest first
index["price_files"] = sorted(index["price_files"], reverse=True)

# always keep correct soc file name
index["soc_file"] = "last_soc.csv"

with open(index_path, "w") as f:
    json.dump(index, f, indent=2)

print("Updated index.json")