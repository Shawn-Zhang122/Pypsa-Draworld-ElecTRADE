# ------------------------------------------------------------
# build_fuel_cost.py
# Build hourly coal prices (EX-VAT) and compute marginal cost
# ON THE FLY for PyPSA, to avoid too large csv files.
# ------------------------------------------------------------

import pandas as pd
import numpy as np

# -----------------------
# Physical & tax constants
# -----------------------
VAT = 1.13
KCAL_PER_KWH = 860.0
COAL_KCAL_PER_KG = 5500.0
MMBTU_TO_KWH_TH = 293.071

KWH_TH_PER_TON = (COAL_KCAL_PER_KG * 1000.0) / KCAL_PER_KWH
MWH_TH_PER_TON = KWH_TH_PER_TON / 1000.0  # 6.39535


# -----------------------
# Time index
# -----------------------
def hourly_index(year: int) -> pd.DatetimeIndex:
    return pd.date_range(
        f"{year}-01-01 00:00:00",
        f"{year}-12-31 23:00:00",
        freq="h"
    )


# -----------------------
# Coal prices
# -----------------------
def load_node_monthly_prices(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = ["node"] + [f"Coal_Price_{m}" for m in range(1, 13)]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"others_setting.csv missing columns: {missing}")
    return df[cols].copy()


def build_node_hourly_coal_price_exvat(
    node_monthly: pd.DataFrame,
    idx: pd.DatetimeIndex
) -> pd.DataFrame:
    """
    Returns: DataFrame [hour x node], RMB/ton, EX-VAT
    """
    out = pd.DataFrame(index=idx, columns=node_monthly["node"], dtype=float)
    monthly = node_monthly.set_index("node")[[f"Coal_Price_{m}" for m in range(1, 13)]].T

    for m in range(1, 13):
        out.loc[idx.month == m, :] = monthly.loc[f"Coal_Price_{m}"].values

    return out / VAT


# -----------------------
# Marginal cost (on the fly)
# -----------------------
def build_marginal_cost(
    year: int,
    others_setting_csv: str,
    generators_units: pd.DataFrame
) -> pd.DataFrame:
    """
    Returns: marginal cost [hour x generator], RMB/MWh_e, EX-VAT
    """

    idx = hourly_index(year)

    # hourly coal price by node
    node_monthly = load_node_monthly_prices(others_setting_csv)
    coal_price = build_node_hourly_coal_price_exvat(node_monthly, idx)

    coal_gens = generators_units[
        generators_units["Fuel"].str.lower().str.contains("coal")
    ].copy()

    HR = coal_gens["Heat_Rate_MMBTU_per_MWh"].values
    VarOM = coal_gens["Var_OM_Cost_per_MWh"].values
    nodes = coal_gens["node"].values
    gen_ids = coal_gens.index.astype(str)

    fuel_cost = (
        coal_price[nodes].values
        * (HR * MMBTU_TO_KWH_TH)
        / (MWH_TH_PER_TON * 1000.0)
    )

    return pd.DataFrame(
        fuel_cost + VarOM,
        index=idx,
        columns=gen_ids
    )

# -----------------------
# In prepare_network.py, please call:
    # mc = build_marginal_cost(
    #     year=2025,
    #     others_setting_csv="data/raw/others_setting.csv",
    #     generators_units=generators_df
    # )

    # network.generators_t.marginal_cost = mc
