"""
PRICE SETTER CLASSIFICATION LOGIC
----------------------------------

This script plots the nodal marginal price (LMP) from the solved LP
and classifies the hourly price-setting mechanism using strict KKT conditions.

Key principles:

1) Market price
---------------
The price used in the plot is:

    lambda_{b,t} = n.buses_t.marginal_price

This is the dual variable of the nodal power balance constraint.
It is NOT reconstructed from generator marginal costs.

2) Local generator sets price if:
----------------------------------
- Dispatch is interior:
      0 < p_{i,t} < p_max_{i,t}
- Marginal cost matches price:
      |MC_{i,t} - lambda_{b,t}| < eps

3) Import sets price if:
-------------------------
- Flow on incoming link is interior:
      0 < f_{l,t} < f_max
- KKT condition holds:
      lambda_{b,t} = eta_l * lambda_{upstream,t} + link_cost

4) Storage sets price if:
--------------------------
- Discharge is interior:
      0 < p_dis,t < p_dis_max
- Intertemporal KKT holds:
      lambda_{b,t} = mu_e,t / eta_dis

where mu_e,t is the dual variable of the storage energy balance constraint.

5) Scarcity pricing (VOLL):
---------------------------
If load shedding is active, price = VOLL.

Notes:
------
- Classification is based strictly on LP optimality conditions.
- No geometric merit-order intersection is used.
- If no condition matches, price is attributed to other binding constraints.
"""

import pypsa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def robust_mc_at_t(n, t, gens):
    mc = pd.Series(index=gens, dtype=float)
    if hasattr(n.generators_t, "marginal_cost") and n.generators_t.marginal_cost is not None:
        mc_df = n.generators_t.marginal_cost
        common = gens.intersection(mc_df.columns)
        mc.loc[common] = mc_df.loc[t, common]
    missing = mc[mc.isna()].index
    if len(missing):
        mc.loc[missing] = n.generators.loc[missing, "marginal_cost"]
    return mc.fillna(0.0)


def robust_pmax_at_t(n, t, gens):
    pmax = pd.Series(1.0, index=gens)
    if hasattr(n.generators_t, "p_max_pu") and n.generators_t.p_max_pu is not None:
        pmax_df = n.generators_t.p_max_pu
        common = gens.intersection(pmax_df.columns)
        pmax.loc[common] = n.generators_t.p_max_pu.loc[t, common]
    return pmax


def classify_price_setter(n, t, bus, tol=1e-3, eps=1e-2):

    lam = float(n.buses_t.marginal_price.loc[t, bus])

    # -------------------
    # 1️⃣ Load shedding
    # -------------------
    shed = f"load_shed_{bus}"
    if shed in n.generators.index:
        if n.generators_t.p.loc[t, shed] > tol:
            return "VOLL"

    # -------------------
    # 2️⃣ Local generator
    # -------------------
    gens = n.generators.index[n.generators.bus == bus]
    gens = gens[n.generators.loc[gens, "carrier"] != "load_shedding"]

    if len(gens):
        p = n.generators_t.p.loc[t, gens]
        cap = n.generators.loc[gens, "p_nom"]
        mc = n.generators.loc[gens, "marginal_cost"]

        interior = (p > tol) & (p < cap - tol)
        close = (mc - lam).abs() < eps

        if (interior & close).any():
            return "Local"

    # -------------------
    # 3️⃣ Import marginal
    # -------------------
    net_links = n.links.index[n.links.carrier.isin(["ac", "dc"])]
    incoming = n.links.index[(n.links.bus1 == bus) & (n.links.index.isin(net_links))]

    for l in incoming:
        p0 = float(n.links_t.p0.loc[t,l])
        cap_l = float(n.links.loc[l,"p_nom"])

        if p0 > tol and p0 < cap_l - tol:

            lam0 = float(n.buses_t.marginal_price.loc[t,n.links.loc[l,"bus0"]])
            eta = float(n.links.loc[l,"efficiency"])
            cl = float(n.links.loc[l,"marginal_cost"])

            if abs(lam - (eta*lam0 + cl)) < eps:
                return "Import"

    # -------------------
    # 4️⃣ Strict storage test (KKT)
    # -------------------

    if not hasattr(n.stores_t, "mu_energy"):
        return "Storage/Intertemporal (no dual)"

    # find storage units connected to this bus
    dis_links = n.links.index[
        (n.links.bus1 == bus) &
        (n.links.carrier.astype(str).str.endswith("_discharge"))
    ]

    for link in dis_links:

        store_bus = n.links.loc[link, "bus0"]

        # find corresponding store
        stores = n.stores.index[n.stores.bus == store_bus]
        if len(stores) == 0:
            continue

        store = stores[0]

        p_dis = float(n.links_t.p0.loc[t, link])
        cap_dis = float(n.links.loc[link, "p_nom"])
        eta = float(n.links.loc[link, "efficiency"])

        if p_dis > tol and p_dis < cap_dis - tol:

            mu_e = float(n.stores_t.mu_energy.loc[t, store])

            if abs(lam - mu_e/eta) < eps:
                return "Storage"

    # -------------------
    # 5️⃣ Residual
    # -------------------
    return "Other/Constraint"

def plot_price_with_setters(
    nc_path,
    bus,
    start,
    end,
    save=True,
    out_dir="results"
):

    n = pypsa.Network(nc_path)

    T = n.snapshots[(n.snapshots >= pd.Timestamp(start)) &
                    (n.snapshots <= pd.Timestamp(end))]

    price = n.buses_t.marginal_price.loc[T, bus]

    setters = []
    for t in T:
        setters.append(classify_price_setter(n, t, bus))

    df = pd.DataFrame({
        "price": price,
        "setter": setters
    }, index=T)

    # color mapping
    color_map = {
        "Local": "tab:blue",
        "Import": "tab:green",
        "Import-Cong": "tab:orange",
        "VOLL": "tab:red",
        "Storage/Intertemporal": "tab:purple"
    }

    fig, ax = plt.subplots(figsize=(14,6))

    for setter, group in df.groupby("setter"):
        ax.scatter(group.index,
                   group.price,
                   color=color_map.get(setter,"black"),
                   label=setter,
                   s=10)

    ax.plot(df.index, df.price, color="black", alpha=0.3)

    ax.set_ylabel("RMB/MWh")
    ax.set_title(f"Nodal price and price setters – {bus}")
    ax.legend(frameon=False, ncol=3)
    plt.tight_layout()

    if save:
        os.makedirs(out_dir, exist_ok=True)
        fname = f"price_setter_{bus}_{start}_{end}.png"
        path = os.path.join(out_dir, fname)
        plt.savefig(path, dpi=300)
        plt.close()
        print("Saved:", path)
    else:
        plt.show()


if __name__ == "__main__":
    plot_price_with_setters(
        nc_path="results/dispatch_2025.nc",
        bus="Guangdong",
        start="2025-07-15 00:00:00",
        end="2025-07-21 23:00:00"
    )
