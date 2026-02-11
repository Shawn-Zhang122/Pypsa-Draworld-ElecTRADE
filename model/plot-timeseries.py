"""
# Plot an integrated view on the hourly price across provinces and weeks. 
"""

# balance_area_plot.py
# Purpose:
#   For one province bus (node), plot:
#     - generation (stacked, >0)
#     - storage discharge + imports (stacked, >0)
#     - storage charging + exports (stacked, <0)
#     - load as dotted black line
#
# Works with your model structure:
#   - generators on main province buses
#   - inter-province transfers via Links (carrier in {"ac","dc"})
#   - storage via Store bus + charge/discharge Links (carrier endswith "_charge"/"_discharge")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pypsa
import os

def _sum_links_p0_times_eff(n, link_names):
    """Return Series: sum_t (p0 * efficiency) across selected links."""
    if len(link_names) == 0:
        return pd.Series(0.0, index=n.snapshots)
    df = n.links_t.p0[link_names]
    eff = n.links.loc[link_names, "efficiency"].astype(float)
    return (df * eff.values).sum(axis=1)


def plot_balance_area(
    nc_path: str,
    bus: str,
    start: str,
    end: str,
    gen_carrier_order=None,
    rename_carriers=None,
    figsize=(14, 6),
):
    """
    nc_path: e.g. "results/dispatch_2025.nc"
    bus:     one of your 33 province nodes, e.g. "Guangdong"
    start/end: time slice inclusive, e.g. "2025-01-01" / "2025-01-07 23:00"
    """

    n = pypsa.Network(nc_path)

    # --- time slice (inclusive for convenience)
    t = n.snapshots[(n.snapshots >= pd.Timestamp(start)) & (n.snapshots <= pd.Timestamp(end))]
    if len(t) == 0:
        raise ValueError("Empty time slice. Check start/end are within network snapshots.")

    # -------------------------
    # 1) LOAD (dotted black line)
    # -------------------------
    load_name = f"load_{bus}"
    if load_name not in n.loads.index:
        raise KeyError(f"Load '{load_name}' not found. Available loads: {list(n.loads.index)[:10]} ...")

    load = n.loads_t.p_set.loc[t, load_name].astype(float)  # MW, positive demand

    # -------------------------
    # 2) GENERATION (positive stack)
    # -------------------------
    gens = n.generators.index[n.generators.bus == bus]
    if len(gens) == 0:
        gen_by_carrier = pd.DataFrame(index=t)
    else:
        gp = n.generators_t.p.loc[t, gens]
        gcar = n.generators.loc[gens, "carrier"].astype(str)
        gen_by_carrier = gp.T.groupby(gcar).sum().T

    # optional renaming / bucketing of generator carriers
    if rename_carriers:
        gen_by_carrier = gen_by_carrier.rename(columns=rename_carriers).groupby(level=0, axis=1).sum()

    # -------------------------
    # 3) STORAGE (charge negative, discharge positive)
    # -------------------------
    # charging links: grid bus0=province -> storage bus1=...
    chg_links = n.links.index[
        (n.links.bus0 == bus) & (n.links.carrier.astype(str).str.endswith("_charge"))
    ]
    charging = n.links_t.p0.loc[t, chg_links].sum(axis=1) if len(chg_links) else pd.Series(0.0, index=t)

    # discharging links: storage bus0=... -> grid bus1=province
    dis_links = n.links.index[
        (n.links.bus1 == bus) & (n.links.carrier.astype(str).str.endswith("_discharge"))
    ]
    discharging = _sum_links_p0_times_eff(n, dis_links).loc[t] if len(dis_links) else pd.Series(0.0, index=t)

    # -------------------------
    # 4) INTER-PROVINCE TRANSFERS (imports positive, exports negative)
    # -------------------------
    net_links = n.links.index[n.links.carrier.astype(str).isin(["ac", "dc"])]

    exp_links = n.links.index[(n.links.bus0 == bus) & (n.links.index.isin(net_links))]
    imp_links = n.links.index[(n.links.bus1 == bus) & (n.links.index.isin(net_links))]

    exports = n.links_t.p0.loc[t, exp_links].sum(axis=1) if len(exp_links) else pd.Series(0.0, index=t)
    imports = _sum_links_p0_times_eff(n, imp_links).loc[t] if len(imp_links) else pd.Series(0.0, index=t)

    # -------------------------
    # 5) Assemble plot stacks
    # -------------------------
    # positive components
    pos = gen_by_carrier.copy()
    pos["Storage discharge"] = discharging
    pos["Imports"] = imports

    # negative components (make them negative for plotting)
    neg = pd.DataFrame(index=t)
    neg["Storage charge"] = -charging
    neg["Exports"] = -exports

    # order (optional)
    if gen_carrier_order:
        # keep only those present, then append remaining
        cols = [c for c in gen_carrier_order if c in pos.columns]
        cols += [c for c in pos.columns if c not in cols]
        pos = pos[cols]

    # drop all-zero columns for readability
    pos = pos.loc[:, (pos.abs().sum(axis=0) > 1e-6)]
    neg = neg.loc[:, (neg.abs().sum(axis=0) > 1e-6)]

    # -------------------------
    # 6) Plot
    # -------------------------
    fig, ax = plt.subplots(figsize=figsize)

    x = t

    # positive stack
    if pos.shape[1] > 0:
        ax.stackplot(x, pos.T.values, labels=pos.columns, alpha=0.85)

    # negative stack
    if neg.shape[1] > 0:
        ax.stackplot(x, neg.T.values, labels=neg.columns, alpha=0.85)

    # load dotted black line (positive)
    ax.plot(x, load.values, "k:", linewidth=2.0, label="Load")

    ax.axhline(0.0, linewidth=1.0)
    ax.set_ylabel("MW")
    ax.set_title(f"Balance area: {bus} | {pd.Timestamp(start)} → {pd.Timestamp(end)}")
    ax.legend(ncol=3, loc="upper left", frameon=False)
    ax.margins(x=0)
    plt.tight_layout()
    
    #plt.show()
    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)
    fname = f"balance_{bus}_{pd.Timestamp(start).date()}_{pd.Timestamp(end).date()}.png"
    out_path = os.path.join(out_dir, fname)
    plt.savefig(out_path, dpi=300)
    print(f"Saved figure to: {out_path}")

    # sanity check (optional): residual of balance (should be ~0 if you included everything relevant)
    # NOTE: load shedding is a Generator carrier ("load_shedding") so it is already in gen_by_carrier.
    balance_residual = pos.sum(axis=1) + neg.sum(axis=1) - load
    print("Balance residual (MW):",
          f"mean={balance_residual.mean():.3f},",
          f"maxabs={balance_residual.abs().max():.3f}")


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    plot_balance_area(
        nc_path="results/dispatch_2025.nc",
        bus="Guangdong",
        start="2025-01-01 00:00:00",
        end="2025-01-07 23:00:00",
        gen_carrier_order=[
            "coal", "gas", "nuclear", "hydro", "onwind", "offwind", "solar", "other", "load_shedding",
            "Storage discharge", "Imports"
        ],
        rename_carriers=None,   # optionally map raw carriers -> display buckets
    )





