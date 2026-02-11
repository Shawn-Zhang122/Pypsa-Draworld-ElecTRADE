import pypsa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def _sum_links_p0_times_eff(n, t, link_names):
    if len(link_names) == 0:
        return 0.0
    p0 = n.links_t.p0.loc[t, link_names]
    eff = n.links.loc[link_names, "efficiency"].astype(float)
    return float((p0 * eff.values).sum())


def storage_flows_at_bus(n, t, bus):
    # charging: grid bus0=bus -> storage bus1
    chg_links = n.links.index[
        (n.links.bus0 == bus) & (n.links.carrier.astype(str).str.endswith("_charge"))
    ]
    chg = float(n.links_t.p0.loc[t, chg_links].sum()) if len(chg_links) else 0.0

    # discharging: storage bus0 -> grid bus1=bus (delivered to grid is p0*eff)
    dis_links = n.links.index[
        (n.links.bus1 == bus) & (n.links.carrier.astype(str).str.endswith("_discharge"))
    ]
    dis = _sum_links_p0_times_eff(n, t, dis_links) if len(dis_links) else 0.0

    return chg, dis


def local_merit_slope(n, t, bus, eps=1e-6):
    """
    Approximate local slope d(lambda)/d(D) by building local merit curve at bus.
    Uses available capacity and marginal cost at time t.
    Returns slope in (RMB/MWh)/MW near clearing.
    """
    gens = n.generators[n.generators.bus == bus].copy()
    gens = gens[gens.carrier != "load_shedding"]

    if len(gens) == 0:
        return np.nan

    # availability
    pmax = pd.Series(1.0, index=gens.index)
    if hasattr(n.generators_t, "p_max_pu") and n.generators_t.p_max_pu is not None:
        common = gens.index.intersection(n.generators_t.p_max_pu.columns)
        pmax.loc[common] = n.generators_t.p_max_pu.loc[t, common]
    cap = gens.p_nom * pmax
    cap = cap[cap > eps]

    # MC robust
    mc = pd.Series(index=cap.index, dtype=float)
    if hasattr(n.generators_t, "marginal_cost") and n.generators_t.marginal_cost is not None:
        common = cap.index.intersection(n.generators_t.marginal_cost.columns)
        mc.loc[common] = n.generators_t.marginal_cost.loc[t, common]
    mc = mc.fillna(n.generators.loc[cap.index, "marginal_cost"]).fillna(0.0)

    df = pd.DataFrame({"cap": cap, "mc": mc}).sort_values("mc")
    df["cum"] = df.cap.cumsum()

    # use nodal net demand (load + exports + charge - imports - discharge)
    L = float(n.loads_t.p_set.loc[t, f"load_{bus}"])
    # network imports/exports
    net_links = n.links.index[n.links.carrier.astype(str).isin(["ac", "dc"])]
    exp_links = n.links.index[(n.links.bus0 == bus) & (n.links.index.isin(net_links))]
    imp_links = n.links.index[(n.links.bus1 == bus) & (n.links.index.isin(net_links))]
    Ex = float(n.links_t.p0.loc[t, exp_links].sum()) if len(exp_links) else 0.0
    Im = _sum_links_p0_times_eff(n, t, imp_links) if len(imp_links) else 0.0
    chg, dis = storage_flows_at_bus(n, t, bus)

    Dnet = L + Ex + chg - Im - dis

    # find segment around Dnet
    k = df[df.cum >= Dnet]
    if len(k) == 0:
        return np.nan
    idx = k.index[0]

    # slope of a step function is infinite at jumps and 0 on flats.
    # For a practical "local slope", use finite difference across nearest jump:
    # pick previous step MC and next step MC around the clearing point.
    pos = df.index.get_loc(idx)
    if pos == 0:
        return np.nan

    mc_prev = df.iloc[pos - 1].mc
    mc_now = df.iloc[pos].mc
    cap_prev = df.iloc[pos - 1].cum
    cap_now = df.iloc[pos].cum
    dmc = mc_now - mc_prev
    dcap = cap_now - cap_prev

    if dcap <= eps:
        return np.nan
    return float(dmc / dcap)


def plot_storage_price_role(nc_path, bus, start, end, save=True, out_dir="results"):
    n = pypsa.Network(nc_path)
    T = n.snapshots[(n.snapshots >= pd.Timestamp(start)) & (n.snapshots <= pd.Timestamp(end))]
    if len(T) == 0:
        raise ValueError("Empty slice.")

    price = n.buses_t.marginal_price.loc[T, bus]

    chg = []
    dis = []
    slope = []
    dprice_hat = []  # approx price effect of storage (RMB/MWh)

    for t in T:
        c, d = storage_flows_at_bus(n, t, bus)
        chg.append(c)
        dis.append(d)

        s = local_merit_slope(n, t, bus)
        slope.append(s)

        # approx Δλ from storage net injection (dis - chg)
        # net injection reduces net demand by (dis - chg)
        if np.isfinite(s):
            dprice_hat.append(s * (-(d - c)))
        else:
            dprice_hat.append(np.nan)

    chg = pd.Series(chg, index=T)
    dis = pd.Series(dis, index=T)
    slope = pd.Series(slope, index=T)
    dprice_hat = pd.Series(dprice_hat, index=T)

    fig, ax1 = plt.subplots(figsize=(14, 6))

    # price
    ax1.plot(T, price.values, label="Nodal price (RMB/MWh)")
    ax1.set_ylabel("RMB/MWh")

    # storage on 2nd axis
    ax2 = ax1.twinx()
    ax2.bar(T, dis.values, width=0.03, label="Storage discharge (MW)", alpha=0.4)
    ax2.bar(T, -chg.values, width=0.03, label="Storage charge (MW)", alpha=0.4)
    ax2.set_ylabel("MW (+discharge / -charge)")

    ax1.set_title(f"Storage role in price – {bus} – {start} → {end}")

    # legend merge
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", ncol=3, frameon=False)

    plt.tight_layout()

    if save:
        os.makedirs(out_dir, exist_ok=True)
        fname = f"storage_price_role_{bus}_{start}_{end}.png".replace(":", "-").replace(" ", "_")
        path = os.path.join(out_dir, fname)
        plt.savefig(path, dpi=300)
        plt.close()
        print("Saved:", path)
    else:
        plt.show()

    # output a compact table for inspection
    out = pd.DataFrame({
        "price": price,
        "charge_MW": chg,
        "discharge_MW": dis,
        "local_slope_(RMB/MWh)/MW": slope,
        "approx_storage_price_effect_RMB/MWh": dprice_hat
    })
    csv_path = os.path.join(out_dir, f"storage_price_role_{bus}_{start}_{end}.csv".replace(":", "-").replace(" ", "_"))
    out.to_csv(csv_path)
    print("Saved table:", csv_path)


if __name__ == "__main__":
    plot_storage_price_role(
        nc_path="results/dispatch_2025.nc",
        bus="Guangdong",
        start="2025-07-15 00:00:00",
        end="2025-07-21 23:00:00",
        save=True
    )
