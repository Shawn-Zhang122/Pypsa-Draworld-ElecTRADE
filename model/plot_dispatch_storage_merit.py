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


def node_terms(n, t, bus):
    # load
    L = float(n.loads_t.p_set.loc[t, f"load_{bus}"])

    # interprovincial links (ac/dc)
    net_links = n.links.index[n.links.carrier.astype(str).isin(["ac", "dc"])]

    exp_links = n.links.index[(n.links.bus0 == bus) & (n.links.index.isin(net_links))]
    imp_links = n.links.index[(n.links.bus1 == bus) & (n.links.index.isin(net_links))]

    Ex = float(n.links_t.p0.loc[t, exp_links].sum()) if len(exp_links) else 0.0
    Im = _sum_links_p0_times_eff(n, t, imp_links) if len(imp_links) else 0.0

    # storage charge/discharge links
    chg_links = n.links.index[(n.links.bus0 == bus) & (n.links.carrier.astype(str).str.endswith("_charge"))]
    dis_links = n.links.index[(n.links.bus1 == bus) & (n.links.carrier.astype(str).str.endswith("_discharge"))]

    Ch = float(n.links_t.p0.loc[t, chg_links].sum()) if len(chg_links) else 0.0
    Dis = _sum_links_p0_times_eff(n, t, dis_links) if len(dis_links) else 0.0

    # power limits for “interior” test
    Dis_cap = float(n.links.loc[dis_links, "p_nom"].sum()) if len(dis_links) else 0.0

    return {
        "Load": L, "Exports": Ex, "Imports": Im,
        "Charge": Ch, "Discharge": Dis,
        "Discharge_cap": Dis_cap
    }


def generator_merit_stack(n, t, bus, drop_carriers=("load_shedding",), eps=1e-9):
    gens = n.generators[n.generators.bus == bus].copy()
    if len(gens) == 0:
        return pd.DataFrame(columns=["cap", "mc", "carrier"])

    gens = gens[~gens.carrier.isin(drop_carriers)]

    # availability
    pmax = pd.Series(1.0, index=gens.index)
    if hasattr(n.generators_t, "p_max_pu") and n.generators_t.p_max_pu is not None:
        common = gens.index.intersection(n.generators_t.p_max_pu.columns)
        pmax.loc[common] = n.generators_t.p_max_pu.loc[t, common]
    cap = (gens.p_nom * pmax).astype(float)
    cap = cap[cap > eps]
    gens = gens.loc[cap.index]

    # marginal cost (robust)
    mc = pd.Series(index=gens.index, dtype=float)
    if hasattr(n.generators_t, "marginal_cost") and n.generators_t.marginal_cost is not None:
        common = gens.index.intersection(n.generators_t.marginal_cost.columns)
        mc.loc[common] = n.generators_t.marginal_cost.loc[t, common]
    mc = mc.fillna(gens.marginal_cost).fillna(0.0)

    df = pd.DataFrame({"cap": cap, "mc": mc, "carrier": gens.carrier.astype(str)})
    df = df.sort_values("mc")
    df["cum"] = df.cap.cumsum()
    return df


def plot_geis_equivalent_SD(
    nc_path,
    hour,
    bus,
    tol=1e-3,
    price_eps=1e-2,
    save=True,
    out_dir="results"
):
    n = pypsa.Network(nc_path)
    t = pd.Timestamp(hour)
    if t not in n.snapshots:
        raise ValueError("hour not in snapshots")

    price = float(n.buses_t.marginal_price.loc[t, bus])

    terms = node_terms(n, t, bus)

    # Equivalent demand at node (net of imports, includes storage charge, excludes storage discharge)
    Qd = terms["Load"] + terms["Exports"] + terms["Charge"] - terms["Imports"]

    # Local generator stack
    gstack = generator_merit_stack(n, t, bus)

    # Storage “price-setting” condition (simple & practical):
    # - discharging is interior (0 < Dis < Dis_cap)
    # - and nodal price is not explained by local generator clearing alone (optional)
    Dis = terms["Discharge"]
    Dis_cap = terms["Discharge_cap"]

    storage_interior = (Dis > tol) and (Dis_cap > tol) and (Dis < Dis_cap - tol)

    # Compute local-only clearing price (if local stack can meet Qd)
    local_clearing_mc = np.nan
    if len(gstack) and gstack["cum"].iloc[-1] >= Qd:
        local_clearing_mc = float(gstack.loc[gstack["cum"] >= Qd, "mc"].iloc[0])

    storage_sets_price = storage_interior and (
        (not np.isfinite(local_clearing_mc)) or (abs(price - local_clearing_mc) > price_eps)
    )

    # Build supply stack for plotting
    # If storage is marginal: add a block at mc=price with cap=Dis (so intersection can lie on storage)
    supply = gstack.copy()
    if storage_sets_price:
        supply = pd.concat(
            [supply, pd.DataFrame([{"cap": Dis, "mc": price, "carrier": "storage_discharge"}])],
            ignore_index=True
        ).sort_values("mc")
        supply["cum"] = supply["cap"].cumsum()

    # Plot
    fig, ax = plt.subplots(figsize=(11, 6))
    if len(supply):
        ax.step(supply["cum"].values, supply["mc"].values, where="post", linewidth=2.0, label="Supply stack (node)")

    ax.axvline(Qd, linestyle="--", color="black", label=f"Demand Qd={Qd:.0f} MW")
    ax.axhline(price, linestyle="--", color="red", label=f"Price λ={price:.2f}")

    # mark intersection point (Qd, price)
    ax.scatter(Qd, price, color="red", zorder=6)

    ax.set_xlabel("MW")
    ax.set_ylabel("RMB/MWh")
    ax.set_title(f"Equivalent Local Supply – Net Demand – {bus} – {t}")

    # annotate storage status
    note = f"storage_dis={Dis:.0f}MW cap={Dis_cap:.0f}MW interior={storage_interior}"
    if storage_sets_price:
        note += " | storage marginal block added"
    else:
        note += " | no storage marginal block"
    ax.text(0.01, 0.02, note, transform=ax.transAxes)

    ax.legend(frameon=False)
    plt.tight_layout()

    if save:
        os.makedirs(out_dir, exist_ok=True)
        fname = f"SD_geis_{bus}_{t}.png".replace(":", "-").replace(" ", "_")
        path = os.path.join(out_dir, fname)
        plt.savefig(path, dpi=300)
        plt.close()
        print("Saved:", path)
    else:
        plt.show()

    print("=== Decomposition ===")
    print("Price λ:", price)
    print("Qd (Load+Ex+Ch-Im):", Qd)
    print("Terms:", terms)
    print("Local clearing MC (no storage block):", local_clearing_mc)
    print("storage/imports_sets_price:", storage_sets_price)


if __name__ == "__main__":
    plot_geis_equivalent_SD(
        nc_path="results/dispatch_2025.nc",
        hour="2025-07-15 18:00:00",
        bus="Guangdong",
        save=True
    )
