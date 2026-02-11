"""
Price setter classification from a solved PyPSA NetCDF (no PyPSA import needed).

What this does
--------------
For a given bus and time window, it:
1) reads LMP = buses_t.marginal_price (dual of nodal balance),
2) tests *interior* KKT conditions to infer which variable is marginal:
   - Local generator dispatch (p in (0, p_max)) with mc ~= LMP
   - Import link flow (p0 in (0, p_nom)) with correct link KKT
   - Storage discharge link (p0 in (0, p_nom)) with correct link KKT
3) exports:
   - a CSV with hourly setter labels
   - a scatter plot of price colored by setter

Key correction vs your draft
----------------------------
For a Link with bus0 -> bus1 and efficiency eta, and decision variable p0:

    bus0 injection:   -p0
    bus1 injection:   +eta * p0

Interior KKT (reduced cost = 0) gives:

    marginal_cost - LMP_bus0 + eta * LMP_bus1 = 0
=>  LMP_bus1 = (LMP_bus0 - marginal_cost) / eta

Your draft used: LMP_bus1 = eta * LMP_bus0 + marginal_cost  (wrong sign / scaling)

Notes / limitations
-------------------
- Your NetCDF does NOT store link marginal_cost (it was all zeros in your run),
  so we assume link marginal_cost = 0.0.
- Generator marginal_cost can be time-dependent; we use generators_t.marginal_cost if present.
- If multiple candidates match, we prefer a coal unit when its MC ties with others (your request).
- If nothing matches strict interior KKT tests, we label "Other" (typically binding bounds / congestion).
"""

from __future__ import annotations
import os
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def _effective_nom(nom: pd.Series, opt: pd.Series) -> pd.Series:
    nom = nom.astype(float).copy()
    opt = opt.astype(float)
    m = opt.notna() & (opt > 0)
    nom.loc[m] = opt.loc[m]
    return nom


def classify_price_setter(
    nc_path: str,
    bus: str,
    start: str,
    end: str,
    tol: float = 1e-3,
    eps: float = 1e-2,
    prefer_carrier: str = "coal",
) -> pd.DataFrame:
    ds = xr.open_dataset(nc_path)
    tindex = pd.to_datetime(ds["snapshots_snapshot"].values)

    # Prices
    lmp = ds["buses_t_marginal_price"].to_pandas()
    lmp.columns = ds["buses_t_marginal_price_i"].to_pandas()
    lmp.index = tindex

    # Generators (static)
    g_bus = ds["generators_bus"].to_pandas()
    g_car = ds["generators_carrier"].to_pandas()
    g_nom = ds["generators_p_nom"].to_pandas()
    g_nom_opt = ds["generators_p_nom_opt"].to_pandas()
    g_mc_static = ds["generators_marginal_cost"].to_pandas().astype(float)

    # Generators (time series)
    g_p = ds["generators_t_p"].to_pandas()
    g_p.columns = ds["generators_t_p_i"].to_pandas()
    g_p.index = tindex

    g_pmaxpu = None
    if "generators_t_p_max_pu" in ds:
        g_pmaxpu = ds["generators_t_p_max_pu"].to_pandas()
        g_pmaxpu.columns = ds["generators_t_p_max_pu_i"].to_pandas()
        g_pmaxpu.index = tindex

    g_mc_tv = None
    if "generators_t_marginal_cost" in ds:
        g_mc_tv = ds["generators_t_marginal_cost"].to_pandas()
        g_mc_tv.columns = ds["generators_t_marginal_cost_i"].to_pandas()
        g_mc_tv.index = tindex

    g_cap = _effective_nom(g_nom, g_nom_opt)

    def gen_mc_at(t, gens):
        mc = pd.Series(index=gens, dtype=float)
        if g_mc_tv is not None:
            common = gens.intersection(g_mc_tv.columns)
            if len(common):
                mc.loc[common] = g_mc_tv.loc[t, common]
        missing = mc[mc.isna()].index
        if len(missing):
            mc.loc[missing] = g_mc_static.loc[missing]
        return mc.fillna(0.0)

    def gen_pmax_at(t, gens):
        pmax = pd.Series(1.0, index=gens)
        if g_pmaxpu is not None:
            common = gens.intersection(g_pmaxpu.columns)
            if len(common):
                pmax.loc[common] = g_pmaxpu.loc[t, common]
        return pmax

    # Links
    l_bus0 = ds["links_bus0"].to_pandas()
    l_bus1 = ds["links_bus1"].to_pandas()
    l_car = ds["links_carrier"].to_pandas()
    l_eta = ds["links_efficiency"].to_pandas().astype(float)
    l_nom = ds["links_p_nom"].to_pandas().astype(float)
    l_nom_opt = ds["links_p_nom_opt"].to_pandas().astype(float)
    l_cap = _effective_nom(l_nom, l_nom_opt)

    l_p0 = ds["links_t_p0"].to_pandas()
    l_p0.columns = ds["links_t_p0_i"].to_pandas()
    l_p0.index = tindex

    # Stores (only for reporting mu_energy; classification is mainly via discharge-link KKT)
    s_bus = ds["stores_bus"].to_pandas()
    s_mu = ds["stores_t_mu_energy"].to_pandas()
    s_mu.columns = ds["stores_t_mu_energy_i"].to_pandas()
    s_mu.index = tindex

    window = lmp.loc[start:end].index

    def classify_one(t):
        lam = float(lmp.loc[t, bus])

        # 0) Load shedding (VOLL), if present
        gens_bus = g_bus[g_bus == bus].index
        for g in gens_bus:
            if str(g_car.loc[g]) in ("load_shedding", "load_shed") or str(g).startswith("load_shed"):
                if float(g_p.loc[t, g]) > tol:
                    return ("VOLL", "load_shedding", g, lam)

        # 1) Local generator, strict interior KKT
        if len(gens_bus):
            p = g_p.loc[t, gens_bus]
            cap = g_cap.loc[gens_bus] * gen_pmax_at(t, gens_bus)
            mc = gen_mc_at(t, gens_bus)

            interior = (p > tol) & (p < cap - tol)
            close = (mc - lam).abs() < eps

            cand = gens_bus[interior & close]
            if len(cand):
                carriers = g_car.loc[cand].astype(str)
                if prefer_carrier in carriers.values:
                    coal_cand = cand[carriers == prefer_carrier]
                    pick = (mc.loc[coal_cand] - lam).abs().idxmin()
                else:
                    pick = (mc.loc[cand] - lam).abs().idxmin()
                return ("Local", str(g_car.loc[pick]), pick, lam)

        # 2) Import (ac/dc), strict interior KKT for links
        # Interior KKT:  marginal_cost - LMP0 + eta*LMP1 = 0
        # Here link marginal_cost is not stored -> assume 0
        incoming = l_bus1[(l_bus1 == bus) & (l_car.isin(["ac", "dc"]))].index
        for link in incoming:
            cap = float(l_cap.loc[link])
            if cap <= 0:
                continue
            p0 = float(l_p0.loc[t, link])
            if (p0 > tol) and (p0 < cap - tol):
                b0 = l_bus0.loc[link]
                lam0 = float(lmp.loc[t, b0])
                eta = float(l_eta.loc[link]) if l_eta.loc[link] not in (0, np.nan) else 1.0

                lam1_implied = lam0 / eta  # (lam0 - c)/eta, with c=0
                if abs(lam - lam1_implied) < eps:
                    return ("Import", "ac/dc", link, lam)

        # 3) Storage discharge-link, strict interior KKT
        dis_links = l_bus1[(l_bus1 == bus) & (l_car.astype(str).str.endswith("_discharge"))].index
        for link in dis_links:
            cap = float(l_cap.loc[link])
            if cap <= 0:
                continue
            p0 = float(l_p0.loc[t, link])
            if (p0 > tol) and (p0 < cap - tol):
                b0 = l_bus0.loc[link]  # storage bus
                lam0 = float(lmp.loc[t, b0])
                eta = float(l_eta.loc[link]) if not pd.isna(l_eta.loc[link]) else 1.0

                # KKT: -LMP0 + eta*LMP1 + c = 0  -> LMP0 = eta*LMP1 + c
                if abs(lam0 - eta * lam) < eps:
                    stores = s_bus[s_bus == b0].index
                    if len(stores):
                        mu = float(s_mu.loc[t, stores[0]])
                        return ("Storage", f"mu_energy={mu:.2f}", link, lam)
                    return ("Storage", "discharge_link", link, lam)

        return ("Other", "constraint", None, lam)

    rows = [classify_one(t) for t in window]
    df = pd.DataFrame(rows, index=window, columns=["setter", "driver", "asset", "price"])
    return df


def plot_setters(df: pd.DataFrame, title: str, out_png: str) -> None:
    colors = {
        "Local": "tab:blue",
        "Import": "tab:green",
        "Storage": "tab:purple",
        "VOLL": "tab:red",
        "Other": "tab:gray",
    }
    fig, ax = plt.subplots(figsize=(14, 4.8))
    for k, g in df.groupby("setter"):
        ax.scatter(g.index, g["price"], s=10, label=k, color=colors.get(k, "black"))
    ax.plot(df.index, df["price"], alpha=0.25)

    ax.set_ylabel("RMB/MWh")
    ax.set_title(title)
    ax.legend(frameon=False, ncol=5)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    nc = "results/China_33nodes_dispatch_2025.nc"
    bus = "Guangdong"
    start = "2025-07-15 00:00:00"
    end = "2025-07-21 23:00:00"

    df = classify_price_setter(nc, bus, start, end, tol=1e-3, eps=1e-2, prefer_carrier="coal")

    os.makedirs("results_price_setter", exist_ok=True)
    out_csv = os.path.join("results", f"price_setter_{bus}_{start[:10]}_{end[:10]}.csv")
    out_png = os.path.join("results", f"price_setter_{bus}_{start[:10]}_{end[:10]}.png")

    df.to_csv(out_csv)
    plot_setters(df, f"Nodal price & inferred price setter ({bus})", out_png)

    print("Saved:", out_csv)
    print("Saved:", out_png)
