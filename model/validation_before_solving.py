# validation_before_solving.py
# ------------------------------------------------------------
# Pre-solve sanity checks & capacity summaries for PyPSA
# Print-only. No mutation of network.
# ------------------------------------------------------------

import pandas as pd
import numpy as np


def _num(x):
    return pd.to_numeric(x, errors="coerce")


def _print(title, obj):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print(obj if obj is not None and len(obj) else "(empty)")


def validation_before_solving(n):
    # ======================
    # 0. BASIC INFO
    # ======================
    print("\n" + "#" * 80)
    print("PRE-SOLVE VALIDATION")
    print("#" * 80)

    print(
        f"snapshots={len(n.snapshots)}, "
        f"buses={len(n.buses)}, "
        f"gens={len(n.generators)}, "
        f"links={len(n.links)}, "
        f"stores={len(n.stores)}, "
        f"loads={len(n.loads)}"
    )

    # ======================
    # 1. LOADS (peak by bus)
    # ======================
    if len(n.loads):
        load_bus = n.loads.bus
        peak_load = (
            n.loads_t.p_set
            .groupby(load_bus, axis=1)
            .sum()
            .max()
            .sort_values(ascending=False)
        )
        _print("LOAD peak by bus (MW)", peak_load)

    # ======================
    # 2. GENERATORS (p_nom)
    # ======================
    if len(n.generators):
        g = n.generators.copy()
        g["p_nom"] = _num(g.p_nom)

        _print(
            "GEN p_nom by bus (MW)",
            g.groupby("bus").p_nom.sum().sort_values(ascending=False)
        )

        _print(
            "GEN p_nom by bus × carrier (MW)",
            g.pivot_table(
                index="bus", columns="carrier",
                values="p_nom", aggfunc="sum", fill_value=0.0
            )
        )

        bad = g[(g.p_nom.isna()) | (g.p_nom < 0)]
        _print("GEN with NaN/negative p_nom (ERROR)", bad[["bus", "carrier", "p_nom"]])

        bad_minmax = g[g.p_min_pu > g.p_max_pu]
        _print("GEN with p_min_pu > p_max_pu (INFEASIBLE)", bad_minmax[
            ["bus", "carrier", "p_nom", "p_min_pu", "p_max_pu"]
        ])

    # ======================
    # 3. LINKS (p_nom)
    # ======================
    if len(n.links):
        l = n.links.copy()
        l["p_nom"] = _num(l.p_nom)

        _print(
            "LINK outbound p_nom by bus0 (MW)",
            l.groupby("bus0").p_nom.sum().sort_values(ascending=False)
        )

        _print(
            "LINK inbound p_nom by bus1 (MW)",
            l.groupby("bus1").p_nom.sum().sort_values(ascending=False)
        )

    # ======================
    # 4. STORES (e_nom)
    # ======================
    if len(n.stores):
        s = n.stores.copy()
        s["e_nom"] = _num(s.e_nom)

        _print(
            "STORE e_nom by bus (MWh)",
            s.groupby("bus").e_nom.sum().sort_values(ascending=False)
        )

    # ======================
    # 5. SYSTEM FEASIBILITY HINT
    # ======================
    if len(n.loads) and len(n.generators):
        peak = n.loads_t.p_set.sum(axis=1).max()
        total_cap = _num(n.generators.p_nom).sum()
        print("\nPeak load (MW):", round(peak, 2))
        print("Total gen p_nom (MW):", round(total_cap, 2))
        print("Cap / peak:", round(total_cap / peak, 3))

    print("\n" + "#" * 80)
    print("END PRE-SOLVE VALIDATION")
    print("#" * 80)

    #6. consistnecy check for marginal costs
    n.consistency_check()


