"""
Normalized cross-city comparison.
Adds area/population normalization and fixed-quantile comparison table.
No recomputation needed — uses existing betweenness.csv and existing.csv.

Usage: conda run -n growbikenet python analysis_normalized.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

BASE = Path(__file__).resolve().parent.parent / "bikenwgrowth-data"
COMP = BASE / "analysis_output" / "comparison"
COMP.mkdir(parents=True, exist_ok=True)

CITIES = ["amsterdam", "barcelona", "berlin", "oslo", "vienna"]
LABELS = {
    "amsterdam": "Amsterdam",
    "barcelona": "Barcelona",
    "berlin":    "Berlin",
    "oslo":      "Oslo",
    "vienna":    "Vienna",
}
COLORS = {
    "amsterdam": "#e41a1c",
    "barcelona": "#377eb8",
    "berlin":    "#4daf4a",
    "oslo":      "#984ea3",
    "vienna":    "#ff7f00",
}

# Population per city (kept hardcoded — no spatial source needed).
POP = {
    "amsterdam":   921_000,
    "barcelona": 1_620_000,
    "berlin":    3_760_000,
    "oslo":        717_000,
    "vienna":    1_930_000,
}

# Area (km²) is read from city_boundaries.csv, which is produced by
# compute_effective_boundaries.py and uses the EXACT same nominatim polygon
# the bikenwgrowth algorithm uses, minus large forest/park areas
# (Wienerwald, Marka, Grunewald, …).
_boundary_csv = COMP / "city_boundaries.csv"
if not _boundary_csv.exists():
    raise FileNotFoundError(
        f"Missing {_boundary_csv}.  Run `python compute_effective_boundaries.py` first."
    )
_boundary_df = pd.read_csv(_boundary_csv).set_index("city")

META = {}
for c in CITIES:
    row = _boundary_df.loc[c]
    META[c] = dict(
        area_km2          = float(row["effective_km2"]),
        area_nominatim_km2= float(row["nominatim_km2"]),
        forest_km2        = float(row["forest_km2"]),
        pop               = POP[c],
    )

FIXED_QUANTILES = [0.25, 0.50, 0.75, 1.00]

# ── Load data ────────────────────────────────────────────────────────────────
data = {}
for city in CITIES:
    syn = pd.read_csv(BASE / "results" / city / f"{city}_poi_grid_betweenness.csv")
    ex  = pd.read_csv(BASE / "results" / city / f"{city}_existing.csv")
    n = len(syn)
    syn["quantile"]  = [round(i/n, 4) for i in range(1, n+1)]
    syn["length_km"] = syn["length"] / 1000
    syn["lcc_share"] = syn["length_lcc"] / syn["length"]
    bt = ex[ex["network"] == "biketrack"].iloc[0]
    bk = ex[ex["network"] == "bikeable"].iloc[0]
    data[city] = {"syn": syn, "biketrack": bt, "bikeable": bk,
                  **META[city]}

# ── 1 · Density table (km per km²) ──────────────────────────────────────────
print("=== Network density (km per km²) ===\n")
density_rows = []
for city in CITIES:
    d = data[city]
    syn_max = d["syn"]["length_km"].max()
    bt_km   = d["biketrack"]["length"] / 1000
    bk_km   = d["bikeable"]["length"]  / 1000
    area    = d["area_km2"]
    pop     = d["pop"]
    density_rows.append({
        "City":                 LABELS[city],
        "Area (km²)":           round(area, 1),
        "Population":           f"{pop/1e6:.2f}M",
        "biketrack (km)":       f"{bt_km:.0f}",
        "biketrack / km²":      f"{bt_km/area:.2f}",
        "bikeable (km)":        f"{bk_km:.0f}",
        "bikeable / km²":       f"{bk_km/area:.2f}",
        "Syn max (km)":         f"{syn_max:.0f}",
        "Syn max / km²":        f"{syn_max/area:.2f}",
        "Syn/bikeable ratio":   f"{syn_max/bk_km:.0%}",
    })
den_df = pd.DataFrame(density_rows)
print(den_df.to_string(index=False))
den_df.to_csv(COMP / "density_table.csv", index=False)

# ── 2 · Fixed-quantile metrics table ────────────────────────────────────────
print("\n=== Metrics at fixed quantiles ===\n")
quant_rows = []
for city in CITIES:
    syn = data[city]["syn"]
    for q in FIXED_QUANTILES:
        # find closest row to target quantile
        idx = (syn["quantile"] - q).abs().idxmin()
        row = syn.loc[idx]
        quant_rows.append({
            "City":          LABELS[city],
            "Quantile":      f"Q={q:.2f}",
            "Length (km)":   f"{row['length_km']:.0f}",
            "Components":    int(row["components"]),
            "LCC share":     f"{row['lcc_share']:.1%}",
            "Directness":    f"{row['directness_lcc']:.3f}",
            "Efficiency":    f"{row['efficiency_global']:.3f}",
            "POI coverage":  int(row["poi_coverage"]),
        })
q_df = pd.DataFrame(quant_rows)
print(q_df.to_string(index=False))
q_df.to_csv(COMP / "fixed_quantile_metrics.csv", index=False)

# ── 3 · Plot A: density bars ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("Normalized density — km of network per km² of effective city area",
             fontsize=12, fontweight="bold")

city_names = [LABELS[c] for c in CITIES]
colors     = [COLORS[c] for c in CITIES]

for ax, (col_key, title, ylabel) in zip(axes, [
    ("biketrack / km²", "A — biketrack (real)", "km of A / km² city"),
    ("bikeable / km²",  "B — bikeable (real)",  "km of B / km² city"),
    ("Syn max / km²",   "C — synthetic",        "km of C / km² city"),
]):
    vals = [float(den_df[den_df["City"] == LABELS[c]][col_key].values[0])
            for c in CITIES]
    bars = ax.bar(city_names, vals, color=colors, alpha=0.85, edgecolor="white")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", labelsize=9)

plt.tight_layout()
fig.savefig(COMP / "density_bars.png", dpi=150)
plt.close(fig)
print("\nSaved: density_bars.png")

# ── 4 · Plot B: normalized metrics at fixed quantiles ───────────────────────
METRIC_PANELS = [
    ("Directness (LCC)",    "directness_lcc",    (0.4, 0.9)),
    ("Global efficiency",   "efficiency_global", (0.2, 0.8)),
    ("LCC share",           "lcc_share",         (0.0, 1.05)),
]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(
    "Network quality at fixed growth quantiles — all cities\n"
    "(comparable regardless of city size)",
    fontsize=11, fontweight="bold"
)

x_pos = np.arange(len(FIXED_QUANTILES))
bar_w = 0.15

for ax, (title, metric, ylim) in zip(axes, METRIC_PANELS):
    for i, city in enumerate(CITIES):
        syn = data[city]["syn"]
        vals = []
        for q in FIXED_QUANTILES:
            idx = (syn["quantile"] - q).abs().idxmin()
            vals.append(syn.loc[idx, metric])
        offset = (i - 2) * bar_w
        bars = ax.bar(x_pos + offset, vals, bar_w,
                      label=LABELS[city], color=COLORS[city], alpha=0.85)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"Q={q:.2f}" for q in FIXED_QUANTILES])
    ax.set_ylabel(title)
    ax.set_title(title, fontweight="bold")
    ax.set_ylim(*ylim)
    if metric == "lcc_share":
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
fig.savefig(COMP / "quantile_comparison.png", dpi=150)
plt.close(fig)
print("Saved: quantile_comparison.png")

# ── 5 · Plot C: directness & efficiency growth curves, colour = city ─────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(
    "Quality metrics over synthetic growth (x = quantile, comparable across cities)",
    fontsize=11, fontweight="bold"
)

for ax, (metric, ylabel) in zip(axes, [
    ("directness_lcc",    "Directness within LCC"),
    ("efficiency_global", "Global efficiency"),
]):
    for city in CITIES:
        syn = data[city]["syn"]
        ax.plot(syn["quantile"], syn[metric],
                color=COLORS[city], lw=2, label=LABELS[city])

        # Mark biketrack reference if reachable
        bt_km  = data[city]["biketrack"]["length"] / 1000
        syn_max = syn["length_km"].max()
        if bt_km <= syn_max:
            idx = (syn["length_km"] - bt_km).abs().idxmin()
            row = syn.loc[idx]
            ax.scatter(row["quantile"], row[metric],
                       color=COLORS[city], s=60, zorder=5, marker="D")

    ax.set_xlabel("Growth quantile (0 = start, 1 = max)")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)

fig.text(0.5, 0.01,
         "◆ = point where synthetic length matches OSM biketrack (only shown if reachable)",
         ha="center", fontsize=8, color="#555")

plt.tight_layout(rect=[0, 0.03, 1, 1])
fig.savefig(COMP / "quality_curves_normalized.png", dpi=150)
plt.close(fig)
print("Saved: quality_curves_normalized.png")

# ── 6 · Summary: the meaningful comparison table ─────────────────────────────
print("\n=== MEANINGFUL COMPARISON (normalized) ===\n")
final_rows = []
for city in CITIES:
    d   = data[city]
    syn = d["syn"]
    end = syn.iloc[-1]
    bt_km = d["biketrack"]["length"] / 1000
    area  = d["area_km2"]
    pop   = d["pop"]
    sp = pd.read_csv(COMP / "spatial_summary.csv")
    sp_row = sp[sp["City"] == LABELS[city]]
    ov_pct = sp_row["Overlap (%)"].values[0] if len(sp_row) else "n/a"
    bk_km = d["bikeable"]["length"] / 1000

    final_rows.append({
        "City":                   LABELS[city],
        "Area (km²)":             round(area, 1),
        "Syn/km² (synthetic)":    f"{end['length_km']/area:.2f}",
        "bk/km² (OSM)":           f"{bk_km/area:.2f}",
        "Directness @ Q=1.0":     f"{end['directness_lcc']:.3f}",
        "Efficiency @ Q=1.0":     f"{end['efficiency_global']:.3f}",
        "Directness @ Q=0.5":     f"{syn.loc[(syn['quantile']-0.5).abs().idxmin(),'directness_lcc']:.3f}",
        "Overlap vs. bikeable":   ov_pct,
    })

final_df = pd.DataFrame(final_rows)
print(final_df.to_string(index=False))
final_df.to_csv(COMP / "normalized_comparison.csv", index=False)
print("\nAll saved to analysis_output/comparison/")
print("Done.")
