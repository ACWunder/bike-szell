"""
Multi-city bicycle network analysis: Amsterdam, Barcelona, Berlin, Oslo, Vienna
All cities use gridl=1701 data from results/{city}/

For each city:
  - Growth curves (length, components, LCC share, directness, efficiency)
  - OSM biketrack/bikeable as reference lines
  - Cross-city comparison plots (x-axis normalised to max synthetic length)
  - Summary comparison table

Usage: conda run -n growbikenet python analysis_multicity.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Config ─────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent / "bikenwgrowth-data"
OUT  = BASE / "analysis_output" / "comparison"
OUT.mkdir(parents=True, exist_ok=True)

CITIES = ["amsterdam", "barcelona", "berlin", "oslo", "vienna"]

CITY_LABELS = {
    "amsterdam": "Amsterdam",
    "barcelona": "Barcelona",
    "berlin":    "Berlin",
    "oslo":      "Oslo",
    "vienna":    "Vienna",
}

# Colour palette — one per city
COLORS = {
    "amsterdam": "#e41a1c",
    "barcelona": "#377eb8",
    "berlin":    "#4daf4a",
    "oslo":      "#984ea3",
    "vienna":    "#ff7f00",
}

# ── Load data ───────────────────────────────────────────────────────────────
print("Loading data...")
data = {}
for city in CITIES:
    syn = pd.read_csv(BASE / "results" / city / f"{city}_poi_grid_betweenness.csv")
    ex  = pd.read_csv(BASE / "results" / city / f"{city}_existing.csv")

    n = len(syn)
    syn["quantile"]  = [round(i / n, 4) for i in range(1, n + 1)]
    syn["length_km"] = syn["length"] / 1000
    syn["lcc_share"] = syn["length_lcc"] / syn["length"]

    biketrack = ex[ex["network"] == "biketrack"].iloc[0]
    bikeable  = ex[ex["network"] == "bikeable"].iloc[0]

    data[city] = {
        "syn":       syn,
        "biketrack": biketrack,
        "bikeable":  bikeable,
    }
    print(
        f"  {CITY_LABELS[city]:12s}: {n} steps, "
        f"syn_max={syn['length_km'].max():.0f} km | "
        f"biketrack={biketrack['length']/1000:.0f} km | "
        f"bikeable={bikeable['length']/1000:.0f} km"
    )


# ── Helper: find closest row to a target length ─────────────────────────────
def match_km(syn, target_km):
    """Return row closest to target_km; None if target > synthetic max."""
    if target_km > syn["length_km"].max():
        return None
    idx = (syn["length_km"] - target_km).abs().idxmin()
    return syn.loc[idx]


# ── 1 · Individual city growth-curve plots ──────────────────────────────────
print("\nPlotting individual cities...")

for city in CITIES:
    syn  = data[city]["syn"]
    bt_m = data[city]["biketrack"]["length"] / 1000   # km
    bk_m = data[city]["bikeable"]["length"]  / 1000
    syn_max = syn["length_km"].max()
    label = CITY_LABELS[city]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        f"Synthetic network growth — {label}  (gridl=1701)",
        fontsize=13, fontweight="bold"
    )

    panels = [
        ("Total length (km)",      "length_km",     axes[0, 0]),
        ("Connected components",   "components",     axes[0, 1]),
        ("LCC share",              "lcc_share",      axes[1, 0]),
        ("Directness (LCC)",       "directness_lcc", axes[1, 1]),
    ]

    for ylabel, col, ax in panels:
        ax.plot(syn["length_km"], syn[col],
                color=COLORS[city], lw=2, label="Synthetic")

        # Reference lines (only if within synthetic range)
        for ref_km, ref_label, ls in [
            (bt_m, "OSM biketrack", "--"),
            (bk_m, "OSM bikeable",  ":")
        ]:
            if ref_km <= syn_max:
                ax.axvline(ref_km, color="#555", ls=ls, lw=1.2, label=ref_label)
                # Mark the metric value at that point
                row = match_km(syn, ref_km)
                if row is not None:
                    ax.scatter(row["length_km"], row[col],
                               color="#555", s=40, zorder=5)
            else:
                # Show as arrow beyond x-axis
                ax.axvline(syn_max * 0.98, color="#bbb", ls=ls, lw=1,
                           label=f"{ref_label} ({ref_km:.0f} km > max)")

        ax.set_xlabel("Synthetic network length (km)")
        ax.set_ylabel(ylabel)
        if col == "lcc_share":
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = OUT / f"growth_curves_{city}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ── 2 · Cross-city comparison plots (x = % of max synthetic length) ─────────
print("\nPlotting cross-city comparisons (normalised x-axis)...")

COMPARE_PANELS = [
    ("LCC share",            "lcc_share",      "LCC share of total length"),
    ("Connected components", "components",      "Number of connected components"),
    ("Directness (LCC)",     "directness_lcc",  "Directness within LCC"),
    ("Global efficiency",    "efficiency_global","Global network efficiency"),
    ("POI coverage",         "poi_coverage",    "POI coverage (# POIs reached)"),
]

# 3 rows × 2 columns layout — one panel per metric, last cell holds a legend.
fig, axes = plt.subplots(3, 2, figsize=(11, 13))
fig.suptitle(
    "Cross-city comparison — synthetic network C growth (gridl=1701)",
    fontsize=13, fontweight="bold", y=0.995
)
axes_flat = axes.flatten()

for panel_idx, (title, col, ylabel) in enumerate(COMPARE_PANELS):
    ax = axes_flat[panel_idx]
    for city in CITIES:
        syn     = data[city]["syn"]
        syn_max = syn["length_km"].max()
        x = syn["length_km"] / syn_max * 100   # 0–100 %
        ax.plot(x, syn[col], color=COLORS[city], lw=2, label=CITY_LABELS[city])

        # Mark biketrack reference as dot on the curve (if reachable)
        bt_km = data[city]["biketrack"]["length"] / 1000
        if bt_km <= syn_max:
            row = match_km(syn, bt_km)
            if row is not None:
                ax.scatter(bt_km / syn_max * 100, row[col],
                           color=COLORS[city], s=50, zorder=5, marker="D")

    ax.set_xlabel("% of max synthetic length")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11, fontweight="bold")
    if col == "lcc_share":
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.grid(True, alpha=0.3)

# Last cell: shared legend + caption (no per-panel legend → less clutter)
ax_leg = axes_flat[5]
ax_leg.set_axis_off()
handles = [plt.Line2D([0], [0], color=COLORS[c], lw=3, label=CITY_LABELS[c])
           for c in CITIES]
handles.append(plt.Line2D([0], [0], marker="D", color="#555",
                          linestyle="None", markersize=8,
                          label="◆ C reaches network A length (where reachable)"))
ax_leg.legend(handles=handles, loc="center", fontsize=11, frameon=False,
              title="Legend", title_fontsize=12)

plt.tight_layout()
out_path = OUT / "comparison_all_cities.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {out_path.name}")


# ── 3 · Comparison bar chart at key reference points ────────────────────────
print("\nPlotting reference-point comparison bars...")

metrics_at_ref = []
for city in CITIES:
    syn     = data[city]["syn"]
    syn_max = syn["length_km"].max()
    label   = CITY_LABELS[city]

    for ref_name, ref_km in [
        ("biketrack", data[city]["biketrack"]["length"] / 1000),
        ("bikeable",  data[city]["bikeable"]["length"]  / 1000),
        ("syn_max",   syn_max),
    ]:
        row = match_km(syn, ref_km) if ref_km <= syn_max else syn.iloc[-1]
        reached = ref_km <= syn_max
        metrics_at_ref.append({
            "city":          label,
            "reference":     ref_name,
            "ref_km":        ref_km,
            "syn_km":        row["length_km"],
            "quantile":      row["quantile"],
            "coverage_pct":  min(row["length_km"] / ref_km * 100, 100) if ref_km > 0 else 0,
            "reached":       reached,
            "lcc_share":     row["lcc_share"],
            "components":    int(row["components"]),
            "directness_lcc": row["directness_lcc"],
            "efficiency_global": row["efficiency_global"],
            "poi_coverage":  row["poi_coverage"],
        })

ref_df = pd.DataFrame(metrics_at_ref)

# Bar chart: how much of OSM bikeable/biketrack does synthetic max cover?
# Bikeable is the primary reference (matches what the algorithm actually
# competes against — all legally cyclable infrastructure). Biketrack is
# shown as secondary for completeness.
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Synthetic network coverage of OSM reference networks",
             fontsize=12, fontweight="bold")

PRIMARY_REF = "bikeable"

for ax, ref in zip(axes, [PRIMARY_REF, "biketrack"]):
    sub = ref_df[ref_df["reference"] == ref].copy()
    bars = ax.bar(sub["city"], sub["coverage_pct"],
                  color=[COLORS[c] for c in CITIES], alpha=0.85, edgecolor="white")
    ax.axhline(100, color="black", lw=1, ls="--", label="100 % (= OSM length)")
    ax.set_ylabel("Synthetic max / OSM length (%)")
    title_suffix = "  ← primary reference" if ref == PRIMARY_REF else "  (secondary)"
    ax.set_title(f"vs. OSM {ref}{title_suffix}",
                 fontweight="bold" if ref == PRIMARY_REF else "normal")
    ax.set_ylim(0, max(sub["coverage_pct"].max() * 1.1, 110))
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    for bar, (_, row) in zip(bars, sub.iterrows()):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{row['coverage_pct']:.0f}%",
                ha="center", va="bottom", fontsize=9)

plt.tight_layout()
out_path = OUT / "coverage_bars.png"
fig.savefig(out_path, dpi=150)
plt.close(fig)
print(f"  Saved: {out_path.name}")


# ── 4 · Summary table ────────────────────────────────────────────────────────
print("\nBuilding summary table...")

# Table at syn_max endpoint
summary_rows = []
for city in CITIES:
    syn     = data[city]["syn"]
    row_end = syn.iloc[-1]
    bt_km   = data[city]["biketrack"]["length"] / 1000
    bk_km   = data[city]["bikeable"]["length"]  / 1000

    row_bt = match_km(syn, bt_km)   # None if not reached
    row_bk = match_km(syn, bk_km)   # None if not reached

    summary_rows.append({
        "City":               CITY_LABELS[city],
        "Syn max (km)":       f"{row_end['length_km']:.0f}",
        "OSM biketrack (km)": f"{bt_km:.0f}",
        "OSM bikeable (km)":  f"{bk_km:.0f}",
        "Syn covers biketrack": "yes" if bt_km <= row_end["length_km"] else f"no ({row_end['length_km']/bt_km*100:.0f}%)",
        "Syn covers bikeable":  "yes" if bk_km <= row_end["length_km"] else f"no ({row_end['length_km']/bk_km*100:.0f}%)",
        "Components @ syn_max": int(row_end["components"]),
        "LCC share @ syn_max":  f"{row_end['lcc_share']:.1%}",
        "Directness @ syn_max": f"{row_end['directness_lcc']:.3f}",
        "Efficiency @ syn_max": f"{row_end['efficiency_global']:.3f}",
    })

summary = pd.DataFrame(summary_rows)
print("\n" + summary.to_string(index=False))

summary.to_csv(OUT / "summary_table.csv", index=False)
print(f"\nSaved: {(OUT / 'summary_table.csv').name}")

# Detailed table at biketrack reference
detail_rows = []
for city in CITIES:
    syn   = data[city]["syn"]
    bt_km = data[city]["biketrack"]["length"] / 1000
    row   = match_km(syn, bt_km)
    note  = "" if row is not None else f"biketrack ({bt_km:.0f} km) > syn max ({syn['length_km'].max():.0f} km) → showing syn max"
    if row is None:
        row = syn.iloc[-1]
    detail_rows.append({
        "City":           CITY_LABELS[city],
        "Note":           note,
        "Syn km":         f"{row['length_km']:.0f}",
        "Quantile":       f"{row['quantile']:.3f}",
        "Components":     int(row["components"]),
        "LCC share":      f"{row['lcc_share']:.1%}",
        "Directness LCC": f"{row['directness_lcc']:.3f}",
        "Eff. global":    f"{row['efficiency_global']:.3f}",
        "POI coverage":   int(row["poi_coverage"]),
    })

detail = pd.DataFrame(detail_rows)
print("\nMetrics at OSM biketrack reference point (or syn_max if not reached):")
print(detail.to_string(index=False))
detail.to_csv(OUT / "metrics_at_biketrack.csv", index=False)

print(f"\nAll outputs in: {OUT}")
print("Done.")
