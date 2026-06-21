"""
SAFETY spatial analysis for all 5 cities: Amsterdam, Barcelona, Berlin, Oslo, Vienna.

This is the A-vs-C counterpart of 02_analysis_multicity_spatial.py (which does B vs C).

REFERENCE NETWORK: OSM biketrack (A) — the dedicated/separated cycle infrastructure
                   (now including snapped cycleway=crossing connectors). This is the
                   *safety* reference: where can you ride on protected paths, and how
                   much of the synthetic optimum C is already covered by protected infra
                   vs. still routed over mixed traffic.

All outputs carry a `_safety` suffix so they never collide with the B-vs-C run:
  analysis_output/{city}/
    osm_biketrack_{city}.geojson
    synthetic_{city}_safety.geojson
    overlap_{city}_safety.geojson    (synthetic edges that run along biketrack, >=50% buffer overlap)
    gap_{city}_safety.geojson        (synthetic edges NOT along biketrack -> unprotected proposals)
    map_{city}_safety.png            (visualisation with 15 m buffer)

  analysis_output/comparison/
    spatial_summary_safety.csv       headline numbers @ 15 m buffer
    buffer_sensitivity_safety.csv    overlap/gap % for 10/15/20 m

Usage: conda run -n growbikenet python pipeline/02b_analysis_safety_spatial.py
"""

from pathlib import Path
import json
import time
import zipfile

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkt as shapely_wkt
from shapely.ops import unary_union
from shapely.strtree import STRtree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ── Config ──────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent / "bikenwgrowth-data"
CRS_GEO = "EPSG:4326"

CITIES = ["amsterdam", "barcelona", "berlin", "oslo", "vienna"]

CITY_LABELS = {
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

REFERENCE_NETWORK = "biketrack"         # A — protected/separated infrastructure
SUFFIX            = "_safety"           # keeps every output distinct from the B run
BUFFER_DEFAULT_M  = 15.0                # headline buffer
BUFFER_SWEEP      = [10.0, 15.0, 20.0]  # for sensitivity analysis

A_NET_COLOR = "#2c7fb8"                 # protected-infra blue (distinct from B's grey)
BG_COLOR    = "#7b838d"                 # all-streets (carall) basemap (clearly visible)

OUT_COMP = BASE / "analysis_output" / "comparison"
OUT_COMP.mkdir(parents=True, exist_ok=True)


# ── Helpers ─────────────────────────────────────────────────────────────────
def load_osm_edges(city, network_type=REFERENCE_NETWORK):
    """Load OSM edges of a given network type from the CSV inside the zip.

    Collapses directed duplicates to undirected (one row per physical segment) so the
    reported length matches existing.csv and overlap geometry is not drawn twice."""
    zpath = BASE / "data" / city / f"{city}_{network_type}_edges.zip"
    with zipfile.ZipFile(zpath) as z:
        csv_name = [n for n in z.namelist() if n.endswith(".csv")][0]
        with z.open(csv_name) as f:
            df = pd.read_csv(f, low_memory=False)
    df = df[df["geometry"].notna()].copy()

    if {"u", "v", "key"}.issubset(df.columns):
        u = df["u"].to_numpy()
        v = df["v"].to_numpy()
        df["_uv_lo"] = np.minimum(u, v)
        df["_uv_hi"] = np.maximum(u, v)
        df = df.drop_duplicates(subset=["_uv_lo", "_uv_hi", "key"])

    df["geometry"] = df["geometry"].apply(
        lambda x: shapely_wkt.loads(x) if isinstance(x, str) else None
    )
    df = df[df["geometry"].notna()]
    gdf = gpd.GeoDataFrame(df[["length", "geometry"]], geometry="geometry", crs=CRS_GEO)
    return gdf


def load_syn_stage(city, target_km=None):
    """Load synthetic stage closest to target_km; if None, load final stage."""
    idx_path = BASE / "analysis_output" / city / "synthetic_stages" / "index.json"
    with open(idx_path) as f:
        idx = json.load(f)
    stage_df = pd.DataFrame(idx)

    if target_km is None:
        row = stage_df.iloc[-1]
    else:
        i = (stage_df["length_km"] - target_km).abs().idxmin()
        row = stage_df.iloc[i]

    stages_dir = BASE / "analysis_output" / city / "synthetic_stages"
    gdf = gpd.read_file(stages_dir / row["file"])
    print(f"  Stage: Q={row['quantile']:.3f}  {row['length_km']:.0f} km  "
          f"({row['n_edges']} edges)")
    return gdf, row


def overlap_ratios(real_gdf_p, syn_gdf_p, buf):
    """For each synthetic edge, fraction of its buffered area that overlaps the
    reference network's buffer at `buf` metres (STRtree-accelerated)."""
    osm_geoms   = list(real_gdf_p.geometry.values)
    real_bufs   = [g.buffer(buf) for g in osm_geoms]
    tree        = STRtree(real_bufs)
    syn_geoms   = list(syn_gdf_p.geometry.values)

    ratios = np.zeros(len(syn_geoms))
    for i, sg in enumerate(syn_geoms):
        b = sg.buffer(buf)
        if b.area <= 0:
            continue
        cand_idx = tree.query(b)
        if len(cand_idx) == 0:
            continue
        local = [real_bufs[k] for k in cand_idx if real_bufs[k].intersects(b)]
        if not local:
            continue
        local_union = local[0] if len(local) == 1 else unary_union(local)
        inter = b.intersection(local_union)
        ratios[i] = inter.area / b.area
    return ratios


def split_overlap_gap(syn_gdf_p, ratios):
    """>=0.5 -> overlap, <0.5 -> gap. Returns (overlap_gdf, gap_gdf, summary)."""
    syn = syn_gdf_p.copy()
    syn["overlap_ratio"] = ratios
    ov  = syn[syn["overlap_ratio"] >= 0.5].copy()
    gap = syn[syn["overlap_ratio"] <  0.5].copy()
    tot_m = syn.geometry.length.sum()
    ov_m  = ov.geometry.length.sum()
    gap_m = gap.geometry.length.sum()
    return ov, gap, dict(
        tot_km=tot_m/1000, ov_km=ov_m/1000, gap_km=gap_m/1000,
        ov_ratio=ov_m/tot_m if tot_m else 0.0,
    )


# ── Main loop ────────────────────────────────────────────────────────────────
spatial_rows = []
sensitivity_rows = []

for city in CITIES:
    label = CITY_LABELS[city]
    print(f"\n{'='*60}\n  {label}  (SAFETY / A vs C)\n{'='*60}")

    OUT_CITY = BASE / "analysis_output" / city
    OUT_CITY.mkdir(parents=True, exist_ok=True)

    # ── Load OSM reference (biketrack / A) ─────────────────────────────────
    print(f"Loading OSM {REFERENCE_NETWORK} (A) edges...")
    osm_gdf = load_osm_edges(city, REFERENCE_NETWORK)
    ref_km = osm_gdf["length"].sum() / 1000
    print(f"  OSM {REFERENCE_NETWORK}: {len(osm_gdf)} edges, {ref_km:.0f} km")

    # All-streets basemap (carall) for map context only — not used in any metric.
    carall_gdf = load_osm_edges(city, "carall")

    # ── Load synthetic network at its FULL maximum (Q = 1.0) ───────────────
    # Unlike the B run (where the matching branch never triggers because bikeable
    # always exceeds syn max), A can be shorter than C. We deliberately always use
    # the full synthetic optimum so that overlap/gap reads as "share of the FULL
    # optimal network already on / missing from protected infrastructure" — a
    # single consistent, cross-city-comparable safety figure.
    syn_csv = pd.read_csv(BASE / "results" / city / f"{city}_poi_grid_betweenness.csv")
    syn_max_km = syn_csv["length"].max() / 1000
    print(f"  A ({ref_km:.0f} km)  vs  C max ({syn_max_km:.0f} km) -> using full C max")
    syn_gdf, srow = load_syn_stage(city, target_km=None)
    note = f"C max ({srow['length_km']:.0f} km, Q=1.000)"

    # ── Project to metric CRS once ─────────────────────────────────────────
    crs_proj = syn_gdf.estimate_utm_crs()
    print(f"  Projected CRS: {crs_proj.srs}")
    osm_p = osm_gdf.to_crs(crs_proj)
    syn_p = syn_gdf.to_crs(crs_proj)
    car_p = carall_gdf.to_crs(crs_proj)

    # ── Buffer sweep (sensitivity) ─────────────────────────────────────────
    print("Buffer sensitivity sweep:")
    cached = {}
    for buf in BUFFER_SWEEP:
        t0 = time.time()
        ratios = overlap_ratios(osm_p, syn_p, buf)
        ov, gap, s = split_overlap_gap(syn_p, ratios)
        cached[buf] = dict(ratios=ratios, ov=ov, gap=gap, **s)
        print(f"  buf={buf:>4.0f}m   "
              f"overlap={s['ov_km']:5.0f} km ({s['ov_ratio']*100:5.1f}%)   "
              f"gap={s['gap_km']:5.0f} km ({(1-s['ov_ratio'])*100:5.1f}%)   "
              f"[{time.time()-t0:5.1f}s]")
        sensitivity_rows.append({
            "city":         label,
            "buffer_m":     buf,
            "synthetic_km": round(s["tot_km"], 1),
            "overlap_km":   round(s["ov_km"], 1),
            "overlap_pct":  round(s["ov_ratio"]*100, 1),
            "gap_km":       round(s["gap_km"], 1),
            "gap_pct":      round((1 - s["ov_ratio"])*100, 1),
        })

    # ── Headline outputs at default buffer (15 m) ──────────────────────────
    sp = cached[BUFFER_DEFAULT_M]

    print(f"Exporting GeoJSONs (buffer={BUFFER_DEFAULT_M:.0f}m)...")
    exports = [
        (osm_gdf,                       f"osm_{REFERENCE_NETWORK}_{city}.geojson"),
        (syn_p.to_crs(CRS_GEO),          f"synthetic_{city}{SUFFIX}.geojson"),
        (sp["ov"].to_crs(CRS_GEO),       f"overlap_{city}{SUFFIX}.geojson"),
        (sp["gap"].to_crs(CRS_GEO),      f"gap_{city}{SUFFIX}.geojson"),
    ]
    for gdf_exp, fname in exports:
        out_path = OUT_CITY / fname
        gdf_exp.to_file(out_path, driver="GeoJSON")
        print(f"  {fname:<45} {gdf_exp.geometry.length.sum()/1000:7.1f} km")

    # ── Per-city map (15 m buffer), clipped to effective boundary ──────────
    print("Plotting per-city safety map...")

    def _lines_only(gdf):
        if gdf is None or len(gdf) == 0:
            return gdf
        return gdf[gdf.geometry.type.isin(["LineString", "MultiLineString"])]

    boundary_path = OUT_CITY / "effective_boundary.geojson"
    if boundary_path.exists():
        bdy = gpd.read_file(boundary_path).to_crs(crs_proj)
        car_for_plot  = _lines_only(gpd.clip(car_p, bdy))
        osm_for_plot  = _lines_only(gpd.clip(osm_p, bdy))
        ov_for_plot   = _lines_only(gpd.clip(sp["ov"], bdy))
        gap_for_plot  = _lines_only(gpd.clip(sp["gap"], bdy))
        clip_note = "  ·  clipped to effective boundary (nominatim − forest)"
    else:
        car_for_plot, osm_for_plot, ov_for_plot, gap_for_plot = car_p, osm_p, sp["ov"], sp["gap"]
        clip_note = ""

    fig, ax = plt.subplots(figsize=(10, 10))
    # All-streets basemap first (lowest layer), for geographic context.
    car_for_plot.plot(ax=ax, color=BG_COLOR, lw=0.45, alpha=0.55, zorder=0)
    if boundary_path.exists():
        bdy.boundary.plot(ax=ax, color="#888", lw=0.6, alpha=0.6, zorder=1)
    osm_for_plot.plot(ax=ax, color=A_NET_COLOR, lw=0.8, alpha=0.75,
               label=f"A — biketrack (protected)  {ref_km:.0f} km", zorder=2)
    gap_for_plot.plot(ax=ax, color="#d73027", lw=0.9, alpha=0.85,
               label=f"Gap — C not on protected infra  {sp['gap_km']:.0f} km  ({(1-sp['ov_ratio'])*100:.0f}%)",
               zorder=3)
    ov_for_plot.plot(ax=ax, color="#4dac26", lw=0.7, alpha=0.8,
              label=f"Overlap — C on protected infra  {sp['ov_km']:.0f} km  ({sp['ov_ratio']*100:.0f}%)",
              zorder=3)
    ax.set_title(
        f"{label} — C (synthetic) vs. A (biketrack / protected)\n"
        f"{note}   ·   buffer = {BUFFER_DEFAULT_M:.0f} m{clip_note}",
        fontsize=11, fontweight="bold"
    )
    handles, _lab = ax.get_legend_handles_labels()
    handles = [Line2D([0], [0], color=BG_COLOR, lw=1.4, alpha=0.8,
                      label="All streets (OSM)")] + handles
    ax.legend(handles=handles, fontsize=9, loc="upper right")
    ax.set_axis_off()
    plt.tight_layout()
    map_path = OUT_CITY / f"map_{city}{SUFFIX}.png"
    fig.savefig(map_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {map_path.name}")

    spatial_rows.append({
        "City":                          label,
        "Note":                          note,
        f"OSM {REFERENCE_NETWORK} (km)": f"{ref_km:.0f}",
        "Synthetic (km)":                f"{sp['tot_km']:.0f}",
        "Overlap (km)":                  f"{sp['ov_km']:.0f}",
        "Overlap (%)":                   f"{sp['ov_ratio']*100:.1f}%",
        "Gap (km)":                      f"{sp['gap_km']:.0f}",
        "Gap (%)":                       f"{(1-sp['ov_ratio'])*100:.1f}%",
        "Buffer (m)":                    int(BUFFER_DEFAULT_M),
    })


# ── Combined overview map ──────────────────────────────────────────────────
print("\nPlotting combined safety overview...")
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle(
    f"Safety spatial analysis: synthetic network vs. OSM {REFERENCE_NETWORK} (protected) "
    f"(all cities, buffer = {BUFFER_DEFAULT_M:.0f} m, clipped to effective boundary)",
    fontsize=13, fontweight="bold"
)
axes_flat = axes.flatten()

for ax_idx, city in enumerate(CITIES):
    ax = axes_flat[ax_idx]
    label = CITY_LABELS[city]
    city_dir = BASE / "analysis_output" / city

    osm_gdf_r = gpd.read_file(city_dir / f"osm_{REFERENCE_NETWORK}_{city}.geojson")
    ov_gdf_r  = gpd.read_file(city_dir / f"overlap_{city}{SUFFIX}.geojson")
    gap_gdf_r = gpd.read_file(city_dir / f"gap_{city}{SUFFIX}.geojson")
    crs_p     = osm_gdf_r.estimate_utm_crs()

    osm_p_r = osm_gdf_r.to_crs(crs_p)
    ov_p_r  = ov_gdf_r.to_crs(crs_p)
    gap_p_r = gap_gdf_r.to_crs(crs_p)

    bdy_path = city_dir / "effective_boundary.geojson"
    if bdy_path.exists():
        bdy_p = gpd.read_file(bdy_path).to_crs(crs_p)
        def _lo(g): return g[g.geometry.type.isin(["LineString", "MultiLineString"])]
        osm_p_r = _lo(gpd.clip(osm_p_r, bdy_p))
        ov_p_r  = _lo(gpd.clip(ov_p_r,  bdy_p))
        gap_p_r = _lo(gpd.clip(gap_p_r, bdy_p))
        bdy_p.boundary.plot(ax=ax, color="#888", lw=0.4, alpha=0.5, zorder=1)

    osm_p_r.plot(ax=ax, color=A_NET_COLOR, lw=0.5, alpha=0.5,
                 label=f"OSM {REFERENCE_NETWORK}", zorder=2)
    gap_p_r.plot(ax=ax, color="#d73027", lw=0.8, alpha=0.8,
                 label="Gap (synth, unprotected)", zorder=3)
    ov_p_r.plot(ax=ax, color="#4dac26", lw=0.6, alpha=0.65,
                label="Overlap (synth on protected)", zorder=3)

    ax.set_title(label, fontsize=10, fontweight="bold", color=COLORS[city])
    ax.legend(fontsize=7, loc="upper right")
    ax.set_axis_off()

axes_flat[5].set_visible(False)
plt.tight_layout()
overview_path = OUT_COMP / "spatial_overview_safety.png"
fig.savefig(overview_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {overview_path.name}")


# ── Headline summary table ──────────────────────────────────────────────────
sp_summary = pd.DataFrame(spatial_rows)
sp_summary.to_csv(OUT_COMP / "spatial_summary_safety.csv", index=False)
print(f"\nHeadline (buffer = {BUFFER_DEFAULT_M:.0f} m):")
print(sp_summary.to_string(index=False))
print(f"\nSaved: spatial_summary_safety.csv")


# ── Sensitivity table ───────────────────────────────────────────────────────
sens = pd.DataFrame(sensitivity_rows)
sens.to_csv(OUT_COMP / "buffer_sensitivity_safety.csv", index=False)
print(f"\nBuffer sensitivity:")
print(sens.to_string(index=False))
print(f"\nSaved: buffer_sensitivity_safety.csv")

print("\nDone.")
