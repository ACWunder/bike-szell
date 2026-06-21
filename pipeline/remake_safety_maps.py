"""
Redraw the per-city SAFETY maps (map_{city}_safety.png) adding a faint grey
all-streets (carall) basemap behind the protected network A / overlap / gap.

Reuses the already-computed overlap/gap GeoJSONs from 02b_analysis_safety_spatial.py
(so the overlap numbers are untouched — this only re-renders the figures, fast).
Styling matches 02b's per-city map exactly, plus the carall background layer.

Usage: conda run -n growbikenet python pipeline/remake_safety_maps.py
"""
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkt as shapely_wkt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BASE = Path(__file__).resolve().parent.parent / "bikenwgrowth-data"
CRS_GEO = "EPSG:4326"
CITIES = ["amsterdam", "barcelona", "berlin", "oslo", "vienna"]
CITY_LABELS = {c: c.capitalize() for c in CITIES}

BUFFER_DEFAULT_M = 15.0
A_NET_COLOR = "#2c7fb8"      # protected-infra blue (matches 02b + report)
BG_COLOR    = "#7b838d"      # all-streets grey basemap (clearly visible)
BG_LW       = 0.45
BG_ALPHA    = 0.55
SUFFIX = "_safety"

spatial = pd.read_csv(BASE / "analysis_output" / "comparison" / "spatial_summary_safety.csv")


def load_osm_edges(city, network_type):
    """Undirected OSM edges for a network from its zip (one row per segment)."""
    zpath = BASE / "data" / city / f"{city}_{network_type}_edges.zip"
    with zipfile.ZipFile(zpath) as z:
        csv_name = [n for n in z.namelist() if n.endswith(".csv")][0]
        with z.open(csv_name) as f:
            df = pd.read_csv(f, low_memory=False)
    df = df[df["geometry"].notna()].copy()
    if {"u", "v", "key"}.issubset(df.columns):
        u = df["u"].to_numpy(); v = df["v"].to_numpy()
        df["_lo"] = np.minimum(u, v); df["_hi"] = np.maximum(u, v)
        df = df.drop_duplicates(subset=["_lo", "_hi", "key"])
    df["geometry"] = df["geometry"].apply(
        lambda x: shapely_wkt.loads(x) if isinstance(x, str) else None)
    df = df[df["geometry"].notna()]
    return gpd.GeoDataFrame(df[["geometry"]], geometry="geometry", crs=CRS_GEO)


def _lines_only(gdf):
    if gdf is None or len(gdf) == 0:
        return gdf
    return gdf[gdf.geometry.type.isin(["LineString", "MultiLineString"])]


for city in CITIES:
    label = CITY_LABELS[city]
    cdir = BASE / "analysis_output" / city
    print(f"  {label} ...", flush=True)

    osm_a = gpd.read_file(cdir / f"osm_biketrack_{city}.geojson")
    ov    = gpd.read_file(cdir / f"overlap_{city}{SUFFIX}.geojson")
    gap   = gpd.read_file(cdir / f"gap_{city}{SUFFIX}.geojson")
    carall = load_osm_edges(city, "carall")

    crs_proj = osm_a.estimate_utm_crs()
    osm_p = osm_a.to_crs(crs_proj)
    ov_p  = ov.to_crs(crs_proj)
    gap_p = gap.to_crs(crs_proj)
    car_p = carall.to_crs(crs_proj)

    row = spatial[spatial["City"] == label].iloc[0]
    ref_km = float(row["OSM biketrack (km)"])
    ov_km  = float(str(row["Overlap (km)"]))
    gap_km = float(str(row["Gap (km)"]))
    ov_pct = float(str(row["Overlap (%)"]).rstrip("%"))
    gap_pct = float(str(row["Gap (%)"]).rstrip("%"))
    note = str(row["Note"])

    bdy_path = cdir / "effective_boundary.geojson"
    if bdy_path.exists():
        bdy = gpd.read_file(bdy_path).to_crs(crs_proj)
        car_plot = _lines_only(gpd.clip(car_p, bdy))
        osm_plot = _lines_only(gpd.clip(osm_p, bdy))
        ov_plot  = _lines_only(gpd.clip(ov_p, bdy))
        gap_plot = _lines_only(gpd.clip(gap_p, bdy))
        clip_note = "  ·  clipped to effective boundary (nominatim − forest)"
    else:
        bdy = None
        car_plot, osm_plot, ov_plot, gap_plot = car_p, osm_p, ov_p, gap_p
        clip_note = ""

    fig, ax = plt.subplots(figsize=(10, 10))
    # all-streets basemap (drawn first, lowest)
    car_plot.plot(ax=ax, color=BG_COLOR, lw=BG_LW, alpha=BG_ALPHA, zorder=0)
    if bdy is not None:
        bdy.boundary.plot(ax=ax, color="#888", lw=0.6, alpha=0.6, zorder=1)
    osm_plot.plot(ax=ax, color=A_NET_COLOR, lw=0.8, alpha=0.75,
                  label=f"A — biketrack (protected)  {ref_km:.0f} km", zorder=2)
    gap_plot.plot(ax=ax, color="#d73027", lw=0.9, alpha=0.85,
                  label=f"Gap — C not on protected infra  {gap_km:.0f} km  ({gap_pct:.0f}%)",
                  zorder=3)
    ov_plot.plot(ax=ax, color="#4dac26", lw=0.7, alpha=0.8,
                 label=f"Overlap — C on protected infra  {ov_km:.0f} km  ({ov_pct:.0f}%)",
                 zorder=3)
    ax.set_title(
        f"{label} — C (synthetic) vs. A (biketrack / protected)\n"
        f"{note}   ·   buffer = {BUFFER_DEFAULT_M:.0f} m{clip_note}",
        fontsize=11, fontweight="bold")
    # legend with an explicit grey "all streets" entry
    handles, labels_ = ax.get_legend_handles_labels()
    handles = [Line2D([0], [0], color=BG_COLOR, lw=1.4, alpha=0.8,
                      label="All streets (OSM)")] + handles
    ax.legend(handles=handles, fontsize=9, loc="upper right")
    ax.set_axis_off()
    plt.tight_layout()
    out = cdir / f"map_{city}{SUFFIX}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved {out.name}", flush=True)

print("Done.")
