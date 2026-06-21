"""
Render the Vienna grid-sensitivity NETWORK comparison: the final synthetic network
(Q = 1.0) grown on a fine 600 m POI grid vs. the standard 1701 m grid, side by side.

Both are extracted the same way from the GrowBikeNet betweenness pickles (straight
segments between graph nodes, lon = x, lat = -y), clipped to Vienna's effective boundary,
so the only visible difference is the grid spacing. Output:
  analysis_output/comparison/vienna_gridl_networks.png

Run from bikenwgrowth-source/scripts (growbikenet env):
  conda run -n growbikenet python ../../pipeline/render_gridl_networks.py
or with the repo paths resolved below from anywhere in the env.
"""
import os
import pickle
from pathlib import Path

import numpy as np
import geopandas as gpd
from shapely.geometry import LineString
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "bikenwgrowth-data"
OUT = DATA / "analysis_output" / "comparison" / "vienna_gridl_networks.png"

RUNS = [
    ("vienna_2", "gridl = 600 m", "fine POI grid", "#6a51a3"),   # finer grid
    ("vienna",   "gridl = 1701 m", "standard (Szell et al.)", "#2c7fb8"),
]

BOUNDARY = DATA / "analysis_output" / "vienna" / "effective_boundary.geojson"


def final_network_gdf(resultid):
    """Final synthetic network (Q=1.0) from the betweenness pickle as a GeoDataFrame
    of straight node-to-node segments in EPSG:4326 (lat = -y), plus total km."""
    p = DATA / "results" / resultid / "vienna_poi_grid_betweenness.pickle"
    with open(p, "rb") as f:
        obj = pickle.load(f)
    g = obj["GTs"][-1]
    xs = np.array(g.vs["x"])
    ys = -np.array(g.vs["y"])            # stored y is negated latitude
    lines = []
    for e in g.es:
        u, v = e.tuple
        lines.append(LineString([(xs[u], ys[u]), (xs[v], ys[v])]))
    total_km = sum(g.es["weight"]) / 1000.0
    gdf = gpd.GeoDataFrame(geometry=lines, crs="EPSG:4326")
    return gdf, total_km, g.ecount()


bdy = gpd.read_file(BOUNDARY).to_crs("EPSG:4326")

# Stacked vertically (one network above the other) for a portrait report page.
fig, axes = plt.subplots(2, 1, figsize=(8.6, 10.8))
for ax, (rid, gl, sub, color) in zip(axes, RUNS):
    gdf, km, ne = final_network_gdf(rid)
    clipped = gpd.clip(gdf, bdy)
    bdy.boundary.plot(ax=ax, color="#888", lw=0.7, alpha=0.6, zorder=1)
    clipped.plot(ax=ax, color=color, lw=0.4, alpha=0.85, zorder=2)
    ax.set_title(f"C  ·  {gl}   ·   {km:.0f} km   ·   {sub}",
                 fontsize=13, fontweight="bold", color=color)
    ax.set_axis_off()
    ax.set_aspect("equal")
    print(f"  {rid}: {gl}  {km:.0f} km  {ne} edges", flush=True)

fig.suptitle("Vienna synthetic optimum (C) at full growth — fine vs. standard POI grid",
             fontsize=14, fontweight="bold", y=0.995)
plt.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=170, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT}")
