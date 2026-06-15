"""
Identify hotspots in the algorithm-proposed "gap" (missing) edges per city.

For every city we read gap_{city}.geojson (synthetic edges that do NOT lie
within 15 m of an OSM bikeable edge), reduce each edge to its midpoint, and
run DBSCAN to find spatial clusters. Each cluster is a place where the
algorithm proposes a *bunch* of new connections — i.e. a structural
priority area, not just an isolated edge.

Per city we keep the top-5 clusters by total km. For each cluster we
record:
  * total length (km) of all gap edges in the cluster
  * number of edges
  * centroid (UTM coords + lon/lat)
  * compass direction relative to the city centre (effective-boundary
    centroid), with a "central" label if the cluster is close to it

Outputs:
  analysis_output/comparison/gap_clusters.csv
  analysis_output/{city}/gap_clusters_{city}.geojson   (clusters as polygons)
  analysis_output/{city}/gap_hotspots_map_{city}.png

Usage: conda run -n growbikenet python analysis_gap_clusters.py
"""

from pathlib import Path
import math

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from sklearn.cluster import DBSCAN

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# ── Config ────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent / "bikenwgrowth-data"
OUT_COMP = BASE / "analysis_output" / "comparison"
OUT_COMP.mkdir(parents=True, exist_ok=True)

CITIES = ["amsterdam", "barcelona", "berlin", "oslo", "vienna"]
CITY_LABELS = {
    "amsterdam": "Amsterdam", "barcelona": "Barcelona", "berlin": "Berlin",
    "oslo": "Oslo", "vienna": "Vienna",
}
COLORS = {
    "amsterdam": "#e41a1c", "barcelona": "#377eb8", "berlin": "#4daf4a",
    "oslo": "#984ea3", "vienna": "#ff7f00",
}

# DBSCAN per-city epsilon (metres). Oslo has so many gap edges in a single
# contiguous area that any eps ≥ 250 m chains them into one mega-cluster;
# 150 m forces the algorithm to split that blob into meaningful sub-areas.
# The other cities keep a comfortable 300 m which leaves them with 5
# well-separated hotspots.
DBSCAN_EPS_BY_CITY = {
    "amsterdam": 300,
    "barcelona": 300,
    "berlin":    300,
    "oslo":      150,
    "vienna":    300,
}
DBSCAN_MIN_PTS   = 8

# Take the N largest clusters per city (by total km)
TOP_N_CLUSTERS   = 5

# "Central" if cluster centroid is closer than this fraction of city radius
CENTRAL_RADIUS_FRAC = 0.20


def compass_direction(dx, dy, radius_frac):
    """
    Convert a relative (dx, dy) vector — east = +x, north = +y — to a
    one-line compass label. If the cluster sits very close to the city
    centre (within radius_frac of the typical city radius), call it
    'central' instead.
    """
    if radius_frac < CENTRAL_RADIUS_FRAC:
        return "central"
    # angle: 0° = east, 90° = north
    angle = math.degrees(math.atan2(dy, dx))
    # Map to 8 compass directions
    # Each octant covers 45°; centred on each direction
    bins = [
        (-22.5,  22.5,  "east"),
        ( 22.5,  67.5,  "north-east"),
        ( 67.5, 112.5,  "north"),
        (112.5, 157.5,  "north-west"),
        (157.5, 180.1,  "west"),
        (-180,  -157.5, "west"),
        (-157.5, -112.5, "south-west"),
        (-112.5, -67.5,  "south"),
        (-67.5,  -22.5,  "south-east"),
    ]
    for lo, hi, name in bins:
        if lo <= angle < hi:
            return name
    return "central"


# ── Main loop ─────────────────────────────────────────────────────────────
all_rows = []

for city in CITIES:
    print(f"\n{'='*60}\n  {city}\n{'='*60}")
    city_dir = BASE / "analysis_output" / city

    gap_path = city_dir / f"gap_{city}.geojson"
    if not gap_path.exists():
        print(f"  ! {gap_path.name} missing — skipping")
        continue

    gap = gpd.read_file(gap_path)
    if len(gap) == 0:
        print("  ! gap is empty — skipping")
        continue

    crs_p = gap.estimate_utm_crs()
    gap_p = gap.to_crs(crs_p)

    # Boundary for the city centre + radius reference
    bdy_path = city_dir / "effective_boundary.geojson"
    if bdy_path.exists():
        bdy = gpd.read_file(bdy_path).to_crs(crs_p)
        bdy_geom = bdy.geometry.iloc[0]
        city_centre = bdy_geom.centroid
        # Use sqrt(area / π) as a typical city radius
        city_radius = math.sqrt(bdy_geom.area / math.pi)
    else:
        # Fall back to the gap edges' overall centroid
        merged = unary_union(list(gap_p.geometry))
        city_centre = merged.centroid
        # 95th percentile distance from centre as a crude radius
        d_to_centre = np.array([
            ((g.centroid.x - city_centre.x) ** 2 +
             (g.centroid.y - city_centre.y) ** 2) ** 0.5
            for g in gap_p.geometry
        ])
        city_radius = float(np.percentile(d_to_centre, 95))

    # 1 · Midpoints of every gap edge
    midpoints = gap_p.geometry.centroid
    coords    = np.array([(p.x, p.y) for p in midpoints])
    lengths_m = gap_p.geometry.length.values

    # 2 · DBSCAN — per-city eps to handle very different edge densities
    eps_m = DBSCAN_EPS_BY_CITY.get(city, 300)
    print(f"  {len(coords)} gap edges, DBSCAN(eps={eps_m} m, "
          f"min={DBSCAN_MIN_PTS})")
    db = DBSCAN(eps=eps_m, min_samples=DBSCAN_MIN_PTS).fit(coords)
    labels = db.labels_

    # 3 · Per-cluster stats
    cluster_rows = []
    for cid in sorted(set(labels)):
        if cid == -1:
            continue  # noise
        mask = labels == cid
        cl_coords  = coords[mask]
        cl_lengths = lengths_m[mask]
        cx = float(cl_coords[:, 0].mean())
        cy = float(cl_coords[:, 1].mean())
        dx = cx - city_centre.x
        dy = cy - city_centre.y
        dist = math.sqrt(dx ** 2 + dy ** 2)
        radius_frac = dist / max(city_radius, 1)
        direction = compass_direction(dx, dy, radius_frac)
        cluster_rows.append({
            "city":          CITY_LABELS[city],
            "cluster_id":    int(cid),
            "n_edges":       int(mask.sum()),
            "total_km":      float(cl_lengths.sum() / 1000),
            "centroid_x_m":  cx,
            "centroid_y_m":  cy,
            "dist_from_centre_km": dist / 1000,
            "radius_frac":   radius_frac,
            "direction":     direction,
        })

    if not cluster_rows:
        print("  ! no clusters found")
        continue

    # Sort by total km, take top N
    cluster_df = pd.DataFrame(cluster_rows)
    cluster_df = cluster_df.sort_values("total_km", ascending=False)
    top = cluster_df.head(TOP_N_CLUSTERS).copy()
    top["rank"] = range(1, len(top) + 1)
    all_rows.extend(top.to_dict("records"))

    total_gap_km     = lengths_m.sum() / 1000
    clustered_km     = cluster_df["total_km"].sum()
    top_km           = top["total_km"].sum()
    print(f"  → {len(cluster_df)} clusters total ({clustered_km:.0f} km), "
          f"top {len(top)} = {top_km:.0f} km "
          f"({100*top_km/total_gap_km:.0f} % of all gap)")
    for _, r in top.iterrows():
        print(f"    #{r['rank']}: {r['total_km']:5.1f} km · {r['n_edges']:3d} edges · "
              f"{r['direction']} ({r['dist_from_centre_km']:.1f} km from centre)")

    # 4 · Export cluster polygons (convex hull) as GeoJSON for QGIS
    poly_rows = []
    for _, r in top.iterrows():
        mask = labels == r["cluster_id"]
        pts = coords[mask]
        # Convex hull as cluster footprint
        if len(pts) >= 3:
            hull = gpd.GeoSeries(
                [Point(x, y) for x, y in pts], crs=crs_p
            ).unary_union.convex_hull
        else:
            hull = Point(pts[0]).buffer(eps_m)
        poly_rows.append({
            "rank":      int(r["rank"]),
            "total_km":  r["total_km"],
            "n_edges":   int(r["n_edges"]),
            "direction": r["direction"],
            "geometry":  hull,
        })
    poly_gdf = gpd.GeoDataFrame(poly_rows, crs=crs_p).to_crs("EPSG:4326")
    poly_gdf.to_file(city_dir / f"gap_clusters_{city}.geojson",
                      driver="GeoJSON")

    # 5 · Render a per-city hotspot map
    fig, ax = plt.subplots(figsize=(10, 10))
    # Boundary outline if available
    if bdy_path.exists():
        bdy.boundary.plot(ax=ax, color="#888", lw=0.6, alpha=0.5, zorder=1)
    # All gap edges in light red
    gap_p.plot(ax=ax, color="#d73027", lw=0.5, alpha=0.35, zorder=2)
    # Top clusters as filled hulls
    for _, r in top.iterrows():
        hull = poly_gdf[poly_gdf["rank"] == r["rank"]].to_crs(crs_p)
        hull.plot(ax=ax, facecolor=COLORS[city], alpha=0.30,
                  edgecolor=COLORS[city], lw=1.5, zorder=3)
        # Number label at cluster centroid
        cx, cy = r["centroid_x_m"], r["centroid_y_m"]
        ax.annotate(
            f"#{int(r['rank'])}",
            (cx, cy), color="white", fontsize=14, fontweight="bold",
            ha="center", va="center",
            bbox=dict(boxstyle="circle,pad=0.35",
                      facecolor=COLORS[city], edgecolor="white", linewidth=1.5),
            zorder=5,
        )
    ax.set_title(
        f"{CITY_LABELS[city]} — gap hotspots\n"
        f"top {len(top)} DBSCAN clusters · "
        f"{top_km:.0f} km of {total_gap_km:.0f} km gap "
        f"({100*top_km/total_gap_km:.0f} % of all gap edges)",
        fontsize=11, fontweight="bold",
    )
    ax.set_axis_off()
    plt.tight_layout()
    out_map = city_dir / f"gap_hotspots_map_{city}.png"
    fig.savefig(out_map, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_map.name}")


# ── Combined cross-city table ─────────────────────────────────────────────
df = pd.DataFrame(all_rows)
df = df[[
    "city", "rank", "total_km", "n_edges",
    "dist_from_centre_km", "direction",
    "centroid_x_m", "centroid_y_m",
]]
out_csv = OUT_COMP / "gap_clusters.csv"
df.to_csv(out_csv, index=False)
print(f"\n{'='*60}")
print(df.to_string(index=False))
print(f"\nSaved: {out_csv}")
print("Done.")
