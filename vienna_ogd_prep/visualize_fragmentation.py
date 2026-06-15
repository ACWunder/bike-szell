"""
Fragmentation map of the Vienna bicycle network.
Each connected component gets its own colour.
LCC = dark blue, top components = coloured, rest = light grey.
Env: growbikenet
"""
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from shapely import wkt as shapely_wkt
from shapely.geometry import LineString, Point

import random
random.seed(42); np.random.seed(42)

BASE     = Path(__file__).resolve().parent.parent / "bikenwgrowth-data"
REAL_CSV = Path(__file__).resolve().parent.parent / "data" / "vienna_ogd" / "Radwege_ogd.csv"
OUT      = BASE / "analysis_output" / "vienna_ogd"
OUT.mkdir(parents=True, exist_ok=True)
CRS_IN, CRS_PROJ = "EPSG:4326", "EPSG:31256"
SNAP  = 2.0
N_TOP = 8  # top components with individual colours

PALETTE = ["#1a6faf", "#e6194b", "#f58231", "#3cb44b",
           "#911eb4", "#42d4f4", "#f032e6", "#bfef45"]


def safe_wkt(x):
    try:
        return shapely_wkt.loads(x) if isinstance(x, str) and x.strip() else None
    except Exception:
        return None


def snap(c):
    return (round(c[0] / SNAP) * SNAP, round(c[1] / SNAP) * SNAP)


def build_graph_with_segments(gdf):
    G = nx.Graph()
    segments = []
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        lines = geom.geoms if geom.geom_type == "MultiLineString" else [geom]
        for line in lines:
            coords = list(line.coords)
            for i in range(len(coords) - 1):
                u = snap(coords[i])
                v = snap(coords[i + 1])
                if u == v:
                    continue
                seg_len = Point(coords[i]).distance(Point(coords[i + 1]))
                if not G.has_edge(u, v):
                    G.add_edge(u, v, length=seg_len)
                segments.append((u, v, coords[i], coords[i + 1]))
    return G, segments


def plot_fragmentation(gdf, label, ax):
    print(f"  Building graph for {label}...")
    G, segments = build_graph_with_segments(gdf)

    # Sort components by total edge length (descending)
    comps = sorted(
        nx.connected_components(G),
        key=lambda c: sum(G[u][v]["length"] for u, v in G.subgraph(c).edges()),
        reverse=True,
    )

    node_rank = {}
    for rank, comp in enumerate(comps):
        for node in comp:
            node_rank[node] = rank

    comp_lens = [
        sum(G[u][v]["length"] for u, v in G.subgraph(c).edges()) / 1000
        for c in comps
    ]
    total_len = sum(comp_lens)

    # Bucket segments: one list per top-rank + one merged grey list
    buckets = {i: [] for i in range(N_TOP)}
    grey = []

    for u, v, c0, c1 in segments:
        rank = node_rank.get(u, node_rank.get(v, len(comps) - 1))
        line = LineString([c0, c1])
        if rank < N_TOP:
            buckets[rank].append(line)
        else:
            grey.append(line)

    print(f"    {len(comps)} components | LCC {comp_lens[0]:.0f} km | "
          f"grey: {len(grey)} segments")

    # Draw grey first (background), then coloured on top
    if grey:
        gpd.GeoDataFrame(geometry=grey, crs=CRS_PROJ).plot(
            ax=ax, color="#cccccc", linewidth=0.35, alpha=0.55, zorder=1
        )

    for rank in range(N_TOP - 1, -1, -1):  # draw largest last (on top)
        segs = buckets[rank]
        if not segs:
            continue
        lw    = 0.9 if rank == 0 else 0.75
        alpha = 0.95 if rank == 0 else 0.88
        gpd.GeoDataFrame(geometry=segs, crs=CRS_PROJ).plot(
            ax=ax, color=PALETTE[rank], linewidth=lw, alpha=alpha, zorder=rank + 2
        )

    # Legend
    handles = []
    for i in range(min(N_TOP, len(comps))):
        if not buckets[i]:
            continue
        pct = comp_lens[i] / total_len * 100
        tag = "LCC" if i == 0 else f"#{i + 1}"
        lbl = f"{tag}: {comp_lens[i]:.0f} km ({pct:.0f}%)"
        handles.append(mpatches.Patch(color=PALETTE[i], label=lbl))
    if len(comps) > N_TOP:
        grey_len = sum(comp_lens[N_TOP:])
        handles.append(mpatches.Patch(
            color="#cccccc",
            label=f"+{len(comps) - N_TOP} more: {grey_len:.0f} km ({grey_len/total_len*100:.0f}%)",
        ))

    ax.legend(handles=handles, fontsize=6.5, loc="upper right",
              framealpha=0.92, title=f"{len(comps)} components total",
              title_fontsize=7)
    ax.set_title(
        f"{label}\n{total_len:.0f} km  |  {len(comps)} comp.  |  "
        f"LCC {comp_lens[0]:.0f} km ({comp_lens[0]/total_len*100:.0f}%)",
        fontsize=10, fontweight="bold",
    )
    ax.set_axis_off()


# ── Load data ──────────────────────────────────────────────────────────────
print("Loading data...")
df_raw = pd.read_csv(REAL_CSV, encoding="utf-8-sig", engine="python", on_bad_lines="warn")
df_raw["geometry"] = df_raw["WKT"].apply(safe_wkt)
raw = gpd.GeoDataFrame(df_raw, geometry="geometry", crs=CRS_IN)

gdf_A = raw[raw["MERKMAL"] == "Getrennte Führung"].copy()
gdf_A = gdf_A[~gdf_A.geometry.isna() & ~gdf_A.geometry.is_empty].to_crs(CRS_PROJ)

gdf_B = raw[raw["MERKMAL"] == "Markierte Anlagen"].copy()
gdf_B = gdf_B[~gdf_B.geometry.isna() & ~gdf_B.geometry.is_empty].to_crs(CRS_PROJ)

gdf_C = raw.copy()
gdf_C = gdf_C[~gdf_C.geometry.isna() & ~gdf_C.geometry.is_empty].to_crs(CRS_PROJ)

# ── Plot ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(22, 9))
fig.suptitle(
    "Fragmentation of the Vienna bicycle network\n"
    "(colour = connected component  |  LCC = dark blue  |  grey = isolated segments)",
    fontsize=13, fontweight="bold",
)

for gdf, label, ax in [
    (gdf_A, "A – Separated infrastructure", axes[0]),
    (gdf_B, "B – Marked lanes",             axes[1]),
    (gdf_C, "C – Full network",             axes[2]),
]:
    plot_fragmentation(gdf, label, ax)

plt.tight_layout()
out_path = OUT / "fragmentation_map_en.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\n→ {out_path}")
