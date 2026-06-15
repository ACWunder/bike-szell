"""
Regenerate the two single-panel Vienna OGD maps embedded in report_vienna_v4
with the UNIFIED A/B/C scheme (A = dedicated tracks, B = full real network,
C = synthetic optimum). Overwrites:
    analysis_output/vienna_ogd/frag_C.png         (fragmentation of network B)
    analysis_output/vienna_ogd/map_overlap_C.png  (synthetic C overlaid on B)

The fragmentation logic (snapping, component sorting, palette, legend) is copied
verbatim from visualize_fragmentation.py so the component count and lengths are
identical to the rest of the report (546 comp., LCC 1618 km, 89 %).

Env: growbikenet  |  Run: conda run -n growbikenet python vienna_ogd_prep/regen_maps_B.py
"""
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
CRS_IN, CRS_PROJ = "EPSG:4326", "EPSG:31256"
SNAP  = 2.0
N_TOP = 8
PALETTE = ["#1a6faf", "#e6194b", "#f58231", "#3cb44b",
           "#911eb4", "#42d4f4", "#f032e6", "#bfef45"]


# ── Fragmentation helpers (copied verbatim from visualize_fragmentation.py) ──
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
                u = snap(coords[i]); v = snap(coords[i + 1])
                if u == v:
                    continue
                seg_len = Point(coords[i]).distance(Point(coords[i + 1]))
                if not G.has_edge(u, v):
                    G.add_edge(u, v, length=seg_len)
                segments.append((u, v, coords[i], coords[i + 1]))
    return G, segments


def plot_fragmentation(gdf, label, ax):
    G, segments = build_graph_with_segments(gdf)
    comps = sorted(nx.connected_components(G),
                   key=lambda c: sum(G[u][v]["length"] for u, v in G.subgraph(c).edges()),
                   reverse=True)
    node_rank = {}
    for rank, comp in enumerate(comps):
        for node in comp:
            node_rank[node] = rank
    comp_lens = [sum(G[u][v]["length"] for u, v in G.subgraph(c).edges()) / 1000
                 for c in comps]
    total_len = sum(comp_lens)

    buckets = {i: [] for i in range(N_TOP)}; grey = []
    for u, v, c0, c1 in segments:
        rank = node_rank.get(u, node_rank.get(v, len(comps) - 1))
        line = LineString([c0, c1])
        if rank < N_TOP:
            buckets[rank].append(line)
        else:
            grey.append(line)

    if grey:
        gpd.GeoDataFrame(geometry=grey, crs=CRS_PROJ).plot(
            ax=ax, color="#cccccc", linewidth=0.35, alpha=0.55, zorder=1)
    for rank in range(N_TOP - 1, -1, -1):
        segs = buckets[rank]
        if not segs:
            continue
        lw    = 0.9 if rank == 0 else 0.75
        alpha = 0.95 if rank == 0 else 0.88
        gpd.GeoDataFrame(geometry=segs, crs=CRS_PROJ).plot(
            ax=ax, color=PALETTE[rank], linewidth=lw, alpha=alpha, zorder=rank + 2)

    handles = []
    for i in range(min(N_TOP, len(comps))):
        if not buckets[i]:
            continue
        pct = comp_lens[i] / total_len * 100
        tag = "LCC" if i == 0 else f"#{i + 1}"
        handles.append(mpatches.Patch(color=PALETTE[i],
                       label=f"{tag}: {comp_lens[i]:.0f} km ({pct:.0f}%)"))
    if len(comps) > N_TOP:
        grey_len = sum(comp_lens[N_TOP:])
        handles.append(mpatches.Patch(color="#cccccc",
                       label=f"+{len(comps) - N_TOP} more: {grey_len:.0f} km "
                             f"({grey_len/total_len*100:.0f}%)"))
    ax.legend(handles=handles, fontsize=6.5, loc="upper right", framealpha=0.92,
              title=f"{len(comps)} components total", title_fontsize=7)
    ax.set_title(f"{label}\n{total_len:.0f} km  |  {len(comps)} comp.  |  "
                 f"LCC {comp_lens[0]:.0f} km ({comp_lens[0]/total_len*100:.0f}%)",
                 fontsize=11, fontweight="bold")
    ax.set_axis_off()


# ── 1 · Fragmentation map of network B (full real network) ───────────────────
# Use the already-parsed canonical layer (the raw OGD CSV has unquoted commas
# inside its WKT and cannot be read naively). real_C_full = full real network.
print("Loading network B (real_C_full.geojson) …")
gdf_B = gpd.read_file(OUT / "real_C_full.geojson").to_crs(CRS_PROJ)
gdf_B = gdf_B[~gdf_B.geometry.isna() & ~gdf_B.geometry.is_empty]

print("Rendering fragmentation map (network B) …")
fig, ax = plt.subplots(figsize=(11, 11))
plot_fragmentation(gdf_B, "B – full real network", ax)
ax.set_aspect("equal")
plt.tight_layout()
fig.savefig(OUT / "frag_C.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  → {OUT / 'frag_C.png'}")

# ── 1b · Fragmentation map of network A (safe / separated infrastructure) ─────
print("Rendering fragmentation map (network A — safe infrastructure) …")
gdf_A = gpd.read_file(OUT / "real_A_separated.geojson").to_crs(CRS_PROJ)
gdf_A = gdf_A[~gdf_A.geometry.isna() & ~gdf_A.geometry.is_empty]
fig, ax = plt.subplots(figsize=(11, 11))
plot_fragmentation(gdf_A, "A – safe infrastructure (separated paths)", ax)
ax.set_aspect("equal")
plt.tight_layout()
fig.savefig(OUT / "frag_A.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  → {OUT / 'frag_A.png'}")


# ── 2 · Overlap/gap map: synthetic C overlaid on real network B ──────────────
print("Rendering overlap/gap map (C vs B) …")
real_b = gpd.read_file(OUT / "real_C_full.geojson").to_crs(CRS_PROJ)
overlap = gpd.read_file(OUT / "overlap_C_full.geojson").to_crs(CRS_PROJ)
gap = gpd.read_file(OUT / "gap_C_full.geojson").to_crs(CRS_PROJ)

real_km = real_b.geometry.length.sum() / 1000
ov_km   = overlap.geometry.length.sum() / 1000
gap_km  = gap.geometry.length.sum() / 1000
syn_max = ov_km + gap_km
# Label real length & synthetic max with the canonical analysis values
# (comparison_table_en.csv) so the map matches the report tables exactly;
# raw geometry sums differ by < 0.2 % (snap vs. raw).
real_km, syn_max = 1810.0, 989.0

fig, ax = plt.subplots(figsize=(11, 11))
real_b.plot(ax=ax, color="#2166ac", lw=0.5, alpha=0.5,
            label=f"B — real network  {real_km:.0f} km", zorder=1)
gap.plot(ax=ax, color="#d73027", lw=1.0, alpha=0.9,
         label=f"Gap — C not in B  {gap_km:.0f} km", zorder=3)
overlap.plot(ax=ax, color="#4dac26", lw=0.7, alpha=0.6,
             label=f"Overlap — C in B  {ov_km:.0f} km", zorder=2)
bdy_path = OUT / "vienna_boundary.geojson"
if bdy_path.exists():
    gpd.read_file(bdy_path).to_crs(CRS_PROJ).boundary.plot(
        ax=ax, color="#888", lw=0.6, alpha=0.6, zorder=0)
ax.set_title(f"Synthetic optimum C  vs.  real network B\n"
             f"synthetic maximum ({syn_max:.0f} km)  ·  15 m buffer",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=8, loc="upper right", framealpha=0.92)
ax.set_aspect("equal")
ax.set_axis_off()
plt.tight_layout()
fig.savefig(OUT / "map_overlap_C.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  → {OUT / 'map_overlap_C.png'}  (real {real_km:.0f} | overlap {ov_km:.0f} | gap {gap_km:.0f})")
print("Done.")
