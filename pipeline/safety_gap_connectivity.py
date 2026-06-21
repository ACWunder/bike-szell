"""
Island-bridging analysis for the SAFETY section.

Question: the protected network A is broken into many disconnected islands. The
A-vs-C gap is the set of synthetic-optimum edges that are NOT yet protected. If
those gap links were built as protected infrastructure, how much would A's
fragmentation drop?

Method (per city, all metric / UTM):
  1. Load A (biketrack, incl. snapped crossings) as an undirected multigraph from
     the canonical graph files -> baseline components + LCC length share.
  2. Take the gap edges (gap_{city}_safety.geojson = C edges not on A) and add them
     to A: each gap vertex is snapped to the nearest A node within TOL metres (so a
     gap link that reaches the protected network docks onto it), and gap-internal
     vertices are merged on a 2 m grid (so a contiguous gap run stays connected).
  3. Recompute components + LCC share on A + gap.

Outputs analysis_output/comparison/safety_gap_connectivity.csv and prints a table.

Usage: conda run -n growbikenet python pipeline/safety_gap_connectivity.py
"""
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import pyproj
from scipy.spatial import cKDTree

BASE = Path(__file__).resolve().parent.parent / "bikenwgrowth-data"
CITIES = ["amsterdam", "barcelona", "berlin", "oslo", "vienna"]
CITY_LABELS = {c: c.capitalize() for c in CITIES}

TOL_M = 15.0      # gap vertex docks onto A node within this distance (= analysis buffer)
GRID_M = 2.0      # gap-internal vertices merged on this grid


def read_zip_csv(zpath):
    with zipfile.ZipFile(zpath) as z:
        name = [n for n in z.namelist() if n.endswith(".csv")][0]
        with z.open(name) as f:
            return pd.read_csv(f, low_memory=False)


def load_A_graph(city):
    """A (biketrack) as an undirected multigraph with node x/y (EPSG:4326)."""
    nodes = read_zip_csv(BASE / "data" / city / f"{city}_biketrack_nodes.zip")
    edges = read_zip_csv(BASE / "data" / city / f"{city}_biketrack_edges.zip")
    # Collapse OSMnx directed duplicates (u->v and v->u) to one undirected edge per
    # physical segment, keyed on the unordered node pair + key — matches csv_to_ig and
    # existing.csv, so the baseline length / LCC share line up with the report.
    edges = edges[edges["length"].notna()].copy()
    u = edges["u"].to_numpy(); v = edges["v"].to_numpy()
    edges["_lo"] = np.minimum(u, v); edges["_hi"] = np.maximum(u, v)
    keycol = "key" if "key" in edges.columns else None
    subset = ["_lo", "_hi"] + ([keycol] if keycol else [])
    edges = edges.drop_duplicates(subset=subset)
    G = nx.MultiGraph()
    for _, r in nodes.iterrows():
        G.add_node(int(r["osmid"]), x=float(r["x"]), y=float(r["y"]))
    for _, r in edges.iterrows():
        G.add_edge(int(r["u"]), int(r["v"]), length=float(r["length"]))
    return G


def comp_lcc(G):
    """(#components, total_len_km, lcc_len_km, lcc_share)."""
    ccs = list(nx.connected_components(G))
    comp_of = {}
    for i, c in enumerate(ccs):
        for n in c:
            comp_of[n] = i
    lengths = np.zeros(len(ccs))
    for u, v, d in G.edges(data=True):
        lengths[comp_of[u]] += d.get("length", 0.0)
    total = lengths.sum()
    lcc = lengths.max() if len(lengths) else 0.0
    return len(ccs), total / 1000, lcc / 1000, (lcc / total if total else 0.0)


rows = []
for city in CITIES:
    label = CITY_LABELS[city]
    print(f"\n=== {label} ===", flush=True)

    G = load_A_graph(city)
    n0, tot0, lcc0, share0 = comp_lcc(G)
    print(f"  A alone:      components={n0:5d}  length={tot0:7.1f} km  "
          f"LCC={lcc0:7.1f} km  LCC-share={100*share0:5.1f}%", flush=True)

    # project A node coords to per-city UTM, build KD-tree
    a_ids = list(G.nodes())
    lon = np.array([G.nodes[n]["x"] for n in a_ids])
    lat = np.array([G.nodes[n]["y"] for n in a_ids])
    pts = gpd.GeoSeries(gpd.points_from_xy(lon, lat), crs="EPSG:4326")
    utm = pts.estimate_utm_crs()
    pts_m = pts.to_crs(utm)
    a_xy = np.column_stack([pts_m.x.values, pts_m.y.values])
    tree = cKDTree(a_xy)
    transformer = pyproj.Transformer.from_crs("EPSG:4326", utm, always_xy=True)

    # load gap edges, project
    gap = gpd.read_file(BASE / "analysis_output" / city / f"gap_{city}_safety.geojson")
    gap_m = gap.to_crs(utm)

    G2 = G.copy()

    def node_for(x, y):
        """Return an existing A node id if within TOL, else a grid-keyed gap node."""
        d, idx = tree.query([x, y])
        if d <= TOL_M:
            return a_ids[idx]
        gid = (round(x / GRID_M) * GRID_M, round(y / GRID_M) * GRID_M)
        if gid not in G2:
            G2.add_node(gid, x=x, y=y)
        return gid

    added = 0
    for geom in gap_m.geometry:
        if geom is None or geom.is_empty:
            continue
        parts = geom.geoms if geom.geom_type == "MultiLineString" else [geom]
        for part in parts:
            cs = list(part.coords)
            if len(cs) < 2:
                continue
            prev_id = node_for(cs[0][0], cs[0][1])
            prev_xy = cs[0]
            for (x, y) in cs[1:]:
                cur_id = node_for(x, y)
                seglen = float(np.hypot(x - prev_xy[0], y - prev_xy[1]))
                if cur_id != prev_id:
                    G2.add_edge(prev_id, cur_id, length=seglen)
                    added += 1
                prev_id, prev_xy = cur_id, (x, y)

    n1, tot1, lcc1, share1 = comp_lcc(G2)
    print(f"  A + built gap: components={n1:5d}  length={tot1:7.1f} km  "
          f"LCC={lcc1:7.1f} km  LCC-share={100*share1:5.1f}%  ({added} gap segs)", flush=True)
    print(f"  >>> components {n0}->{n1} ({n1-n0:+d}),  "
          f"LCC-share {100*share0:.1f}%->{100*share1:.1f}% ({100*(share1-share0):+.1f}pp)",
          flush=True)

    rows.append({
        "City": label,
        "A components": n0,
        "A LCC share %": round(100 * share0, 1),
        "A+gap components": n1,
        "A+gap LCC share %": round(100 * share1, 1),
        "components removed": n0 - n1,
        "LCC share gain pp": round(100 * (share1 - share0), 1),
        "A length km": round(tot0, 1),
        "A+gap length km": round(tot1, 1),
    })

out = pd.DataFrame(rows)
out.to_csv(BASE / "analysis_output" / "comparison" / "safety_gap_connectivity.csv", index=False)
print("\n" + out.to_string(index=False))
print("\nSaved: safety_gap_connectivity.csv")
