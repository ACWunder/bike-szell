"""
Bootstrap directness uncertainty for each synthetic stage of each city.

Each stage's directness statistic (mean over 1000 random pairs in the LCC)
is a single point estimate.  We re-sample 10 times to get an IQR band that
shows how stable the headline directness numbers really are.

Per stage:
  1. Load the synthetic-stage GeoJSON, project to UTM, build a networkx graph
  2. Take the largest connected component
  3. Pick 100 random source nodes, compute single-source Dijkstra to all
     other nodes once (cached)
  4. Draw 10 bootstrap samples — each picks 1000 source-target pairs from
     the cached source set, computes the mean directness ratio
  5. Record median + Q25 + Q75 of the 10 sample means

Output: analysis_output/comparison/directness_bootstrap.csv
        columns: city, stage_idx, quantile, length_km,
                 dir_median, dir_q25, dir_q75, dir_min, dir_max

Usage: conda run -n growbikenet python analysis_directness_bootstrap.py
"""

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx

# ── Config ──────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent / "bikenwgrowth-data"
OUT_COMP = BASE / "analysis_output" / "comparison"
OUT_COMP.mkdir(parents=True, exist_ok=True)

CITIES = ["amsterdam", "barcelona", "berlin", "oslo", "vienna"]

# Sample only every N-th stage to keep total runtime manageable
STAGE_STRIDE   = 4
N_SOURCES      = 100      # source nodes whose distance maps we cache
N_BOOTSTRAPS   = 10       # bootstrap repeats
PAIRS_PER_BOOT = 1000     # pairs per bootstrap mean
BASE_SEED      = 42


def build_graph(geojson_path):
    """Load synthetic stage GeoJSON and return a metric undirected graph."""
    gdf = gpd.read_file(geojson_path)
    if len(gdf) == 0:
        return None
    crs_p = gdf.estimate_utm_crs()
    gdf_p = gdf.to_crs(crs_p)
    G = nx.Graph()
    for geom in gdf_p.geometry:
        coords = list(geom.coords)
        for a, b in zip(coords[:-1], coords[1:]):
            length = ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
            # If the edge already exists, keep the shorter one.
            if G.has_edge(a, b):
                if G[a][b]["weight"] > length:
                    G[a][b]["weight"] = length
            else:
                G.add_edge(a, b, weight=length)
    return G


def bootstrap_directness(G, n_sources=N_SOURCES, n_bootstraps=N_BOOTSTRAPS,
                         pairs_per_boot=PAIRS_PER_BOOT, seed=BASE_SEED):
    """Return a list of n_bootstraps bootstrap mean directness values."""
    if G.number_of_nodes() == 0:
        return [float("nan")] * n_bootstraps
    # Largest connected component
    cc = max(nx.connected_components(G), key=len)
    nodes_cc = list(cc)
    n_nodes = len(nodes_cc)
    if n_nodes < 4:
        return [float("nan")] * n_bootstraps

    rng = np.random.default_rng(seed)
    src_idx = rng.choice(n_nodes,
                          size=min(n_sources, n_nodes),
                          replace=False)

    # Cache single-source Dijkstra per source — re-used across all bootstraps
    src_dists = {}
    for si in src_idx:
        s = nodes_cc[si]
        src_dists[int(si)] = nx.single_source_dijkstra_path_length(
            G, s, weight="weight"
        )

    sample_means = []
    for boot in range(n_bootstraps):
        boot_rng = np.random.default_rng(seed + 1 + boot)
        ratios = []
        # Pick pairs_per_boot pairs: random source from the cache, random target
        src_picks  = boot_rng.choice(src_idx, size=pairs_per_boot, replace=True)
        tgt_picks  = boot_rng.choice(n_nodes,  size=pairs_per_boot, replace=True)
        for si, ti in zip(src_picks, tgt_picks):
            s = nodes_cc[si]
            t = nodes_cc[ti]
            if s == t:
                continue
            net_d = src_dists[int(si)].get(t)
            if net_d is None or net_d <= 0:
                continue
            eucl_d = ((s[0] - t[0]) ** 2 + (s[1] - t[1]) ** 2) ** 0.5
            ratios.append(eucl_d / net_d)
        sample_means.append(float(np.mean(ratios)) if ratios else float("nan"))

    return sample_means


# ── Main loop ──────────────────────────────────────────────────────────────
rows = []
t_start = time.time()

for city in CITIES:
    print(f"\n{'='*60}\n  {city}\n{'='*60}")
    idx_path = BASE / "analysis_output" / city / "synthetic_stages" / "index.json"
    stages_dir = idx_path.parent
    with open(idx_path) as f:
        idx = json.load(f)

    # Stage indices to evaluate (every STAGE_STRIDE'th + always the last)
    stage_indices = sorted(set(
        list(range(0, len(idx), STAGE_STRIDE)) + [len(idx) - 1]
    ))
    print(f"  Evaluating {len(stage_indices)} / {len(idx)} stages")

    for k, st_i in enumerate(stage_indices):
        stage = idx[st_i]
        t_stage = time.time()
        G = build_graph(stages_dir / stage["file"])
        if G is None:
            continue
        means = bootstrap_directness(G)
        means_arr = np.array([m for m in means if not np.isnan(m)])
        if len(means_arr) == 0:
            row = dict(
                city=city, stage_idx=st_i,
                quantile=float(stage["quantile"]),
                length_km=float(stage["length_km"]),
                dir_median=np.nan, dir_q25=np.nan, dir_q75=np.nan,
                dir_min=np.nan, dir_max=np.nan,
            )
        else:
            row = dict(
                city=city, stage_idx=st_i,
                quantile=float(stage["quantile"]),
                length_km=float(stage["length_km"]),
                dir_median=float(np.median(means_arr)),
                dir_q25=float(np.quantile(means_arr, 0.25)),
                dir_q75=float(np.quantile(means_arr, 0.75)),
                dir_min=float(means_arr.min()),
                dir_max=float(means_arr.max()),
            )
        rows.append(row)
        spread = row["dir_q75"] - row["dir_q25"] if not np.isnan(row["dir_q25"]) else float("nan")
        print(f"  stage {st_i:2d}  Q={stage['quantile']:.3f}  "
              f"len={stage['length_km']:5.0f}km  "
              f"dir={row['dir_median']:.3f} (IQR {spread:.4f})  "
              f"[{time.time()-t_stage:5.1f}s]")

df = pd.DataFrame(rows)
out_csv = OUT_COMP / "directness_bootstrap.csv"
df.to_csv(out_csv, index=False)
print(f"\nSaved: {out_csv}")
print(f"Total time: {time.time()-t_start:.0f}s")
