"""
Structural comparison: Vienna bicycle network vs. synthetically grown network
Variant A: Separated infrastructure (Getrennte Führung)
Variant B: Marked lanes (Markierte Anlagen)
Variant C: Full network (all categories)
Env: growbikenet
"""
import json
import random
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from shapely import wkt as shapely_wkt
from shapely.geometry import Point

random.seed(42); np.random.seed(42)

# ── Paths ──────────────────────────────────────────────────────────────────
BASE      = Path(__file__).resolve().parent.parent / "bikenwgrowth-data"
REAL_CSV  = Path(__file__).resolve().parent.parent / "data" / "vienna_ogd" / "Radwege_ogd.csv"
SYN_CSV   = BASE / "results/vienna_2/vienna_poi_grid_betweenness.csv"
EXIST_CSV = BASE / "results/vienna_2/vienna_existing.csv"
# Vienna OGD-derived outputs are kept under analysis_output/vienna_ogd/
# so they don't pollute the analysis_output root.
STAGES    = BASE / "analysis_output/vienna_ogd/synthetic_stages"
OUT       = BASE / "analysis_output/vienna_ogd"
OUT.mkdir(parents=True, exist_ok=True)

CRS_IN, CRS_PROJ = "EPSG:4326", "EPSG:31256"

assert REAL_CSV.exists(), f"Not found: {REAL_CSV}"
assert STAGES.exists(), "Run first: conda run -n growbikenet python extract_synthetic_geo.py"
print("All paths OK.")

# ── 1 · Load real network ──────────────────────────────────────────────────
def safe_wkt(x):
    try:
        return shapely_wkt.loads(x) if isinstance(x, str) and x.strip() else None
    except Exception:
        return None

df_raw = pd.read_csv(REAL_CSV, encoding="utf-8-sig", engine="python", on_bad_lines="warn")
df_raw["geometry"] = df_raw["WKT"].apply(safe_wkt)
raw = gpd.GeoDataFrame(df_raw, geometry="geometry", crs=CRS_IN)

print(f"\nTotal features in dataset: {len(raw)}")
print(raw["MERKMAL"].value_counts().to_string())

# Variant A: Separated infrastructure
gdf_A = raw[raw["MERKMAL"] == "Getrennte Führung"].copy()
gdf_A = gdf_A[~gdf_A.geometry.isna() & ~gdf_A.geometry.is_empty].to_crs(CRS_PROJ)

# Variant B: Marked lanes
gdf_B = raw[raw["MERKMAL"] == "Markierte Anlagen"].copy()
gdf_B = gdf_B[~gdf_B.geometry.isna() & ~gdf_B.geometry.is_empty].to_crs(CRS_PROJ)

# Variant C: Full network (all classes)
gdf_C = raw.copy()
gdf_C = gdf_C[~gdf_C.geometry.isna() & ~gdf_C.geometry.is_empty].to_crs(CRS_PROJ)

len_A = gdf_A.geometry.length.sum() / 1000
len_B = gdf_B.geometry.length.sum() / 1000
len_C = gdf_C.geometry.length.sum() / 1000

print(f"\n{'Variant':<35} {'Segments':>8}  {'Length (km)':>11}")
print("-" * 58)
print(f"{'A – Separated infrastructure':<35} {len(gdf_A):>8}  {len_A:>11.1f}")
print(f"{'B – Marked lanes':<35} {len(gdf_B):>8}  {len_B:>11.1f}")
print(f"{'C – Full network (all classes)':<35} {len(gdf_C):>8}  {len_C:>11.1f}")

# ── 2 · Synthetic growth curve ────────────────────────────────────────────
syn_df   = pd.read_csv(SYN_CSV)
exist_df = pd.read_csv(EXIST_CSV)
n = len(syn_df)
syn_df["quantile"]  = [round(i/n, 3) for i in range(1, n+1)]
syn_df["length_km"] = syn_df["length"] / 1000
syn_df["lcc_share"] = syn_df["length_lcc"] / syn_df["length"]

with open(STAGES / "index.json") as f:
    stage_idx = pd.DataFrame(json.load(f))

biketrack = exist_df[exist_df["network"] == "biketrack"].iloc[0]
bikeable  = exist_df[exist_df["network"] == "bikeable"].iloc[0]

def match_stage(km):
    i = (syn_df["length_km"] - km).abs().idxmin()
    return syn_df.loc[i]

m_A, m_B, m_C = match_stage(len_A), match_stage(len_B), match_stage(len_C)

print(f"\n{'Variant':<35} {'Real (km)':>10}  {'Syn. quantile':>14}  {'Syn. (km)':>10}")
print("-" * 75)
print(f"{'A – Separated infrastructure':<35} {len_A:>10.1f}  {m_A['quantile']:>14.3f}  {m_A['length_km']:>10.1f}")
print(f"{'B – Marked lanes':<35} {len_B:>10.1f}  {m_B['quantile']:>14.3f}  {m_B['length_km']:>10.1f}")
print(f"{'C – Full network':<35} {len_C:>10.1f}  {'> max (988 km)':>14}  {syn_df['length_km'].max():>10.1f}")
print(f"\nSynthetic maximum: {syn_df['length_km'].max():.1f} km")
print(f"OSM biketrack:     {biketrack['length']/1000:.1f} km")
print(f"OSM bikeable:      {bikeable['length']/1000:.1f} km")

# ── 3 · Build graphs ───────────────────────────────────────────────────────
def build_graph(gdf, snap_m=2.0):
    G = nx.Graph()
    for geom in gdf.geometry:
        if geom is None or geom.is_empty: continue
        lines = geom.geoms if geom.geom_type == "MultiLineString" else [geom]
        for line in lines:
            coords = list(line.coords)
            for i in range(len(coords) - 1):
                u = (round(coords[i][0]/snap_m)*snap_m, round(coords[i][1]/snap_m)*snap_m)
                v = (round(coords[i+1][0]/snap_m)*snap_m, round(coords[i+1][1]/snap_m)*snap_m)
                if u == v: continue
                seg = Point(coords[i]).distance(Point(coords[i+1]))
                if G.has_edge(u, v):
                    if seg < G[u][v]["length"]: G[u][v]["length"] = seg
                else:
                    G.add_edge(u, v, length=seg)
    return G

print("\nBuilding graphs...")
G_A = build_graph(gdf_A); print(f"  A: {G_A.number_of_nodes():,} nodes, {G_A.number_of_edges():,} edges")
G_B = build_graph(gdf_B); print(f"  B: {G_B.number_of_nodes():,} nodes, {G_B.number_of_edges():,} edges")
G_C = build_graph(gdf_C); print(f"  C: {G_C.number_of_nodes():,} nodes, {G_C.number_of_edges():,} edges")

# ── 4 · Graph metrics ──────────────────────────────────────────────────────
def graph_metrics(G, name, n_sample=1000):
    """
    Computes:
    - components:        number of connected components
    - lcc_share:         share of largest connected component by length
    - directness_lcc:    mean(euclidean / network distance) for connected pairs in LCC only
    - efficiency_global: mean(euclidean / network distance) over ALL node pairs
                         (disconnected pairs count as 0) → penalises fragmentation
    """
    if G.number_of_nodes() == 0:
        print(f"{name}: Graph empty!"); return None
    comps = list(nx.connected_components(G))
    lcc_nodes = max(comps, key=len)
    G_lcc = G.subgraph(lcc_nodes).copy()
    total_len = sum(d["length"] for _,_,d in G.edges(data=True))
    lcc_len   = sum(d["length"] for _,_,d in G_lcc.edges(data=True))
    lcc_share = lcc_len / total_len if total_len else 0

    all_nodes = list(G.nodes())

    # Directness (LCC only): connected pairs within the largest component
    lcc_node_list = list(lcc_nodes)
    pairs_lcc, seen = [], set()
    for _ in range(n_sample * 8):
        if len(pairs_lcc) >= n_sample: break
        a, b = random.sample(lcc_node_list, 2)
        key = (min(a,b), max(a,b))
        if key not in seen:
            seen.add(key); pairs_lcc.append(key)
    dvals_lcc = []
    for a, b in pairs_lcc:
        try:
            nd = nx.shortest_path_length(G_lcc, a, b, weight="length")
            ed = Point(a).distance(Point(b))
            if nd > 0 and ed > 0: dvals_lcc.append(ed/nd)
        except nx.NetworkXNoPath: pass
    directness = float(np.mean(dvals_lcc)) if dvals_lcc else None

    # Global efficiency: all node pairs, disconnected = 0
    pairs_all, seen2 = [], set()
    for _ in range(n_sample * 8):
        if len(pairs_all) >= n_sample: break
        a, b = random.sample(all_nodes, 2)
        key = (min(a,b), max(a,b))
        if key not in seen2:
            seen2.add(key); pairs_all.append(key)
    eff_vals = []
    for a, b in pairs_all:
        try:
            nd = nx.shortest_path_length(G, a, b, weight="length")
            ed = Point(a).distance(Point(b))
            if nd > 0 and ed > 0:
                eff_vals.append(ed / nd)
        except nx.NetworkXNoPath:
            eff_vals.append(0.0)
    efficiency_global = float(np.mean(eff_vals)) if eff_vals else None

    print(f"  {name}: {total_len/1000:.1f} km | {len(comps)} comp. | "
          f"LCC {lcc_len/1000:.1f} km ({lcc_share:.1%}) | "
          f"directness {directness:.3f} | efficiency {efficiency_global:.3f}")
    return dict(name=name, length_km=total_len/1000, lcc_length_km=lcc_len/1000,
                components=len(comps), lcc_share=lcc_share,
                directness_lcc=directness, efficiency_global=efficiency_global,
                G_lcc=G_lcc)

print("Computing metrics (takes ~3 min)...")
met_A = graph_metrics(G_A, "A – Separated infrastructure")
met_B = graph_metrics(G_B, "B – Marked lanes")
met_C = graph_metrics(G_C, "C – Full network")

# ── 4b · Coverage (area within 500m buffer) ───────────────────────────────
import osmnx as ox

COVERAGE_BUFFER_M = 500
VIENNA_BOUNDARY_FILE = OUT / "vienna_boundary.geojson"

if VIENNA_BOUNDARY_FILE.exists():
    print("\nLoading Vienna boundary from cache...")
    vienna = gpd.read_file(VIENNA_BOUNDARY_FILE).to_crs(CRS_PROJ)
else:
    print("\nLoading Vienna boundary from OpenStreetMap...")
    vienna_raw = ox.geocode_to_gdf("Vienna, Austria")
    vienna_raw.to_file(VIENNA_BOUNDARY_FILE, driver="GeoJSON")
    vienna = vienna_raw.to_crs(CRS_PROJ)

vienna_area_km2 = vienna.geometry.area.sum() / 1e6
print(f"Vienna city area: {vienna_area_km2:.1f} km²")
vienna_union = vienna.geometry.unary_union

def compute_coverage(gdf, city_polygon, buffer_m=COVERAGE_BUFFER_M):
    """Area in km² within buffer_m of the network, clipped to city boundary."""
    buffered = gdf.geometry.buffer(buffer_m).unary_union
    covered  = buffered.intersection(city_polygon)
    return covered.area / 1e6

print("Computing coverage...")
cov_A = compute_coverage(gdf_A, vienna_union)
cov_B = compute_coverage(gdf_B, vienna_union)
cov_C = compute_coverage(gdf_C, vienna_union)

# Coverage from synthetic CSV (already in km²)
cov_syn_A = float(match_stage(met_A["length_km"])["coverage"])
cov_syn_B = float(match_stage(met_B["length_km"])["coverage"])
cov_syn_max = float(syn_df.iloc[-1]["coverage"])
cov_osm_biketrack = float(biketrack["coverage"])
cov_osm_bikeable  = float(bikeable["coverage"])

print(f"\n{'Variant':<35} {'Coverage (km²)':>15}  {'% of Vienna':>11}")
print("-" * 65)
for label, cov in [
    ("A – Separated (real)", cov_A),
    (f"  Syn. Q={match_stage(met_A['length_km'])['quantile']:.3f}", cov_syn_A),
    ("  OSM biketrack", cov_osm_biketrack),
    ("B – Marked lanes (real)", cov_B),
    (f"  Syn. Q={match_stage(met_B['length_km'])['quantile']:.3f}", cov_syn_B),
    ("  OSM bikeable", cov_osm_bikeable),
    ("C – Full network (real)", cov_C),
    ("  Syn. maximum (Q=1.0)", cov_syn_max),
]:
    print(f"  {label:<33} {cov:>15.1f}  {cov/vienna_area_km2:>11.1%}")

# Store in metric dicts for table
met_A["coverage_km2"] = cov_A
met_B["coverage_km2"] = cov_B
met_C["coverage_km2"] = cov_C

# ── 5 · Comparison table ───────────────────────────────────────────────────
m_A = match_stage(met_A["length_km"])
m_B = match_stage(met_B["length_km"])

rows = []
for label, met, m, osm_row in [
    ("A: Separated infrastructure", met_A, m_A, biketrack),
    ("B: Marked lanes",             met_B, m_B, bikeable),
]:
    rows += [
        {"Network": label + " (real)",
         "Length (km)": f"{met['length_km']:.1f}",
         "Components": met["components"],
         "LCC share": f"{met['lcc_share']:.1%}",
         "Coverage (km²)": f"{met['coverage_km2']:.1f}",
         "Coverage (%)": f"{met['coverage_km2']/vienna_area_km2:.1%}",
         "Directness": f"{met['directness_lcc']:.3f}",
         "Efficiency": f"{met['efficiency_global']:.3f}"},
        {"Network": f"  Syn. Q={m['quantile']:.3f}",
         "Length (km)": f"{m['length_km']:.1f}",
         "Components": int(m["components"]),
         "LCC share": f"{m['lcc_share']:.1%}",
         "Coverage (km²)": f"{m['coverage']:.1f}",
         "Coverage (%)": f"{m['coverage']/vienna_area_km2:.1%}",
         "Directness": f"{m['directness_lcc']:.3f}",
         "Efficiency": f"{m['efficiency_global']:.3f}"},
        {"Network": f"  OSM {osm_row['network']}",
         "Length (km)": f"{osm_row['length']/1000:.1f}",
         "Components": int(osm_row["components"]),
         "LCC share": f"{osm_row['length_lcc']/osm_row['length']:.1%}",
         "Coverage (km²)": f"{osm_row['coverage']:.1f}",
         "Coverage (%)": f"{osm_row['coverage']/vienna_area_km2:.1%}",
         "Directness": f"{osm_row['directness_lcc']:.3f}",
         "Efficiency": f"{osm_row['efficiency_global']:.3f}"},
        {"Network": "─"*28, "Length (km)": "", "Components": "", "LCC share": "",
         "Coverage (km²)": "", "Coverage (%)": "", "Directness": "", "Efficiency": ""},
    ]

# Variant C — no direct synthetic match possible (longer than synthetic max)
syn_max = syn_df.iloc[-1]
rows += [
    {"Network": "C: Full network (real)",
     "Length (km)": f"{met_C['length_km']:.1f}",
     "Components": met_C["components"],
     "LCC share": f"{met_C['lcc_share']:.1%}",
     "Coverage (km²)": f"{met_C['coverage_km2']:.1f}",
     "Coverage (%)": f"{met_C['coverage_km2']/vienna_area_km2:.1%}",
     "Directness": f"{met_C['directness_lcc']:.3f}",
     "Efficiency": f"{met_C['efficiency_global']:.3f}"},
    {"Network": "  Syn. maximum (Q=1.0)",
     "Length (km)": f"{syn_max['length_km']:.1f}",
     "Components": int(syn_max["components"]),
     "LCC share": f"{syn_max['lcc_share']:.1%}",
     "Coverage (km²)": f"{syn_max['coverage']:.1f}",
     "Coverage (%)": f"{syn_max['coverage']/vienna_area_km2:.1%}",
     "Directness": f"{syn_max['directness_lcc']:.3f}",
     "Efficiency": f"{syn_max['efficiency_global']:.3f}"},
    {"Network": "  OSM bikeable",
     "Length (km)": f"{bikeable['length']/1000:.1f}",
     "Components": int(bikeable["components"]),
     "LCC share": f"{bikeable['length_lcc']/bikeable['length']:.1%}",
     "Coverage (km²)": f"{bikeable['coverage']:.1f}",
     "Coverage (%)": f"{bikeable['coverage']/vienna_area_km2:.1%}",
     "Directness": f"{bikeable['directness_lcc']:.3f}",
     "Efficiency": f"{bikeable['efficiency_global']:.3f}"},
]

tbl = pd.DataFrame(rows)
print("\n" + tbl.to_string(index=False))
tbl.to_csv(OUT / "comparison_table_en.csv", index=False)
print(f"\n→ {OUT / 'comparison_table_en.csv'}")

# ── 6 · Growth curves ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 5, figsize=(25, 5))
fig.suptitle("Synthetic network growth — Vienna (Grid + Betweenness)", fontsize=13, fontweight="bold")

C = {"syn": "#2166ac", "A": "#d73027", "B": "#fc8d59", "C": "#7b2d8b", "osm": "#4dac26"}

graph_plots = [
    ("Fragmentation", "components", "Connected components", 1,
     [(met_A, "A"), (met_B, "B"), (met_C, "C")]),
    ("LCC share", "lcc_share", "LCC share (%)", 100,
     [(met_A, "A"), (met_B, "B"), (met_C, "C")]),
    ("Directness (LCC)", "directness_lcc", "Directness (Eucl./Network)", 1,
     [(met_A, "A"), (met_B, "B"), (met_C, "C")]),
]

for ax, (title, col, ylabel, scale, variants) in zip(axes[:3], graph_plots):
    ax.plot(syn_df["length_km"], syn_df[col]*scale, color=C["syn"], lw=2, label="Synthetic")
    for met, var in variants:
        val = met[col if col != "lcc_share" else "lcc_share"]
        val_plot = val * scale if isinstance(val, float) else val
        km = met["length_km"]
        ax.axvline(km, color=C[var], ls="--", lw=1.5, alpha=0.8)
        if val_plot is not None:
            ax.axhline(val_plot, color=C[var], ls=":", lw=1, alpha=0.6)
            ax.plot(km, val_plot, "o", color=C[var], ms=9, zorder=5,
                    label=f"{var}: {met['name'].split('–')[1].strip()} ({km:.0f} km)")
    ax.axvline(biketrack["length"]/1000, color=C["osm"], ls="-.", lw=1, alpha=0.6, label="OSM biketrack")
    ax.axvline(syn_df["length_km"].max(), color=C["syn"], ls=":", lw=1, alpha=0.4)
    ax.set(xlabel="Network length (km)", ylabel=ylabel, title=title)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

# 4th panel: Coverage
ax = axes[3]
ax.plot(syn_df["length_km"], syn_df["coverage"], color=C["syn"], lw=2, label="Synthetic")
for met, var, cov_real in [
    (met_A, "A", cov_A),
    (met_B, "B", cov_B),
    (met_C, "C", cov_C),
]:
    km = met["length_km"]
    ax.axvline(km, color=C[var], ls="--", lw=1.5, alpha=0.8)
    ax.axhline(cov_real, color=C[var], ls=":", lw=1, alpha=0.6)
    ax.plot(km, cov_real, "o", color=C[var], ms=9, zorder=5,
            label=f"{var}: {met['name'].split('–')[1].strip()} ({km:.0f} km)")
ax.axhline(cov_osm_biketrack, color=C["osm"], ls="-.", lw=1, alpha=0.6,
           label=f"OSM biketrack ({cov_osm_biketrack:.0f} km²)")
ax.axvline(syn_df["length_km"].max(), color=C["syn"], ls=":", lw=1, alpha=0.4)
ax.set(xlabel="Network length (km)", ylabel="Coverage (km²)", title="Coverage (500m buffer)")
ax.legend(fontsize=7); ax.grid(alpha=0.3)

# 5th panel: Global efficiency
ax = axes[4]
ax.plot(syn_df["length_km"], syn_df["efficiency_global"], color=C["syn"], lw=2, label="Synthetic")
for met, var in [(met_A, "A"), (met_B, "B"), (met_C, "C")]:
    km = met["length_km"]
    val = met["efficiency_global"]
    ax.axvline(km, color=C[var], ls="--", lw=1.5, alpha=0.8)
    if val is not None:
        ax.axhline(val, color=C[var], ls=":", lw=1, alpha=0.6)
        ax.plot(km, val, "o", color=C[var], ms=9, zorder=5,
                label=f"{var}: {met['name'].split('–')[1].strip()} ({km:.0f} km)")
ax.axvline(biketrack["length"]/1000, color=C["osm"], ls="-.", lw=1, alpha=0.6, label="OSM biketrack")
ax.axvline(syn_df["length_km"].max(), color=C["syn"], ls=":", lw=1, alpha=0.4)
ax.set(xlabel="Network length (km)", ylabel="Global efficiency (Eucl./Network)", title="Global efficiency")
ax.legend(fontsize=7); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUT / "growth_curves_en.png", dpi=150, bbox_inches="tight")
print(f"→ {OUT / 'growth_curves_en.png'}")

# ── 7 · Spatial analysis ───────────────────────────────────────────────────
def load_stage(km):
    i = (stage_idx["length_km"] - km).abs().idxmin()
    row = stage_idx.loc[i]
    gdf = gpd.read_file(STAGES / row["file"]).to_crs(CRS_PROJ)
    print(f"  Q={row['quantile']:.3f}, {row['length_km']:.0f} km, {len(gdf)} edges")
    return gdf, row

def spatial_analysis(real_gdf, syn_gdf, buf=15.0, label=""):
    print(f"Spatial analysis {label} (buffer={buf}m)...")
    real_buf = real_gdf.geometry.buffer(buf).unary_union
    syn = syn_gdf.copy()
    syn["cov"] = syn.geometry.apply(
        lambda g: g.buffer(buf).intersection(real_buf).length / max(g.buffer(buf).length, 1e-9)
    )
    ov  = syn[syn["cov"] >= 0.5]
    gap = syn[syn["cov"] < 0.5]
    tot = syn.geometry.length.sum()
    ov_km  = ov.geometry.length.sum() / 1000
    gap_km = gap.geometry.length.sum() / 1000
    print(f"  Synthetic total: {tot/1000:.1f} km")
    print(f"  Overlap:         {ov_km:.1f} km  ({ov.geometry.length.sum()/tot:.1%})")
    print(f"  Gap:             {gap_km:.1f} km  ({gap.geometry.length.sum()/tot:.1%})")
    return dict(syn=syn, ov=ov, gap=gap, ov_km=ov_km, gap_km=gap_km,
                ov_ratio=ov.geometry.length.sum()/tot)

print("\nLoading synthetic stages...")
syn_A, srow_A = load_stage(met_A["length_km"])
syn_B, srow_B = load_stage(met_B["length_km"])
syn_C, srow_C = load_stage(syn_df["length_km"].max())

print()
sp_A = spatial_analysis(gdf_A, syn_A, label="A – Separated infrastructure")
print()
sp_B = spatial_analysis(gdf_B, syn_B, label="B – Marked lanes")
print()
sp_C = spatial_analysis(gdf_C, syn_C, label="C – Full network vs. Syn. max")

# ── 8 · Export for QGIS ────────────────────────────────────────────────────
print("\nExporting GeoJSONs...")
for suffix, real_gdf, sp in [
    ("A_separated", gdf_A, sp_A),
    ("B_marked",    gdf_B, sp_B),
    ("C_full",      gdf_C, sp_C),
]:
    for name, gdf in [("real", real_gdf), ("synthetic", sp["syn"]),
                      ("gap", sp["gap"]), ("overlap", sp["ov"])]:
        out_path = OUT / f"{name}_{suffix}.geojson"
        gdf.to_crs(CRS_IN).to_file(out_path, driver="GeoJSON")
        print(f"  {out_path.name:<45} {gdf.geometry.length.sum()/1000:.1f} km")

# ── 9 · Maps ───────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(21, 8))
for ax, (real_gdf, sp, srow, title) in zip(axes, [
    (gdf_A, sp_A, srow_A, f"A – Separated infrastructure\nSyn. Q={srow_A['quantile']:.3f} ({srow_A['length_km']:.0f} km)"),
    (gdf_B, sp_B, srow_B, f"B – Marked lanes\nSyn. Q={srow_B['quantile']:.3f} ({srow_B['length_km']:.0f} km)"),
    (gdf_C, sp_C, srow_C, f"C – Full network\nSyn. maximum ({srow_C['length_km']:.0f} km)"),
]):
    real_gdf.plot(ax=ax, color="#2166ac", lw=0.5, alpha=0.5,
                  label=f"Real {real_gdf.geometry.length.sum()/1000:.0f} km")
    sp["gap"].plot(ax=ax, color="#d73027", lw=1.0, alpha=0.9,
                   label=f"Gap {sp['gap_km']:.0f} km")
    sp["ov"].plot(ax=ax, color="#4dac26", lw=0.7, alpha=0.6,
                  label=f"Overlap {sp['ov_km']:.0f} km")
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_axis_off()

plt.tight_layout()
plt.savefig(OUT / "maps_en.png", dpi=150, bbox_inches="tight")
print(f"→ {OUT / 'maps_en.png'}")

# ── Summary ────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
for met, sp, srow, label in [
    (met_A, sp_A, srow_A, "A – Separated infrastructure"),
    (met_B, sp_B, srow_B, "B – Marked lanes"),
    (met_C, sp_C, srow_C, "C – Full network"),
]:
    m = match_stage(met["length_km"])
    print(f"\n{label}")
    print(f"  Real:    {met['length_km']:.1f} km | LCC {met['lcc_length_km']:.1f} km ({met['lcc_share']:.1%}) | "
          f"{met['components']} comp. | dir={met['directness_lcc']:.3f} | "
          f"eff={met['efficiency_global']:.3f} | cov={met['coverage_km2']:.1f} km² ({met['coverage_km2']/vienna_area_km2:.1%})")
    if met["length_km"] <= syn_df["length_km"].max():
        print(f"  Syn:     {m['length_km']:.1f} km | LCC {m['length_lcc']/1000:.1f} km ({m['lcc_share']:.1%}) | "
              f"{int(m['components'])} comp. | dir={m['directness_lcc']:.3f}  (Q={m['quantile']:.3f})")
    else:
        print(f"  Syn:     (Full network exceeds synthetic maximum of {syn_df['length_km'].max():.0f} km)")
    print(f"  Overlap: {sp['ov_km']:.1f} km ({sp['ov_ratio']:.1%}) | Gap: {sp['gap_km']:.1f} km")
print("\n" + "=" * 70)
print(f"Output files in: {OUT}")
