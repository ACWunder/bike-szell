"""
Compute per-city effective boundaries for coverage and area normalisation.

Strategy: take the exact Nominatim polygon that the bikenwgrowth algorithm
uses (geocode_to_gdf + extract_relevant_polygon + fill_holes), then subtract
large green spaces (forest, woods, parks, protected areas) above an area
threshold. The remaining "effective" polygon represents the urban area
where cycling infrastructure is meaningfully measurable.

Outputs (per city, in analysis_output/{city}/):
  nominatim_boundary.geojson   raw Nominatim polygon (= algorithm's polygon)
  forests.geojson              filtered green spaces that were cut out
  effective_boundary.geojson   nominatim minus forests
  boundary_summary.json        areas in km²

Output (shared):
  analysis_output/comparison/city_boundaries.csv
    columns: city, nominatim_km2, forest_km2, effective_km2

Usage: conda run -n growbikenet python compute_effective_boundaries.py
"""

from pathlib import Path
import json
import time

import geopandas as gpd
import osmnx as ox
import pandas as pd
from shapely.geometry import Polygon, MultiPolygon
from shapely import ops as shapely_ops

# ── Config ──────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent / "bikenwgrowth-data"
OUT_COMP = BASE / "analysis_output" / "comparison"
OUT_COMP.mkdir(parents=True, exist_ok=True)

# Same nominatim strings the bikenwgrowth algorithm uses (parameters/cities.csv)
NOMINATIM = {
    "amsterdam": "amsterdam",
    "barcelona": "Barcelona, Barcelonès, Barcelona, Catalonia, 08001, Spain",
    "berlin":    "Berlin, Germany",
    "oslo":      "oslo",
    "vienna":    "vienna, austria",
}

# OSM tags that count as "green / non-urban" and should be excluded
FOREST_TAGS = {
    "natural":  ["wood"],
    "landuse":  ["forest"],
    "leisure":  ["park", "nature_reserve"],
    "boundary": ["protected_area", "national_park"],
}

# Only cut out green polygons larger than this (km²) — small parks stay
MIN_FOREST_AREA_KM2 = 0.5


# ── Helpers (replicating the algorithm's logic) ─────────────────────────────
def extract_relevant_polygon(mp):
    """Largest polygon of a MultiPolygon, otherwise pass through."""
    if isinstance(mp, Polygon):
        return mp
    return max(mp.geoms, key=lambda p: p.area)


def fill_holes(cov):
    """Fill interior holes of a (Multi)Polygon — matches functions.py logic."""
    holes = []
    if isinstance(cov, MultiPolygon):
        polys = list(cov.geoms)
        for p in polys:
            holes.extend(list(p.interiors))
    elif isinstance(cov, Polygon) and not cov.is_empty:
        polys = [cov]
        holes.extend(list(cov.interiors))
    else:
        return cov
    eps = 1e-8
    return shapely_ops.unary_union(
        polys + [Polygon(h).buffer(eps) for h in holes]
    )


def km2_of(geom, crs_proj):
    """Area in km² (projects to metric CRS first)."""
    gs = gpd.GeoSeries([geom], crs="EPSG:4326").to_crs(crs_proj)
    return float(gs.area.iloc[0]) / 1e6


# ── Main ────────────────────────────────────────────────────────────────────
rows = []
for city, nstr in NOMINATIM.items():
    print(f"\n{'='*60}\n  {city}\n{'='*60}")
    city_dir = BASE / "analysis_output" / city
    city_dir.mkdir(parents=True, exist_ok=True)

    # 1 · Nominatim polygon (= the algorithm's view)
    print(f"  Geocoding '{nstr}' …")
    gdf_nom = ox.geocoder.geocode_to_gdf(nstr)
    poly_nom = extract_relevant_polygon(gdf_nom.geometry.iloc[0])
    poly_nom = fill_holes(poly_nom)

    # Estimate UTM CRS for metric ops
    nom_gs = gpd.GeoSeries([poly_nom], crs="EPSG:4326")
    crs_proj = nom_gs.estimate_utm_crs()
    nom_km2 = km2_of(poly_nom, crs_proj)
    print(f"  Nominatim area: {nom_km2:.1f} km²  (CRS={crs_proj.srs})")

    # Save raw nominatim
    gpd.GeoDataFrame({"name": [city]}, geometry=[poly_nom], crs="EPSG:4326") \
        .to_file(city_dir / "nominatim_boundary.geojson", driver="GeoJSON")

    # 2 · Green / forest features inside that polygon
    print(f"  Fetching green features (this can take a moment)…")
    t0 = time.time()
    try:
        feats = ox.features.features_from_polygon(poly_nom, FOREST_TAGS)
    except Exception as e:
        print(f"  ! features_from_polygon failed: {e}")
        feats = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    print(f"  Retrieved {len(feats)} features in {time.time()-t0:.1f}s")

    # Clip green features to the nominatim polygon, then filter by area
    nom_p = nom_gs.to_crs(crs_proj).iloc[0]
    forest_union_p = None
    forest_km2 = 0.0
    if len(feats):
        feats = feats[feats.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
        if len(feats):
            feats_p = feats.to_crs(crs_proj)
            # Clip every feature to the city polygon (intersection)
            feats_p["geometry"] = feats_p.geometry.intersection(nom_p)
            feats_p = feats_p[~feats_p.geometry.is_empty].copy()
            feats_p["area_km2"] = feats_p.geometry.area / 1e6
            big = feats_p[feats_p["area_km2"] >= MIN_FOREST_AREA_KM2].copy()
            print(f"  Polygonal: {len(feats)},  after clip≥{MIN_FOREST_AREA_KM2} km²: {len(big)}")
            if len(big):
                forest_union_p = big.geometry.unary_union
                # Make absolutely sure the union also lies inside nominatim
                forest_union_p = forest_union_p.intersection(nom_p)
                forest_km2 = forest_union_p.area / 1e6
                big.to_crs("EPSG:4326").to_file(
                    city_dir / "forests.geojson", driver="GeoJSON"
                )

    # 3 · effective = nominatim − forests  (in projected CRS for accuracy)
    if forest_union_p is not None:
        eff_p = nom_p.difference(forest_union_p)
    else:
        eff_p = nom_p
    eff_km2 = eff_p.area / 1e6

    eff_gdf = gpd.GeoDataFrame({"name": [city]},
                                geometry=[eff_p], crs=crs_proj).to_crs("EPSG:4326")
    eff_gdf.to_file(city_dir / "effective_boundary.geojson", driver="GeoJSON")

    print(f"  Forest area    : {forest_km2:8.1f} km²  ({forest_km2/nom_km2:.1%} of nominatim)")
    print(f"  Effective area : {eff_km2:8.1f} km²")

    # 4 · Per-city JSON summary
    summary = {
        "city": city,
        "nominatim_string": nstr,
        "nominatim_km2":   round(nom_km2, 2),
        "forest_km2":      round(forest_km2, 2),
        "effective_km2":   round(eff_km2, 2),
        "forest_threshold_km2": MIN_FOREST_AREA_KM2,
        "crs_proj": crs_proj.srs,
    }
    with open(city_dir / "boundary_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    rows.append(summary)


# ── Shared CSV ──────────────────────────────────────────────────────────────
df = pd.DataFrame(rows)
df = df[["city", "nominatim_km2", "forest_km2", "effective_km2"]]
csv_path = OUT_COMP / "city_boundaries.csv"
df.to_csv(csv_path, index=False)

print(f"\n{'='*60}")
print(df.to_string(index=False))
print(f"\nSaved: {csv_path}")
print("Done.")
