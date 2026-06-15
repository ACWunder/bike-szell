#!/usr/bin/env python3
"""
Extrahiert alle synthetischen Wachstumsstufen aus dem Pickle einer Stadt
und speichert sie als GeoJSON in bikenwgrowth-data/analysis_output/{city}/synthetic_stages/

Verwendung:
    conda run -n growbikenet python extract_synthetic_geo.py vienna
    conda run -n growbikenet python extract_synthetic_geo.py amsterdam
    conda run -n growbikenet python extract_synthetic_geo.py copenhagen
    conda run -n growbikenet python extract_synthetic_geo.py berlin
    conda run -n growbikenet python extract_synthetic_geo.py barcelona

Ohne Argument: alle Städte in bikenwgrowth-data/results/ werden verarbeitet.
"""
import sys
from pathlib import Path
import pickle
import json

RESULTS_BASE = Path(__file__).resolve().parent.parent / "bikenwgrowth-data" / "results"
OUT_BASE     = Path(__file__).resolve().parent.parent / "bikenwgrowth-data" / "analysis_output"


def extract_city(city: str):
    # Results folder: prefer plain city name, fall back to city_2, city_1
    candidates = [RESULTS_BASE / city] + sorted(RESULTS_BASE.glob(f"{city}_*"), reverse=True)
    result_dir = next((p for p in candidates if p.is_dir()), None)
    if result_dir is None:
        print(f"[{city}] Kein results-Ordner gefunden – überspringe.")
        return

    pickle_path = result_dir / f"{city}_poi_grid_betweenness.pickle"
    if not pickle_path.exists():
        print(f"[{city}] Pickle nicht gefunden: {pickle_path} – überspringe.")
        return

    out_dir = OUT_BASE / city / "synthetic_stages"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[{city}] Lade Pickle: {pickle_path}")
    with open(pickle_path, "rb") as f:
        pkl = pickle.load(f)

    quantiles = pkl["prune_quantiles"]
    graphs    = pkl["GTs"]
    print(f"[{city}] {len(graphs)} Wachstumsstufen gefunden.")

    for i, (q, G_ig) in enumerate(zip(quantiles, graphs)):
        xs = G_ig.vs["x"]
        ys = G_ig.vs["y"]   # negated in bikenwgrowth → negate back

        features = []
        for e in G_ig.es:
            u, v = e.source, e.target
            lon_u, lat_u = xs[u], -ys[u]
            lon_v, lat_v = xs[v], -ys[v]
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[float(lon_u), float(lat_u)], [float(lon_v), float(lat_v)]]
                },
                "properties": {
                    "weight_m": float(e["weight"]),
                    "osmid": int(e["osmid"]) if e["osmid"] is not None else None,
                    "quantile": float(q),
                    "stage": int(i)
                }
            })

        total_len_km = sum(e["weight"] for e in G_ig.es) / 1000
        fname = out_dir / f"stage_{i:02d}_q{q:.3f}_{total_len_km:.0f}km.geojson"
        with open(fname, "w") as f:
            json.dump({"type": "FeatureCollection", "features": features}, f)

        print(f"  [{i:02d}] Quantil {q:.3f} → {total_len_km:.1f} km, {len(features)} Kanten")

    index = [
        {
            "stage": i,
            "quantile": q,
            "length_km": round(sum(G.es["weight"]) / 1000, 2),
            "n_edges": G.ecount(),
            "n_nodes": G.vcount(),
            "file": f"stage_{i:02d}_q{q:.3f}_{sum(G.es['weight'])/1000:.0f}km.geojson"
        }
        for i, (q, G) in enumerate(zip(quantiles, graphs))
    ]
    with open(out_dir / "index.json", "w") as f:
        json.dump(index, f, indent=2)

    print(f"[{city}] Fertig → {out_dir}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        cities = sys.argv[1:]
    else:
        # Auto-detect all cities with results
        cities = [p.name for p in RESULTS_BASE.iterdir() if p.is_dir()]
        # Deduplicate city_1/city_2 variants → keep base name
        base_cities = sorted({c.split("_")[0] for c in cities})
        cities = base_cities
        print(f"Keine Stadt angegeben – verarbeite alle: {cities}")

    for city in cities:
        extract_city(city)
