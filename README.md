# Vienna Bicycle Network — Structural Comparison

Interdisciplinary Project · Arthur Wunder · May 2026

Structural comparison of the existing Vienna bicycle network against a
synthetically grown algorithmic optimum
([bikenwgrowth](https://doi.org/10.1038/s41598-022-10783-y), Szell et al. 2022),
plus a cross-city extension covering Amsterdam, Barcelona, Berlin, Oslo and Vienna.

---

## Final reports → `reports/`

| Report | Pages | Scope |
|---|---:|---|
| **`report_multicity.pdf`** | 17 | Five-city comparison against OSM bikeable — effective boundaries, buffer + gridl sensitivity, bootstrap CIs, DBSCAN gap hotspots |
| **`report_vienna_v4.pdf`** | 10 | Vienna deep-dive against the city's Open Government Data (OGD), with cross-city positioning and OGD-based gap hotspots |
| `expose.pdf` | 3 | Original project proposal (Mar 2026) |

---

## Project layout

```
bike/
├── README.md
│
├── pipeline/                                    ← 8 analysis scripts, numbered in run order
│   ├── 01_compute_effective_boundaries.py        Nominatim polygon − forest = effective city area
│   ├── 02_analysis_multicity_spatial.py          overlap / gap / buffer sensitivity, 5 cities
│   ├── 03_analysis_multicity.py                  growth curves + coverage bars
│   ├── 04_analysis_normalized.py                 km/km² density + quantile comparison
│   ├── 05_analysis_directness_bootstrap.py       10 × 1000-pair resampling, IQR per stage
│   ├── 06_analysis_gap_clusters.py               DBSCAN hotspots in the gap edges per city
│   ├── 07_generate_report_multicity.py           assembles the 17-page multicity PDF
│   ├── 08_generate_report_vienna.py              assembles the 10-page Vienna v4 PDF
│   └── run_all.sh                                runs steps 1–8 end-to-end (~10 min)
│
├── vienna_ogd_prep/                             ← Vienna OGD-specific preprocessing
│   ├── analysis_vienna_script.py                 produces gap_C_full.geojson etc. from Radwege_ogd
│   ├── extract_synthetic_geo.py
│   └── visualize_fragmentation.py
│
├── reports/                                     ← final PDFs
│   ├── report_multicity.pdf
│   ├── report_vienna_v4.pdf
│   └── expose.pdf
│
├── data/                                        ← source data (small files)
│   ├── vienna_ogd/                               Vienna OGD CSV (Radwege_ogd.csv, …)
│   └── qgis/                                     all .qgz QGIS project files
│
├── archive/                                     ← old / unused, kept for reference
│   ├── scripts/                                  earlier generate_report variants
│   ├── shell/                                    bikenwgrowth pipeline run-scripts
│   ├── notebooks/                                older Jupyter notebooks
│   └── reports/                                  earlier PDF iterations (v2, v3, drafts)
│
├── bikenwgrowth-data/                       ← LARGE data store (≈3 GB) — do not move
│   ├── data/{city}/                              OSM edges, POI grids (input)
│   ├── results/{city}/                           synthetic growth CSVs (input)
│   │   ├── vienna/      gridl = 1701 m            (multicity)
│   │   └── vienna_2/    gridl =  600 m            (Vienna deep-dive)
│   ├── plots/, plotsnetworks/, videos/, logs/    other bikenwgrowth outputs (kept for reference)
│   └── analysis_output/
│       ├── report_multicity.pdf                  ← final, mirrored into reports/
│       ├── report_vienna_v4.pdf                  ← final, mirrored into reports/
│       ├── {city}/                               per-city multicity output (5 cities)
│       ├── comparison/                           cross-city CSVs + plots
│       └── vienna_ogd/                           Vienna OGD-specific output
│                                                 (Variant A/B/C geojsons, frag_C.png,
│                                                  map_overlap_C.png, synthetic_stages/, …)
│
├── bikenwgrowth-source/                    ← upstream bikenwgrowth pipeline (input source)
│
└── cache/                                       ← osmnx HTTP cache (~60 MB, regenerable)
```

---

## How to reproduce

```bash
conda activate growbikenet
bash pipeline/run_all.sh
```

Total runtime: ~10 min on a 2024-era MacBook.

Each script can also be run on its own — all of them use
`Path(__file__).resolve().parent.parent` as the project root, so they work
from any working directory.

---

## Data sources

### Real networks

* **Vienna OGD** (`data/vienna_ogd/Radwege_ogd.csv`) — 14,777 segments of
  Vienna's open-government cycling infrastructure. Three variants used in
  `report_vienna_v4.pdf`:
  * A — Separated paths (862 km)
  * B — Marked lanes (596 km)
  * C — Full rideable network (1 813 km)  ← only variant used for headline metrics
* **OSM** (`bikenwgrowth-data/data/{city}/`) — `bikeable` and
  `biketrack` edge lists per city, extracted via osmnx in the bikenwgrowth
  pipeline.

### Synthetic networks

Five city runs from the bikenwgrowth pipeline live in
`bikenwgrowth-data/results/`.  Two Vienna runs at different POI-grid
spacings are kept side-by-side, plus one experimental Vienna run that
is **not used** by any current pipeline script:

| `results/` folder | gridl | Syn max | Used in | Status |
|---|---:|---:|---|---|
| `amsterdam/`, `barcelona/`, `berlin/`, `oslo/` | 1 701 m | — | `report_multicity` | active |
| `vienna/` | 1 701 m | 626 km | `report_multicity` (cross-city) | active |
| `vienna_2/` | 600 m | 989 km | `report_vienna_v4` (deep dive) | active |
| `vienna_1/` | — | 640 km | — | unused, kept as a snapshot |

The two active Vienna runs are compared head-to-head on
`report_multicity.pdf` p. 12.  Where each run's downstream outputs live:

| Run | Source CSV | Stage GeoJSONs | Spatial analysis outputs |
|---|---|---|---|
| Vienna (1 701 m) | `results/vienna/` | `analysis_output/vienna/synthetic_stages/` | `analysis_output/vienna/` (against OSM bikeable) |
| Vienna_2 (600 m) | `results/vienna_2/` | `analysis_output/vienna_ogd/synthetic_stages/` | `analysis_output/vienna_ogd/` (against Vienna OGD) |

---

## Output files reference

Every analysis script writes its outputs into one of three places under
`bikenwgrowth-data/analysis_output/`:

### Per-city multicity outputs — `analysis_output/{city}/`

One folder per city (`amsterdam/`, `barcelona/`, `berlin/`, `oslo/`,
`vienna/`).  All five contain the same file set:

```
{city}/
├── nominatim_boundary.geojson         raw Nominatim polygon
├── effective_boundary.geojson         nominatim − large forest/park ≥ 0.5 km²
├── forests.geojson                    what was subtracted
├── boundary_summary.json              area numbers (nominatim_km² / forest_km² / effective_km²)
├── osm_bikeable_{city}.geojson        real cycling network (primary reference)
├── osm_biketrack_{city}.geojson       real cycling network (restrictive: dedicated tracks only)
├── synthetic_{city}.geojson           synthetic edges at the matched length stage
├── overlap_{city}.geojson             syn edges with ≥ 50 % buffer overlap (built)
├── gap_{city}.geojson                 syn edges with  < 50 % buffer overlap (missing)
├── gap_clusters_{city}.geojson        top-5 DBSCAN hotspot polygons over the gap layer
├── map_{city}.png                     synthetic vs bikeable overlay (used in report p.7–11)
├── gap_hotspots_map_{city}.png        DBSCAN hotspot map (used in report p.15)
└── synthetic_stages/                  all 40 growth stages as GeoJSON + index.json
```

Overlap / gap files use a default **15 m buffer**; see
`comparison/buffer_sensitivity.csv` for 10/20 m variants.

### Cross-city CSVs and PNGs — `analysis_output/comparison/`

```
comparison/
├── city_boundaries.csv              nominatim_km² / forest_km² / effective_km² per city
├── summary_table.csv                metrics at synthetic max per city
├── spatial_summary.csv              overlap / gap @ 15 m buffer
├── buffer_sensitivity.csv           overlap / gap at 10, 15, 20 m
├── density_table.csv                bikeable + biketrack + syn km/km²
├── fixed_quantile_metrics.csv       directness/efficiency/LCC at Q=0.25/0.50/0.75/1.0
├── normalized_comparison.csv        density-normalised cross-city headline
├── metrics_at_biketrack.csv         per city: syn metrics at biketrack-length stage
├── directness_bootstrap.csv         10 × 1000-pair resampling, median/Q25/Q75 per stage
├── gap_clusters.csv                 top-5 DBSCAN hotspots × 5 cities (km, edges, direction)
└── *.png                            all cross-city plots embedded in report_multicity.pdf
```

### Vienna OGD deep-dive outputs — `analysis_output/vienna_ogd/`

Variant A/B/C analysis against Vienna's Open Government Data network,
using the gridl = 600 m synthetic run:

```
vienna_ogd/
├── vienna_boundary.geojson                              Vienna city polygon (from OGD)
├── real_{A_separated, B_marked, C_full}.geojson         OGD real network per variant
├── synthetic_{A_separated, B_marked, C_full}.geojson    synthetic stage matched to that variant's length
├── overlap_{A_separated, B_marked, C_full}.geojson      ≥ 50 % buffer overlap
├── gap_{A_separated, B_marked, C_full}.geojson          < 50 % buffer overlap (e.g. gap_C_full.geojson = 429 km used in v4 p.9)
├── frag_C.png, map_overlap_C.png                        figures used in report_vienna_v4.pdf
├── fragmentation_map_en.png, growth_curves_en.png, maps_en.png   older legacy figures
├── comparison_table_en.csv                              legacy table
└── synthetic_stages/                                    40 growth stages of the gridl = 600 m run
```

**What the variant suffixes mean:**

* `_A_separated` — **Variant A** (Separated infrastructure): physically
  separated cycle paths only, 862 km. Highly fragmented.
* `_B_marked` — **Variant B** (Marked lanes): painted lanes on streets,
  596 km. Even more fragmented.
* `_C_full` — **Variant C** (Full network): A + B + every other rideable
  bit, 1 813 km. The only variant with usable connectivity, and the
  one used for all headline metrics in `report_vienna_v4.pdf`.

---

## Environment

* **Conda env**: `growbikenet`
* Key packages: osmnx 1.9.4, shapely 2.0.7, geopandas 0.14.4,
  networkx, scikit-learn, matplotlib, pandas
* CRS conventions:
  * EPSG:4326 (lon/lat) for I/O
  * `gdf.estimate_utm_crs()` for metric ops in the multicity scripts
  * EPSG:31256 (MGI Austria GK East) for Vienna OGD analysis

---

## Key methodological choices

Inline in each script and on the Methodology pages of the reports:

* **bikeable** as primary reference, not **biketrack** — biketrack alone is
  too restrictive in cities that rely on marked lanes (Vienna, Berlin) and
  unfairly penalises them.
* **Effective city area** = Nominatim polygon (the one the bikenwgrowth
  algorithm itself uses, via `osmnx.geocoder.geocode_to_gdf`)
  *minus* large forest / park polygons ≥ 0.5 km². Critical for Oslo
  (Marka cuts 480 km² → 168 km²) and Vienna (Wienerwald cuts 415 → 325).
* **15 m buffer** for spatial-overlap classification, with explicit
  10 m / 20 m sensitivity reported on `report_multicity.pdf` p. 4.
* **gridl = 1701 m** for the multicity comparison (uniform across all
  five cities); Vienna's deep-dive uses gridl = 600 m. Both are compared
  side-by-side on `report_multicity.pdf` p. 12.
* **No historic / cultural speculation** — only explanations derivable
  from quantities already in the report (area, density, directness,
  efficiency, overlap).

---

## Reproducing the headline numbers

| Number | Where it appears | How to verify |
|---|---|---|
| Vienna Variant C: 1 813 km, GE 0.557 | vienna v4 p. 2 | `vienna_ogd_prep/analysis_vienna_script.py` |
| Vienna effective area: 325 km² | both reports | `pipeline/01_compute_effective_boundaries.py` |
| Overlap Berlin vs OSM bikeable: 75.2 % | multicity p. 12, 13 | `pipeline/02_analysis_multicity_spatial.py` |
| Bootstrap IQR Berlin directness: ±0.0024 | multicity p. 13 | `pipeline/05_analysis_directness_bootstrap.py` |
| Vienna OGD gap hotspot #1: 49 km south | vienna v4 p. 9 | DBSCAN block inside `pipeline/08_generate_report_vienna.py` |
| Vienna OSM bikeable gap hotspot #1: 28 km SW | multicity p. 14 | `pipeline/06_analysis_gap_clusters.py` |

---

## Limitations

Stated explicitly in both reports:

* The synthetic algorithm runs on the car road graph and is unaware of
  terrain, social barriers, or cycling-policy constraints.
* Spatial overlap depends on the buffer threshold; the sensitivity sweep
  shows the city ordering is preserved across 10 / 15 / 20 m so the
  15 m headline is robust.
* `gridl` is one POI-sampling choice. Vienna shows a +57 % length swing
  between gridl = 600 m and gridl = 1701 m; directness moves by only
  0.015, so the cross-city ordinal ranking does not flip but absolute
  lengths cannot be compared across grids.
* Vienna OGD and OSM bikeable are not the same network. Headline numbers
  for Vienna therefore differ between the two reports. Both are correct
  for what they measure.
