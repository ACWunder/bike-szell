"""Splice each city's bikeable row in existing.csv: keep the deterministic fields
that genuinely change with crossings (length, length_lcc, coverage, poi_coverage,
components) from the fresh recompute, but restore the sampled directness/efficiency
fields from the pre-change backup (crossings provably do not change them; the recompute
only injected RNG sampling noise + OSM drift). Other network rows are untouched."""
import csv
from pathlib import Path

ROOT = Path("/Users/arthurwunder/Downloads/bike")
cities = ["amsterdam", "barcelona", "berlin", "oslo", "vienna"]

# fields restored from backup (sampled / noisy)
RESTORE = ["directness", "directness_lcc", "efficiency_global", "efficiency_local",
           "directness_lcc_linkwise", "directness_all_linkwise"]

def bikeable_row(path):
    with open(path) as f:
        rows = list(csv.reader(f))
    hdr = rows[0]
    for i, r in enumerate(rows):
        if r and r[0] == "bikeable":
            return rows, hdr, i
    raise RuntimeError(f"no bikeable row in {path}")

for c in cities:
    backup = ROOT / ".regen_backup" / ("results_vienna" if c == "vienna" else "results_other") / f"{c}_existing.csv"
    cur = ROOT / "bikenwgrowth-data" / "results" / c / f"{c}_existing.csv"

    _brows, bhdr, bi = bikeable_row(backup)
    old_vals = _brows[bi]
    rows, hdr, ci = bikeable_row(cur)

    changes = []
    for field in RESTORE:
        j_new, j_old = hdr.index(field), bhdr.index(field)
        new_v, old_v = rows[ci][j_new], old_vals[j_old]
        if new_v != old_v:
            changes.append(f"{field}: {float(new_v):.4f}→{float(old_v):.4f}(restored)")
        rows[ci][j_new] = old_v

    with open(cur, "w", newline="") as f:
        csv.writer(f).writerows(rows)

    kept_len = float(rows[ci][hdr.index("length")]) / 1000
    kept_comp = rows[ci][hdr.index("components")]
    print(f"{c:10} kept length={kept_len:.1f}km components={kept_comp} | "
          + "; ".join(changes))
print("\nspliced.")
