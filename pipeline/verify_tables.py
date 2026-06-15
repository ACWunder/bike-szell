"""
verify_tables.py — Audit every number that appears in report_multicity.pdf.

The report (07_generate_report_multicity.py) renders all tables straight from
CSV files; nothing is typed by hand. This script re-reads the SAME source CSVs,
prints each table the way it appears in the PDF, and — crucially — cross-checks
the values that are supposed to be identical across pages but are computed by
different scripts (e.g. "OSM bikeable km").

Run:   conda run -n growbikenet python pipeline/verify_tables.py
       (plain `python pipeline/verify_tables.py` also works if pandas is on PATH)

Read the printout next to the PDF: if a number here differs from the PDF, the
PDF is stale (re-run the generator). If two SOURCES disagree (flagged with
!!!), that is a real pipeline inconsistency to fix in the analysis scripts.
"""

from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent.parent / "bikenwgrowth-data"
COMP = BASE / "analysis_output" / "comparison"
CITIES = ["amsterdam", "barcelona", "berlin", "oslo", "vienna"]
LABEL = {c: c.capitalize() for c in CITIES}


def rule(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ── Raw source 1: per-city existing.csv (used by the p2 + p6 tables) ──────────
def existing_lengths():
    """biketrack / bikeable length in km, straight from each city_existing.csv."""
    out = {}
    for c in CITIES:
        df = pd.read_csv(BASE / "results" / c / f"{c}_existing.csv")
        bt = df.loc[df.network == "biketrack", "length"].iloc[0] / 1000
        bk = df.loc[df.network == "bikeable", "length"].iloc[0] / 1000
        out[c] = (bt, bk)
    return out


# ── Raw source 2: per-city synthetic growth table ────────────────────────────
def synthetic_max():
    """Final synthetic length (km) and number of growth steps per city."""
    out = {}
    for c in CITIES:
        df = pd.read_csv(BASE / "results" / c / f"{c}_poi_grid_betweenness.csv")
        out[c] = (df["length"].max() / 1000, len(df))
    return out


def show_csv(name, path):
    rule(name + f"   ({path.relative_to(BASE.parent)})")
    if not path.exists():
        print("  [missing]")
        return None
    df = pd.read_csv(path)
    with pd.option_context("display.width", 200,
                           "display.max_columns", None):
        print(df.to_string(index=False))
    return df


def main():
    ex = existing_lengths()
    syn = synthetic_max()

    # ---- PAGE 2: City overview (biketrack/bikeable from existing.csv) --------
    rule("PAGE 2 — City overview  (source: results/<city>/<city>_existing.csv)")
    print(f"{'City':<11}{'biketrack km':>14}{'bikeable km':>14}"
          f"{'Syn max km':>13}{'Steps':>7}")
    for c in CITIES:
        bt, bk = ex[c]
        sm, steps = syn[c]
        print(f"{LABEL[c]:<11}{bt:>14.0f}{bk:>14.0f}{sm:>13.0f}{steps:>7}")

    # ---- Pages that read pre-aggregated comparison CSVs ----------------------
    spatial = show_csv("PAGES 7–10 + 16 — spatial_summary",
                       COMP / "spatial_summary.csv")
    density = show_csv("PAGE 6 — density_table", COMP / "density_table.csv")
    show_csv("PAGE 16 — summary_table", COMP / "summary_table.csv")
    show_csv("PAGE 4 — buffer_sensitivity", COMP / "buffer_sensitivity.csv")
    show_csv("PAGE 5 — fixed_quantile_metrics", COMP / "fixed_quantile_metrics.csv")
    show_csv("PAGES 14–15 — gap_clusters", COMP / "gap_clusters.csv")

    # ---- CROSS-SOURCE CONSISTENCY CHECK -------------------------------------
    rule("CONSISTENCY CHECK — does 'OSM bikeable km' agree across sources?")
    print("existing.csv  → p2 overview + p6 density   |   "
          "spatial_summary.csv → p7–10 maps + p16\n")
    print(f"{'City':<11}{'existing.csv':>14}{'spatial_summary':>18}"
          f"{'density_table':>16}{'':>6}")
    bad = []
    for c in CITIES:
        _, bk_ex = ex[c]
        bk_sp = bk_de = float("nan")
        if spatial is not None:
            row = spatial[spatial["City"] == LABEL[c]]
            if len(row):
                bk_sp = float(row["OSM bikeable (km)"].iloc[0])
        if density is not None and "bikeable (km)" in density.columns:
            row = density[density["City"] == LABEL[c]]
            if len(row):
                bk_de = float(row["bikeable (km)"].iloc[0])
        # mismatch if existing vs spatial differ by > 2 km
        flag = ""
        if abs(bk_ex - bk_sp) > 2:
            flag = "  !!! MISMATCH"
            bad.append(c)
        print(f"{LABEL[c]:<11}{bk_ex:>14.0f}{bk_sp:>18.0f}{bk_de:>16.0f}{flag}")

    print()
    if bad:
        print(f"⚠  {len(bad)} cities disagree: {', '.join(LABEL[c] for c in bad)}.")
        print("   The overview/density tables use existing.csv (GrowBikeNet "
              "results table);")
        print("   the per-city maps use spatial_summary.csv (02_analysis_"
              "multicity_spatial.py,")
        print("   which re-measures bikeable length from the *_biketrackcarall "
              "graph).")
        print("   → Pick ONE definition of 'bikeable length' and use it "
              "everywhere, then")
        print("     re-run the spatial analysis + report generator.")
    else:
        print("✓  All sources agree on bikeable length.")


if __name__ == "__main__":
    main()
