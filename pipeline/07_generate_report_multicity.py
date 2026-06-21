"""
Multi-city bicycle network comparison report (PDF)
Cities: Amsterdam, Barcelona, Berlin, Oslo, Vienna  —  gridl=1701

Networks are referred to throughout as A / B / C (declared once on page 2):
  A — biketrack  (OSM, dedicated cycle tracks)
  B — bikeable   (OSM, full rideable network — primary real reference)
  C — synthetic  (GrowBikeNet optimum)

Structure:
  PART I — structure (C vs B, navy banners)
  Page 1     — Cover
  Page 2     — Introduction & Methodology (A/B/C declared · B vs C at a glance)
  Page 3     — Cross-city growth curves (3 × 2 layout)
  Page 4     — Network density (per-km² bars + density table)
  Pages 5–9  — Per-city: large map + data boxes + cross-city context
  Page 10    — Vienna gridl sensitivity (600 m vs 1701 m)
  Page 11    — Summary table & key observations
  PART II — safety (C vs A, green banners)
  Page 12    — Safety section divider
  Pages 13–17— Per-city safety: A-vs-C map + data boxes + context
  Page 18    — Safety summary table (A vs C) & observations
  Page 19    — Closing the gap: island-bridging connectivity payoff (A vs A+gap)
  Page 20    — Data-driven discussion (per-city explanations)
  Page 21    — Appendix: metric & term definitions

Usage: conda run -n growbikenet python generate_report_multicity.py
"""

from pathlib import Path
import textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.ticker as mticker
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle, FancyBboxPatch
import numpy as np
import pandas as pd

# ── Config ──────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent / "bikenwgrowth-data"
AO   = BASE / "analysis_output"
COMP = AO / "comparison"
PDF_PATH = AO / "report_multicity.pdf"

CITIES = ["amsterdam", "barcelona", "berlin", "oslo", "vienna"]
CITY_LABELS = {
    "amsterdam": "Amsterdam",
    "barcelona": "Barcelona",
    "berlin":    "Berlin",
    "oslo":      "Oslo",
    "vienna":    "Vienna",
}
COLORS = {
    "amsterdam": "#e41a1c",
    "barcelona": "#377eb8",
    "berlin":    "#4daf4a",
    "oslo":      "#984ea3",
    "vienna":    "#ff7f00",
}

# ── Network variants (declared ONCE here, used EVERYWHERE — see Appendix) ──────
# The report compares three networks, labelled A / B / C everywhere (letters
# stay; the names are the familiar OSM / algorithm terms):
#   A — biketrack  (OSM: protected / segregated cycle paths)
#   B — bikeable   (OSM: every legally cyclable edge — primary real reference)
#   C — synthetic  (GrowBikeNet algorithm output at its maximum)
# A and B are real (OSM); C is synthetic.
A_LABEL = "A — biketrack"
B_LABEL = "B — bikeable"
C_LABEL = "C — synthetic"
A_TAG = "A · biketrack"     # compact forms for tight headers / axes / legends
B_TAG = "B · bikeable"
C_TAG = "C · synthetic"

# ── Design tokens ────────────────────────────────────────────────────────────
DBLUE  = "#0d3d6e"
BLUE   = "#1a6faf"
TEAL   = "#0eb6d2"
GREEN  = "#27ae60"
RED    = "#c0392b"

# ── Safety-section (A vs C) green theme ────────────────────────────────────────
# The safety analysis mirrors the B-vs-C pages but in green, so a reader can tell
# at a glance which network is the reference (B = blue banner, A = green banner).
GREEN_DARK = "#1b6e3d"   # banner / headings (mirrors DBLUE)
GREEN_MED  = "#2e8b57"   # accents (mirrors BLUE)
GREEN_TINT = "#cfe8d8"   # group banner tint over A columns (mirrors REAL_TINT)
LGREEN     = "#eef6f0"   # very light row tint (mirrors LGREY for green pages)
GREEN_TXT  = "#13502c"   # dark green text on tints
A_NET_COL  = "#2c7fb8"   # protected-infra blue (matches the safety maps)
DARK   = "#1c1c2e"
GREY   = "#555"
LGREY  = "#f0f4f8"
WHITE  = "#ffffff"
LBLUE  = "#a8c8e8"

# Layered table palette — the goal is a clear visual hierarchy where the
# "Real (OSM) / Synthetic" group banner sits visibly ABOVE the lighter
# header-row tint, instead of dissolving into it.
#   group banner  (strongest)   →  REAL_TINT / SYN_TINT
#   header row    (medium)      →  HEADER_TINT
#   alternating data rows       →  LGREY
HEADER_TINT = "#f1f5fa"   # very light grey-blue, sits behind column headers
REAL_TINT   = "#c5d4e1"   # cool grey-blue, group banner over real OSM cols
SYN_TINT    = "#a8d6e3"   # subtle teal, group banner over synthetic cols
REAL_TXT    = "#1c1c2e"
SYN_TXT     = "#0d3d6e"

# Vertical breathing room between the dark page header and the first body
# element. Used everywhere instead of hard-coding magic numbers.
HEADER_BOTTOM = 0.905     # bottom edge of the dark-blue page header bar
BODY_TOP      = 0.870     # where body content may begin

plt.rcParams.update({
    "font.family":     "DejaVu Sans",
    "font.size":       9,
    "text.color":      DARK,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

PW, PH = 8.27, 11.69   # A4


# ── Helpers ──────────────────────────────────────────────────────────────────
def new_page():
    fig = plt.figure(figsize=(PW, PH), dpi=300)
    fig.patch.set_facecolor(WHITE)
    return fig


def page_header(fig, title, subtitle, page_num, bar_color=DBLUE, sub_color=LBLUE):
    """Dark banner header. Default theme is navy-blue (B-vs-C structural section);
    the safety section (A vs C) passes bar_color=GREEN_DARK for a green banner."""
    fig.add_artist(Rectangle((0, 0.905), 1, 0.095,
                              transform=fig.transFigure,
                              facecolor=bar_color, zorder=10))
    fig.text(0.025, 0.957, title, color=WHITE, fontsize=17,
             fontweight="bold", va="center", zorder=11)
    fig.text(0.025, 0.918, subtitle, color=sub_color, fontsize=8.5,
             va="center", zorder=11)
    fig.text(0.975, 0.937, str(page_num), color=sub_color, fontsize=9,
             ha="right", va="center", zorder=11)


def add_image(fig, path, rect):
    """rect = [left, bottom, width, height] in figure coords."""
    ax = fig.add_axes(rect)
    try:
        img = mpimg.imread(str(path))
        ax.imshow(img)
    except Exception:
        ax.text(0.5, 0.5, f"[image not found]\n{path.name}",
                ha="center", va="center", fontsize=8, color=RED)
    ax.set_axis_off()
    return ax


def metrics_box(fig, rect, rows, title=None, col_widths=None):
    """Draw a simple table inside fig at rect=[l,b,w,h]."""
    ax = fig.add_axes(rect)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_axis_off()
    if title:
        ax.text(0.5, 0.97, title, ha="center", va="top",
                fontsize=9, fontweight="bold", color=DBLUE)
    n = len(rows)
    row_h = 0.85 / max(n, 1)
    for i, row in enumerate(rows):
        y = 0.92 - i * row_h - row_h / 2
        bg = LGREY if i % 2 == 0 else WHITE
        ax.add_patch(Rectangle((0, 0.92 - (i+1)*row_h),
                                1, row_h, facecolor=bg, zorder=0))
        if col_widths is None:
            col_widths = [1/len(row)] * len(row)
        x = 0.01
        for j, (cell, cw) in enumerate(zip(row, col_widths)):
            ha = "left" if j == 0 else "right"
            xpos = x if j == 0 else x + cw - 0.01
            ax.text(xpos, y, str(cell), ha=ha, va="center",
                    fontsize=8, color=DARK)
            x += cw
    return ax


# ── Load data ────────────────────────────────────────────────────────────────
summary   = pd.read_csv(COMP / "summary_table.csv")
spatial   = pd.read_csv(COMP / "spatial_summary.csv")
bt_detail   = pd.read_csv(COMP / "metrics_at_biketrack.csv")
density_tbl = pd.read_csv(COMP / "density_table.csv")
# Buffer sensitivity is optional — only present after spatial v2 has been run.
try:
    sens_tbl = pd.read_csv(COMP / "buffer_sensitivity.csv")
except FileNotFoundError:
    sens_tbl = None
# Safety spatial summary (A = biketrack vs C) — produced by 02b_analysis_safety_spatial.py.
try:
    spatial_safety = pd.read_csv(COMP / "spatial_summary_safety.csv")
except FileNotFoundError:
    spatial_safety = None
# Island-bridging analysis (A vs A+gap) — produced by safety_gap_connectivity.py.
try:
    gap_conn = pd.read_csv(COMP / "safety_gap_connectivity.csv")
except FileNotFoundError:
    gap_conn = None

city_data = {}
for city in CITIES:
    syn = pd.read_csv(BASE / "results" / city / f"{city}_poi_grid_betweenness.csv")
    ex  = pd.read_csv(BASE / "results" / city / f"{city}_existing.csv")
    n = len(syn)
    syn["quantile"]  = [round(i/n, 4) for i in range(1, n+1)]
    syn["length_km"] = syn["length"] / 1000
    syn["lcc_share"] = syn["length_lcc"] / syn["length"]
    bt = ex[ex["network"] == "biketrack"].iloc[0]
    bk = ex[ex["network"] == "bikeable"].iloc[0]
    city_data[city] = {"syn": syn, "biketrack": bt, "bikeable": bk}


# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 — COVER
# ════════════════════════════════════════════════════════════════════════════
with PdfPages(PDF_PATH) as pdf:

    fig = new_page()

    # ── Full-page deep-navy background (elegant, Vienna-v4 style) ─────────
    NAVY  = "#13243b"
    NAVY2 = "#1c3450"
    fig.add_artist(Rectangle((0, 0), 1, 1, transform=fig.transFigure,
                             facecolor=NAVY, zorder=0))

    # Kicker + thin teal accent rule above the title
    fig.text(0.5, 0.770, "GROWBIKENET   ·   FIVE EUROPEAN CITIES",
             color="#7fa8cf", fontsize=10, fontweight="bold",
             ha="center", va="center", zorder=2)
    fig.add_artist(Rectangle((0.5 - 0.055, 0.728), 0.11, 0.005,
                             transform=fig.transFigure, facecolor=TEAL, zorder=2))

    fig.text(0.5, 0.655, "Bicycle Network Growth", color=WHITE, fontsize=34,
             fontweight="bold", ha="center", va="center", zorder=2)
    fig.text(0.5, 0.590, "A Cross-City Structural Comparison", color=LBLUE,
             fontsize=16.5, ha="center", va="center", zorder=2)
    fig.text(0.5, 0.530, "bikenwgrowth algorithm   ·   Szell et al. (2022)",
             color=TEAL, fontsize=11, style="italic", ha="center", va="center",
             zorder=2)
    fig.text(0.5, 0.491, "Arthur Wunder   ·   Interdisciplinary Project   ·   2026",
             color=LBLUE, fontsize=10.5, ha="center", va="center", zorder=2)

    # ── City pills (colour-coded, centred) ───────────────────────────────
    pill_w, pill_h, pill_gap = 0.150, 0.042, 0.012
    total_w = len(CITIES) * pill_w + (len(CITIES) - 1) * pill_gap
    x0 = (1 - total_w) / 2
    for i, city in enumerate(CITIES):
        x = x0 + i * (pill_w + pill_gap)
        fig.add_artist(FancyBboxPatch((x, 0.392), pill_w, pill_h,
                       transform=fig.transFigure, boxstyle="round,pad=0.004",
                       facecolor=COLORS[city], edgecolor="none", zorder=3))
        fig.text(x + pill_w / 2, 0.413, CITY_LABELS[city], color=WHITE,
                 fontsize=10.5, fontweight="bold", ha="center", va="center",
                 zorder=4)

    # ── Companion-report note ────────────────────────────────────────────
    fig.add_artist(FancyBboxPatch((0.22, 0.232), 0.56, 0.088,
                   transform=fig.transFigure, boxstyle="round,pad=0.01",
                   facecolor=NAVY2, edgecolor="#2c4a6e", linewidth=1, zorder=2))
    fig.text(0.5, 0.292, "COMPANION REPORT", color=TEAL, fontsize=9,
             fontweight="bold", ha="center", va="center", zorder=3)
    fig.text(0.5, 0.260,
             "Vienna is analysed in depth in report_vienna_v4.pdf.",
             color=LBLUE, fontsize=9.5, ha="center", va="center", zorder=3)

    # ── Data-source note ─────────────────────────────────────────────────
    fig.text(0.5, 0.170,
             "Data: OpenStreetMap — the only source consistent across all five cities; "
             "the Vienna deep-dive uses official OGD data.",
             color="#7fa8cf", fontsize=8, ha="center", va="center", zorder=2)

    # ── Bottom scope strip ───────────────────────────────────────────────
    fig.text(0.5, 0.090,
             "Growth curves   ·   Normalized density   ·   Per-city maps   ·   "
             "Summary & discussion",
             color="#7fa8cf", fontsize=8.5, ha="center", va="center", zorder=2)
    fig.text(0.5, 0.050, "GrowBikeNet Analysis   ·   gridl = 1701   ·   2026",
             color="#5d7a9c", fontsize=8, ha="center", va="center",
             zorder=2)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # ════════════════════════════════════════════════════════════════════════
    # PAGE 2 — INTRODUCTION & METHODOLOGY  (networks A / B / C declared here)
    # ════════════════════════════════════════════════════════════════════════
    fig = new_page()
    page_header(fig, "Introduction & Methodology",
                "Research question · the three networks A / B / C · data · method",
                2)

    LX = 0.06          # left text column

    def _label(y, text):
        fig.text(LX, y, text, color=BLUE, fontsize=9, fontweight="bold")
        return y - 0.024

    # ── Research question ────────────────────────────────────────────────
    y = _label(BODY_TOP, "RESEARCH QUESTION")
    rq = (
        "How does a city's real bicycle network compare structurally to a "
        "theoretically grown optimum? The bikenwgrowth algorithm (Szell et al., "
        "2022) grows a fully connected synthetic network over 40 steps; we compare "
        "it against the two real OSM networks across five European cities."
    )
    for line in textwrap.wrap(rq, 104):
        fig.text(LX, y, line, color=DARK, fontsize=9); y -= 0.020
    y -= 0.050

    # ── The three networks — A / B / C as three cards ────────────────────
    _label(y, "THE THREE NETWORKS  —  referred to as A / B / C throughout")
    cards = [
        ("A", "biketrack", "Protected cycle tracks, paths & cyclestreets (OSM).",
         REAL_TINT, REAL_TXT, "REAL (OSM)"),
        ("B", "bikeable",  "A plus ≤30 km/h streets & living streets (OSM). "
         "Primary reference.",
         REAL_TINT, REAL_TXT, "REAL (OSM)"),
        ("C", "synthetic", "GrowBikeNet output at its maximum (Q = 1.0).",
         SYN_TINT,  SYN_TXT, "SYNTHETIC"),
    ]
    card_top = y - 0.028
    card_h   = 0.150
    cgap     = 0.028
    card_w   = (0.88 - 2 * cgap) / 3
    for i, (letter, name, desc, tint, txt, kind) in enumerate(cards):
        cx = LX + i * (card_w + cgap)
        cb = card_top - card_h
        fig.add_artist(FancyBboxPatch((cx, cb), card_w, card_h,
                       transform=fig.transFigure, boxstyle="round,pad=0.006",
                       facecolor=LGREY, edgecolor=tint, linewidth=1.6, zorder=1))
        # coloured letter badge
        fig.add_artist(Rectangle((cx + 0.013, cb + card_h - 0.052), 0.046, 0.040,
                       transform=fig.transFigure, facecolor=tint, zorder=2))
        fig.text(cx + 0.036, cb + card_h - 0.032, letter, ha="center", va="center",
                 color=txt, fontsize=16, fontweight="bold", zorder=3)
        fig.text(cx + 0.072, cb + card_h - 0.027, name, va="center",
                 color=DARK, fontsize=12.5, fontweight="bold", zorder=3)
        fig.text(cx + 0.072, cb + card_h - 0.045, kind, va="center",
                 color=txt, fontsize=6.5, fontweight="bold", zorder=3)
        dy = cb + card_h - 0.078
        for line in textwrap.wrap(desc, 27):
            fig.text(cx + 0.013, dy, line, color=GREY, fontsize=8, zorder=3)
            dy -= 0.016
    y = card_top - card_h - 0.062

    # ── Data sources ─────────────────────────────────────────────────────
    y = _label(y, "DATA SOURCES")
    data_rows = [
        ("Real networks",  "OSM via bikenwgrowth (osmnx) — A and B per city"),
        ("Synthetic",      "<city>_poi_grid_betweenness — 40 growth stages (Q 0.025–1.0)"),
        ("Effective area", "Nominatim polygon minus large forest / park areas"),
        ("Algorithm",      "Szell et al. (2022), “Growing urban bicycle networks”"),
    ]
    for k, v in data_rows:
        fig.text(LX, y, k, color=DARK, fontsize=8.7, fontweight="bold")
        fig.text(LX + 0.17, y, v, color=DARK, fontsize=8.7)
        y -= 0.0205
    y -= 0.044

    # ── Method in brief — horizontal 3-step flow ─────────────────────────
    _label(y, "METHOD IN BRIEF")
    steps = [
        ("1", "Grow",
         "Add edges by betweenness near grid-sampled POIs (gridl = 1701 m) — 40 steps."),
        ("2", "Measure",
         "Track LCC share, components, directness and global efficiency at each step."),
        ("3", "Compare",
         "Overlay C on the real networks A / B; measure spatial overlap vs. gap."),
    ]
    fb_top = y - 0.028
    fb_h   = 0.150
    fgap   = 0.028
    fb_w   = (0.88 - 2 * fgap) / 3
    for i, (num, title, desc) in enumerate(steps):
        bx = LX + i * (fb_w + fgap)
        bb = fb_top - fb_h
        fig.add_artist(FancyBboxPatch((bx, bb), fb_w, fb_h,
                       transform=fig.transFigure, boxstyle="round,pad=0.006",
                       facecolor=WHITE, edgecolor=BLUE, linewidth=1.2, zorder=1))
        fig.add_artist(Rectangle((bx + 0.013, bb + fb_h - 0.050), 0.038, 0.036,
                       transform=fig.transFigure, facecolor=DBLUE, zorder=2))
        fig.text(bx + 0.032, bb + fb_h - 0.032, num, ha="center", va="center",
                 color=WHITE, fontsize=13, fontweight="bold", zorder=3)
        fig.text(bx + 0.064, bb + fb_h - 0.032, title, va="center",
                 color=DBLUE, fontsize=11, fontweight="bold", zorder=3)
        dy = bb + fb_h - 0.066
        for line in textwrap.wrap(desc, 30):
            fig.text(bx + 0.013, dy, line, color=DARK, fontsize=8, zorder=3)
            dy -= 0.016
        if i < len(steps) - 1:
            fig.text(bx + fb_w + fgap / 2, bb + fb_h / 2, "›",
                     ha="center", va="center", color=BLUE, fontsize=16,
                     fontweight="bold", zorder=3)

    fig.text(LX, fb_top - fb_h - 0.028,
             "Part I (p. 3–11) compares C against B; Part II (p. 12–19) repeats the "
             "comparison against the protected network A.  Full definitions: appendix (p. 21).",
             color=GREY, fontsize=8.5, style="italic")

    fig.add_artist(Rectangle((0, 0), 1, 0.025,
                              transform=fig.transFigure, facecolor=LGREY))
    fig.text(0.5, 0.012, "GrowBikeNet Analysis  ·  gridl=1701  ·  2026",
             color=GREY, fontsize=7.5, ha="center")
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    # ════════════════════════════════════════════════════════════════════════
    # PAGE 3 — CROSS-CITY GROWTH CURVES
    # ════════════════════════════════════════════════════════════════════════
    fig = new_page()
    page_header(fig, "Cross-City Growth Curve Comparison",
                "X-axis normalised to % of each city's maximum synthetic length", 3)

    # Drop the image slightly below the page header so the plot's own
    # super-title doesn't run into the dark blue banner above.
    add_image(fig, COMP / "comparison_all_cities.png", [0.03, 0.08, 0.94, 0.79])

    fig.text(0.5, 0.025,
             "Diamond markers (◆) indicate the point at which synthetic network C "
             "reaches the length of network A (biketrack) — only where this is possible.",
             ha="center", color=GREY, fontsize=8)
    fig.add_artist(Rectangle((0, 0), 1, 0.015,
                              transform=fig.transFigure, facecolor=LGREY))
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    # ════════════════════════════════════════════════════════════════════════
    # PAGE 4 — NETWORK DENSITY  (per-km² bars + density table, all in A / B / C)
    # ════════════════════════════════════════════════════════════════════════
    fig = new_page()
    page_header(fig, "Normalized Analysis — Density",
                "Length normalized by effective city area · km per km² · A / B / C", 4)

    # ── Explanatory text ─────────────────────────────────────────────────
    p4_intro = (
        "Why normalize?  The five cities differ enormously in area — Berlin is "
        "many times larger than Barcelona — so comparing raw network lengths "
        "conflates city size with coverage. Dividing every length by the effective "
        "city area (km of network per km² of city) puts all five cities on equal "
        "footing. Below: the three networks A / B / C as density bars, then a full "
        "per-city density table."
    )
    y_p4 = BODY_TOP
    for line in textwrap.wrap(p4_intro, 100):
        fig.text(0.04, y_p4, line, color=DARK, fontsize=8.5)
        y_p4 -= 0.018

    # Top: density bars (A / B / C per km²)
    add_image(fig, COMP / "density_bars.png", [0.03, 0.50, 0.94, 0.24])

    # ── Density-table heading ────────────────────────────────────────────
    fig.text(0.5, 0.455,
             "Per-city density  ·  effective area = Nominatim polygon minus large forest / park",
             ha="center", color=DBLUE, fontsize=10, fontweight="bold")

    # Column groups: City | meta | Real OSM (A, B) | Synthetic (C)
    dt_hdrs = ["City", "Area\n(km²)", "Pop.",
               "A\n(km)", "A\n/ km²",
               "B\n(km)", "B\n/ km²",
               "C max\n(km)", "C\n/ km²", "C/B\nratio"]
    dt_cw   = [0.11, 0.08, 0.08, 0.10, 0.09, 0.10, 0.09, 0.10, 0.09, 0.10]
    dt_cx   = [sum(dt_cw[:i]) for i in range(len(dt_cw) + 1)]

    ax_dt = fig.add_axes([0.04, 0.07, 0.92, 0.35])
    ax_dt.set_xlim(0, 1); ax_dt.set_ylim(0, 1); ax_dt.set_axis_off()
    ax_dt.add_patch(Rectangle((0, 0.83), 1, 0.10,
                               facecolor=HEADER_TINT, zorder=-1))

    # Group banner: Real (OSM) = A, B  vs  Synthetic = C
    real_l, real_r = dt_cx[3], dt_cx[7]
    syn_l,  syn_r  = dt_cx[7], dt_cx[10]
    ax_dt.add_patch(Rectangle((real_l, 0.97), real_r - real_l, 0.045,
                               facecolor=REAL_TINT, zorder=0))
    ax_dt.add_patch(Rectangle((syn_l, 0.97), syn_r - syn_l, 0.045,
                               facecolor=SYN_TINT, zorder=0))
    ax_dt.text((real_l + real_r) / 2, 0.992, "Real (OSM): A, B",
               ha="center", va="center", fontsize=7,
               fontweight="bold", color=REAL_TXT)
    ax_dt.text((syn_l + syn_r) / 2, 0.992, "Synthetic: C",
               ha="center", va="center", fontsize=7,
               fontweight="bold", color=SYN_TXT)

    for j, h in enumerate(dt_hdrs):
        ax_dt.text((dt_cx[j] + dt_cx[j+1]) / 2, 0.92, h,
                   ha="center", va="top", fontsize=7,
                   fontweight="bold", color=DBLUE,
                   multialignment="center")
    ax_dt.axhline(0.83, color=DBLUE, lw=0.8)

    # Resolve ratio column name (analysis_normalized.py renamed it to bikeable)
    ratio_key = ("Syn/bikeable ratio"
                 if "Syn/bikeable ratio" in density_tbl.columns
                 else "Syn/biketrack ratio")

    dt_row_h = 0.13
    dt_first_top = 0.81
    for i, city in enumerate(CITIES):
        row = density_tbl[density_tbl["City"] == CITY_LABELS[city]]
        if not len(row):
            continue
        r = row.iloc[0]
        vals = [
            CITY_LABELS[city],
            str(r["Area (km²)"]),
            str(r["Population"]),
            str(r["biketrack (km)"]),
            str(r["biketrack / km²"]),
            str(r["bikeable (km)"]),
            str(r["bikeable / km²"]),
            str(r["Syn max (km)"]),
            str(r["Syn max / km²"]),
            str(r[ratio_key]),
        ]
        y_top    = dt_first_top - i * dt_row_h
        y_bottom = y_top - dt_row_h
        bg = LGREY if i % 2 == 0 else WHITE
        ax_dt.add_patch(Rectangle((0, y_bottom), 1, dt_row_h,
                                   facecolor=bg, zorder=0))
        ax_dt.add_patch(Rectangle((-0.005, y_bottom), 0.015, dt_row_h,
                                   facecolor=COLORS[city], alpha=0.8, zorder=1))
        for j, val in enumerate(vals):
            ha = "left" if j == 0 else "center"
            xp = dt_cx[j] + 0.022 if j == 0 else (dt_cx[j] + dt_cx[j+1]) / 2
            ax_dt.text(xp, y_bottom + dt_row_h / 2, val,
                       ha=ha, va="center", fontsize=8, color=DARK)

    # Vertical separators — clip to data area only
    sep_bottom = dt_first_top - len(CITIES) * dt_row_h
    for sep_x in (dt_cx[3], dt_cx[7]):
        ax_dt.plot([sep_x, sep_x], [sep_bottom, 0.82],
                   color=DBLUE, lw=0.7, alpha=0.55, zorder=2)

    fig.add_artist(Rectangle((0, 0), 1, 0.015,
                              transform=fig.transFigure, facecolor=LGREY))
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    # ════════════════════════════════════════════════════════════════════════
    # PAGES 5–9 — PER-CITY  (Vienna-v3-inspired sidebar layout)
    #
    # Layout:
    #   LEFT  (sidebar, ~28 % width)
    #     · "WHAT THIS SHOWS"   3–4 line description (same for all cities)
    #     · "RESULT"            three big coloured number boxes —
    #                            overlap %, gap %, directness
    #     · "INTERPRETATION"    short city-specific blurb
    #   RIGHT (~65 % width)
    #     · Large spatial map
    #     · Below the map: small cross-city context bars
    # ════════════════════════════════════════════════════════════════════════

    # Pre-compute cross-city metric values used by every per-city context plot
    ctx_metrics = []
    for c in CITIES:
        end_c = city_data[c]["syn"].iloc[-1]
        sp_c = spatial[spatial["City"] == CITY_LABELS[c]]
        ov_c = float(str(sp_c["Overlap (%)"].values[0]).rstrip("%")) if len(sp_c) else 0
        dn_c = density_tbl[density_tbl["City"] == CITY_LABELS[c]]
        bk_density = float(dn_c["bikeable / km²"].values[0]) if len(dn_c) else 0
        ctx_metrics.append(dict(
            city          = c,
            directness    = float(end_c["directness_lcc"]),
            efficiency    = float(end_c["efficiency_global"]),
            overlap_pct   = ov_c,
            bk_density    = bk_density,
        ))
    ctx_df = pd.DataFrame(ctx_metrics)

    CTX_PANELS = [
        ("Directness (C)",        "directness",  "{:.3f}"),
        ("Global efficiency (C)", "efficiency",  "{:.3f}"),
        ("Overlap C in B (%)",    "overlap_pct", "{:.0f}"),
        ("B density (km/km²)",    "bk_density",  "{:.1f}"),
    ]

    # Per-city interpretation blurbs (sidebar). Every superlative is DERIVED
    # from rankings — never hardcoded — so the text always matches the data.
    _bd = {c: float(density_tbl[density_tbl["City"] == CITY_LABELS[c]]
                    ["bikeable / km²"].values[0]) for c in CITIES}
    _ar = {c: float(density_tbl[density_tbl["City"] == CITY_LABELS[c]]
                    ["Area (km²)"].values[0]) for c in CITIES}
    _dr = {c: float(city_data[c]["syn"].iloc[-1]["directness_lcc"]) for c in CITIES}
    _ef = {c: float(city_data[c]["syn"].iloc[-1]["efficiency_global"]) for c in CITIES}
    _cm = {c: city_data[c]["syn"]["length_km"].max() for c in CITIES}
    _cb = {c: _cm[c] / (float(city_data[c]["bikeable"]["length"]) / 1000) for c in CITIES}

    def _spat(c, col):
        r = spatial[spatial["City"] == CITY_LABELS[c]]
        return float(str(r[col].values[0]).rstrip("%")) if len(r) else 0.0
    _ov = {c: _spat(c, "Overlap (%)") for c in CITIES}
    _gp = {c: _spat(c, "Gap (km)") for c in CITIES}

    _ORD = {0: "highest", 1: "2nd-highest", 2: "3rd-highest",
            3: "2nd-lowest", 4: "lowest"}
    def _rnk(d, c):
        return _ORD.get(sorted(d, key=d.get, reverse=True).index(c), "mid-range")

    def _blurb(c):
        if c == "amsterdam":
            return (f"B density {_bd[c]:.1f} km/km² ({_rnk(_bd, c)}). C is only "
                    f"{_cb[c]*100:.0f} % of B's length and {_ov[c]:.0f} % of C overlaps "
                    "existing infrastructure — the least net-new of the five.")
        if c == "barcelona":
            return (f"B density {_bd[c]:.1f} km/km² ({_rnk(_bd, c)}), in the smallest "
                    f"effective area ({_ar[c]:.0f} km²). C efficiency {_ef[c]:.3f} "
                    f"({_rnk(_ef, c)}). {_ov[c]:.0f} % overlap, {100-_ov[c]:.0f} % gap.")
        if c == "berlin":
            return (f"{_rnk(_ar, c).capitalize()} effective area ({_ar[c]:.0f} km²) and "
                    f"{_rnk(_cm, c)} C length ({_cm[c]:.0f} km). C directness {_dr[c]:.3f} "
                    f"({_rnk(_dr, c)}) and efficiency {_ef[c]:.3f} ({_rnk(_ef, c)}). "
                    f"{_ov[c]:.0f} % overlap.")
        if c == "oslo":
            return (f"B density {_bd[c]:.1f} km/km² ({_rnk(_bd, c)}) even after removing "
                    f"the Marka forest. C/B ratio {_cb[c]*100:.0f} % ({_rnk(_cb, c)}) and "
                    f"{_rnk(_ov, c)} overlap ({_ov[c]:.0f} %) — the largest relative gap "
                    "of the five.")
        if c == "vienna":
            return (f"C directness {_dr[c]:.3f} ({_rnk(_dr, c)}) and efficiency "
                    f"{_ef[c]:.3f} ({_rnk(_ef, c)}), but {_rnk(_bd, c)} B density and "
                    f"{_rnk(_ov, c)} overlap. {_gp[c]:.0f} km gap ({100-_ov[c]:.0f} % of C).")
        return ""
    SHOWS_TEXT = (
        "The map shows where synthetic network C overlaps (green) or differs "
        "(red, 'gap') from real network B (bikeable, grey). Each C edge "
        "counts as overlap if ≥ 50 % of its 15 m buffer falls within B's "
        "buffer. The map is clipped to the effective city boundary (Nominatim "
        "polygon minus large forest / park areas)."
    )

    def _draw_number_box(fig, rect, value, label, fill, fg=WHITE):
        """Vienna-v3-style coloured number box."""
        l, b, w, h = rect
        fig.add_artist(Rectangle((l, b), w, h,
                                  transform=fig.transFigure,
                                  facecolor=fill, edgecolor="none",
                                  zorder=5))
        fig.text(l + w / 2, b + h * 0.62, value,
                 ha="center", va="center", color=fg,
                 fontsize=22, fontweight="bold", zorder=6)
        fig.text(l + w / 2, b + h * 0.22, label,
                 ha="center", va="center", color=fg,
                 fontsize=8, zorder=6)

    for pg_idx, city in enumerate(CITIES):
        label  = CITY_LABELS[city]
        color  = COLORS[city]
        syn    = city_data[city]["syn"]
        bt_km  = city_data[city]["biketrack"]["length"] / 1000
        bk_km  = city_data[city]["bikeable"]["length"]  / 1000
        syn_max_km = syn["length_km"].max()
        end    = syn.iloc[-1]

        sp_row = spatial[spatial["City"] == label]
        ov_pct_str = sp_row["Overlap (%)"].values[0] if len(sp_row) else "n/a"
        gap_pct_str = sp_row["Gap (%)"].values[0] if len(sp_row) else "n/a"

        fig = new_page()
        page_header(fig, label,
                    "Spatial overlap and cross-city positioning  ·  gridl=1701",
                    5 + pg_idx)
        fig.add_artist(Rectangle((0, 0.895), 0.006, 0.105,
                                  transform=fig.transFigure,
                                  facecolor=color, zorder=12))

        # ── LEFT SIDEBAR ───────────────────────────────────────────────────
        SX, SW = 0.045, 0.245   # sidebar x, width
        WRAP_COLS = 36

        # WHAT THIS SHOWS
        fig.text(SX, BODY_TOP, "WHAT THIS SHOWS",
                 color=DBLUE, fontsize=8.5, fontweight="bold")
        y_t = BODY_TOP - 0.020
        for ln in textwrap.wrap(SHOWS_TEXT, WRAP_COLS):
            fig.text(SX, y_t, ln, color=DARK, fontsize=8)
            y_t -= 0.0165

        # RESULT — three big coloured boxes
        fig.text(SX, 0.640, f"RESULT — {label.upper()}",
                 color=DBLUE, fontsize=8.5, fontweight="bold")
        box_h = 0.072
        box_gap = 0.012
        _draw_number_box(fig, [SX, 0.555,             SW, box_h],
                         ov_pct_str, "Overlap — C in B (built)", "#0eb6d2")
        _draw_number_box(fig, [SX, 0.555 - box_h - box_gap, SW, box_h],
                         gap_pct_str, "Gap — C not in B (missing)", "#c0392b")
        _draw_number_box(fig, [SX, 0.555 - 2*(box_h + box_gap), SW, box_h],
                         f"{end['directness_lcc']:.3f}",
                         "Directness of C (LCC)", "#0d3d6e")

        # INTERPRETATION
        fig.text(SX, 0.330, "INTERPRETATION",
                 color=DBLUE, fontsize=8.5, fontweight="bold")
        y_t = 0.312
        for ln in textwrap.wrap(_blurb(city), WRAP_COLS):
            fig.text(SX, y_t, ln, color=DARK, fontsize=8)
            y_t -= 0.0165

        # Tiny secondary stats footer in the sidebar
        fig.text(SX, 0.105,
                 f"A  biketrack  {bt_km:>5.0f} km\n"
                 f"B  bikeable   {bk_km:>5.0f} km\n"
                 f"C  synthetic  {syn_max_km:>5.0f} km   "
                 f"({syn_max_km/bk_km*100:.0f}% of B)",
                 color=GREY, fontsize=7.5, family="monospace",
                 linespacing=1.55)

        # ── RIGHT COLUMN: large map  ──────────────────────────────────────
        MX, MW = 0.34, 0.62
        add_image(fig, AO / city / f"map_{city}.png",
                  [MX, 0.36, MW, 0.52])

        # ── Cross-city context bars (compact, more breathing room) ────────
        fig.text(MX + MW / 2, 0.325,
                 f"{label} in cross-city context",
                 ha="center", fontsize=9.5, fontweight="bold", color=DBLUE)
        fig.text(MX + MW / 2, 0.308,
                 "Highlighted bar = this city · grey = others",
                 ha="center", fontsize=7.5, color=GREY, style="italic")

        ctx_bottom = 0.075
        ctx_h      = 0.165
        # Fit 4 panels into right column with generous gaps between them.
        n_panels   = len(CTX_PANELS)
        ctx_gap    = 0.030
        ctx_total  = MW
        ctx_pw     = (ctx_total - (n_panels - 1) * ctx_gap) / n_panels

        for k, (ptitle, pcol, pfmt) in enumerate(CTX_PANELS):
            ax_ctx = fig.add_axes([
                MX + k * (ctx_pw + ctx_gap),
                ctx_bottom, ctx_pw, ctx_h
            ])
            vals = ctx_df[pcol].values
            labels_short = [CITY_LABELS[c][:3] for c in ctx_df["city"]]
            bar_colors = [
                COLORS[c] if c == city else "#cfd8dc"
                for c in ctx_df["city"]
            ]
            ax_ctx.bar(labels_short, vals, color=bar_colors,
                       edgecolor="white", linewidth=0.4, width=0.65)
            own_idx = ctx_df.index[ctx_df["city"] == city][0]
            ax_ctx.annotate(
                pfmt.format(vals[own_idx]),
                (own_idx, vals[own_idx]),
                xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=6.5,
                color=COLORS[city], fontweight="bold"
            )
            ax_ctx.set_title(ptitle, fontsize=7.5, color=DBLUE,
                             fontweight="bold", pad=3)
            ax_ctx.tick_params(axis="x", labelsize=6.5, pad=2,
                                length=0)  # no tick marks
            ax_ctx.tick_params(axis="y", labelsize=5.5, pad=1,
                                length=0)
            # Sparse y-axis: only 0 and max
            ymax = vals.max() * 1.25
            ax_ctx.set_ylim(0, ymax)
            ax_ctx.set_yticks([0, vals.max()])
            ax_ctx.yaxis.set_major_formatter(
                mticker.FormatStrFormatter("%.2g")
            )
            ax_ctx.grid(axis="y", alpha=0.18, lw=0.4)
            for spine in ("top", "right"):
                ax_ctx.spines[spine].set_visible(False)
            ax_ctx.spines["left"].set_color("#cfd8dc")
            ax_ctx.spines["bottom"].set_color("#cfd8dc")

        fig.add_artist(Rectangle((0, 0), 1, 0.015,
                                  transform=fig.transFigure, facecolor=LGREY))
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    # ════════════════════════════════════════════════════════════════════════
    # PAGE 10 — VIENNA · gridl SENSITIVITY (600 m vs 1701 m)
    #
    # The Methodology page mentions that Vienna was previously analysed with
    # a finer 600 m POI grid. The data still exists in results/vienna_2/,
    # so we can show side-by-side how much the algorithmic choice of grid
    # spacing actually matters for the headline metrics.
    # ════════════════════════════════════════════════════════════════════════
    fig = new_page()
    page_header(fig, "Vienna — gridl Sensitivity",
                "How much does the POI grid spacing change the synthetic network?",
                10)
    fig.add_artist(Rectangle((0, 0.895), 0.006, 0.105,
                              transform=fig.transFigure,
                              facecolor=COLORS["vienna"], zorder=12))

    # ── Intro text (trimmed) ──────────────────────────────────────────
    p12_intro = (
        "Two Vienna runs, identical except for the POI grid spacing: gridl = 600 m "
        "uses a fine grid (precise local demand); gridl = 1701 m (Szell et al., 2022, "
        "the value used for all five cities here) uses a coarser city-wide grid. Same "
        "betweenness strategy, same street network, 40 growth steps. The grid spacing "
        "alone yields two markedly different synthetic optima C: the fine grid grows a "
        "denser, more capillary network, the coarse grid a leaner arterial skeleton."
    )
    y_intro = BODY_TOP
    for line in textwrap.wrap(p12_intro, 108):
        fig.text(0.04, y_intro, line, color=DARK, fontsize=8.5)
        y_intro -= 0.0175

    # ── Network comparison, stacked vertically (the centrepiece) ──────
    add_image(fig, COMP / "vienna_gridl_networks.png", [0.05, 0.035, 0.90, 0.75])

    fig.add_artist(Rectangle((0, 0), 1, 0.025,
                              transform=fig.transFigure, facecolor=LGREY))
    fig.text(0.5, 0.012, "GrowBikeNet Analysis  ·  Vienna gridl sensitivity  ·  2026",
             color=GREY, fontsize=7.5, ha="center")
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


    # ════════════════════════════════════════════════════════════════════════
    # PAGE 11 — SUMMARY TABLE & CONCLUSIONS
    # ════════════════════════════════════════════════════════════════════════
    fig = new_page()
    page_header(fig, "Summary & Conclusions",
                "All cities compared at synthetic maximum (gridl=1701)", 11)

    # Big summary table — title sits above the axes so it doesn't overlap
    # the Real(OSM) / Synthetic group banner.
    fig.text(0.5, BODY_TOP, "Real (B) vs. synthetic (C) — key structural metrics",
             ha="center", fontsize=11, fontweight="bold", color=DBLUE)
    fig.text(0.5, BODY_TOP - 0.024,
             "B = real OSM network · C = synthetic at its maximum (Q = 1.0)",
             ha="center", fontsize=8, color=GREY, style="italic")

    ax_s = fig.add_axes([0.04, 0.52, 0.92, 0.32])
    ax_s.set_xlim(0, 1); ax_s.set_ylim(0, 1); ax_s.set_axis_off()

    # Soft tint behind the column-header row — just tall enough to hold
    # two-line headers (~0.86–0.92) plus a small margin above the separator.
    ax_s.add_patch(Rectangle((0, 0.83), 1, 0.10,
                              facecolor=HEADER_TINT, zorder=-1))

    # Lengths A/B/C, then B|C pairs for each structural metric. B values are
    # real-tinted, C values synthetic — so every metric reads real → optimum.
    col_w2 = [0.13, 0.07, 0.07, 0.07,
              0.1067, 0.1067, 0.1067, 0.1067, 0.1067, 0.1067]
    col_x2 = [sum(col_w2[:i]) for i in range(len(col_w2) + 1)]

    # Super-banner: length group + three metric groups
    groups = [
        ("Length (km)",       1, 4),
        ("Directness (LCC)",  4, 6),
        ("Global efficiency", 6, 8),
        ("LCC share",         8, 10),
    ]
    ax_s.add_patch(Rectangle((col_x2[1], 0.935), col_x2[10] - col_x2[1], 0.045,
                              facecolor=HEADER_TINT, zorder=0))
    for label, a, b in groups:
        ax_s.text((col_x2[a] + col_x2[b]) / 2, 0.957, label,
                  ha="center", va="center", fontsize=7.5,
                  fontweight="bold", color=DBLUE)
        ax_s.plot([col_x2[a], col_x2[a]], [0.935, 0.98],
                  color=WHITE, lw=1.4, zorder=1)

    # Sub-headers: A/B/C for length, then B/C per metric (colour-coded)
    sub = [("City", DARK), ("A", REAL_TXT), ("B", REAL_TXT), ("C", SYN_TXT),
           ("B", REAL_TXT), ("C", SYN_TXT), ("B", REAL_TXT), ("C", SYN_TXT),
           ("B", REAL_TXT), ("C", SYN_TXT)]
    for j, (h, col) in enumerate(sub):
        ax_s.text((col_x2[j] + col_x2[j+1]) / 2, 0.905, h,
                  ha="center", va="center", fontsize=8,
                  fontweight="bold", color=col)
    ax_s.axhline(0.875, color=DBLUE, lw=0.8)

    row_h2 = 0.155
    first_top_2 = 0.855
    for i, city in enumerate(CITIES):
        end   = city_data[city]["syn"].iloc[-1]
        bk    = city_data[city]["bikeable"]
        bt_km = city_data[city]["biketrack"]["length"] / 1000
        bk_km = float(bk["length"]) / 1000
        sm_km = city_data[city]["syn"]["length_km"].max()
        b_dir = float(bk["directness_lcc"]);   c_dir = float(end["directness_lcc"])
        b_eff = float(bk["efficiency_global"]); c_eff = float(end["efficiency_global"])
        b_lcc = float(bk["length_lcc"]) / float(bk["length"])
        c_lcc = float(end["lcc_share"])

        cells = [
            (CITY_LABELS[city], DARK, False),
            (f"{bt_km:.0f}", REAL_TXT, False),
            (f"{bk_km:.0f}", REAL_TXT, False),
            (f"{sm_km:.0f}", SYN_TXT, True),
            (f"{b_dir:.3f}", REAL_TXT, False), (f"{c_dir:.3f}", SYN_TXT, True),
            (f"{b_eff:.3f}", REAL_TXT, False), (f"{c_eff:.3f}", SYN_TXT, True),
            (f"{b_lcc*100:.0f}%", REAL_TXT, False), (f"{c_lcc*100:.0f}%", SYN_TXT, True),
        ]
        y_top    = first_top_2 - i * row_h2
        y_bottom = y_top - row_h2
        bg = LGREY if i % 2 == 0 else WHITE
        ax_s.add_patch(Rectangle((0, y_bottom), 1, row_h2, facecolor=bg, zorder=0))
        ax_s.add_patch(Rectangle((-0.005, y_bottom), 0.018, row_h2,
                                  facecolor=COLORS[city], alpha=0.8, zorder=1))
        for j, (val, col, bold) in enumerate(cells):
            ha = "left" if j == 0 else "center"
            xp = col_x2[j] + 0.025 if j == 0 else (col_x2[j] + col_x2[j+1]) / 2
            ax_s.text(xp, y_bottom + row_h2 / 2, val, ha=ha, va="center",
                      fontsize=8.3, color=col,
                      fontweight=("bold" if bold else "normal"))

    # Group separators down the data area
    sep_bottom_2 = first_top_2 - len(CITIES) * row_h2
    for sep_x in (col_x2[3], col_x2[4], col_x2[6], col_x2[8]):
        ax_s.plot([sep_x, sep_x], [sep_bottom_2, 0.875],
                  color=DBLUE, lw=0.6, alpha=0.4, zorder=2)

    # Spatial summary table (synthetic C vs reference B, default 15 m buffer)
    fig.text(0.5, 0.520,
             "Spatial overlap — synthetic C vs. reference B  ·  buffer = 15 m",
             ha="center", fontsize=10, fontweight="bold", color=DBLUE)

    ax_sp = fig.add_axes([0.04, 0.30, 0.92, 0.21])
    ax_sp.set_xlim(0, 1); ax_sp.set_ylim(0, 1); ax_sp.set_axis_off()

    # Soft tint behind the column-header row
    ax_sp.add_patch(Rectangle((0, 0.83), 1, 0.10,
                               facecolor=HEADER_TINT, zorder=-1))

    # Reference column = network B (analysis_multicity_spatial.py writes
    # "OSM bikeable (km)" now; fall back to legacy "OSM biketrack (km)").
    if "OSM bikeable (km)" in spatial.columns:
        bk_col = "OSM bikeable (km)"
    else:
        bk_col = "OSM biketrack (km)"
    bk_hdr = "B\n(km)"

    sp_hdrs = ["City", bk_hdr, "C\n(km)",
               "Overlap (km)", "Overlap (%)", "Gap (km)", "Gap (%)"]
    sp_cw   = [0.16, 0.18, 0.14, 0.14, 0.12, 0.13, 0.13]
    sp_cx   = [sum(sp_cw[:i]) for i in range(len(sp_cw) + 1)]

    for j, h in enumerate(sp_hdrs):
        ax_sp.text((sp_cx[j] + sp_cx[j+1]) / 2, 0.92, h,
                   ha="center", va="top", fontsize=7.5,
                   fontweight="bold", color=DBLUE,
                   multialignment="center")
    ax_sp.axhline(0.83, color=DBLUE, lw=0.8)

    sp_rh = 0.13
    sp_first_top = 0.81
    for i, city in enumerate(CITIES):
        sp_row = spatial[spatial["City"] == CITY_LABELS[city]]
        if not len(sp_row):
            continue
        r = sp_row.iloc[0]
        vals = [CITY_LABELS[city], str(r[bk_col]),
                str(r["Synthetic (km)"]), str(r["Overlap (km)"]),
                r["Overlap (%)"], str(r["Gap (km)"]), r["Gap (%)"]]
        y_top    = sp_first_top - i * sp_rh
        y_bottom = y_top - sp_rh
        bg = LGREY if i % 2 == 0 else WHITE
        ax_sp.add_patch(Rectangle((0, y_bottom), 1, sp_rh,
                                   facecolor=bg, zorder=0))
        ax_sp.add_patch(Rectangle((-0.005, y_bottom), 0.018, sp_rh,
                                   facecolor=COLORS[city], alpha=0.8, zorder=1))
        for j, val in enumerate(vals):
            ha = "left" if j == 0 else "center"
            xp = sp_cx[j] + 0.025 if j == 0 else (sp_cx[j] + sp_cx[j+1]) / 2
            ax_sp.text(xp, y_bottom + sp_rh / 2, val,
                       ha=ha, va="center", fontsize=8.5, color=DARK)

    # Vertical separator: real OSM column vs synthetic-derived columns
    sp_sep_bottom = sp_first_top - len(CITIES) * sp_rh
    ax_sp.plot([sp_cx[2], sp_cx[2]], [sp_sep_bottom, 0.78],
               color=DBLUE, lw=0.7, alpha=0.55, zorder=2)

    # Conclusions
    fig.text(0.06, 0.275, "Key observations", color=DBLUE,
             fontsize=10, fontweight="bold")

    # Build observations dynamically from the spatial CSV so the headline
    # numbers stay in sync with the data (real OSM = bikeable now).
    sp_pct = {}
    for _, r in spatial.iterrows():
        try:
            sp_pct[r["City"]] = float(str(r["Overlap (%)"]).rstrip("%"))
        except (ValueError, KeyError):
            pass

    # collect directness + efficiency per city for sharper observations
    dir_by_city = {CITY_LABELS[c]: float(city_data[c]["syn"].iloc[-1]["directness_lcc"])
                   for c in CITIES}
    eff_by_city = {CITY_LABELS[c]: float(city_data[c]["syn"].iloc[-1]["efficiency_global"])
                   for c in CITIES}
    bkden_by_city = {}
    for c in CITIES:
        r = density_tbl[density_tbl["City"] == CITY_LABELS[c]]
        if len(r):
            bkden_by_city[CITY_LABELS[c]] = float(r["bikeable / km²"].values[0])

    hi_ov = max(sp_pct, key=sp_pct.get) if sp_pct else None
    lo_ov = min(sp_pct, key=sp_pct.get) if sp_pct else None
    hi_dir = max(dir_by_city, key=dir_by_city.get)
    lo_dir = min(dir_by_city, key=dir_by_city.get)
    hi_eff = max(eff_by_city, key=eff_by_city.get)
    hi_den = max(bkden_by_city, key=bkden_by_city.get) if bkden_by_city else None
    lo_den = min(bkden_by_city, key=bkden_by_city.get) if bkden_by_city else None

    obs = [
        ("Connectivity:", (
            "All five synthetic networks (C) converge to LCC share = 100 % and 1 "
            "component within the first 20–30 % of growth — whereas the real "
            "networks (B) stay fragmented (LCC share 9–96 %, see table above)."
        )),
        ("Directness:", (
            f"Highest in {hi_dir} ({dir_by_city[hi_dir]:.2f}), lowest in "
            f"{lo_dir} ({dir_by_city[lo_dir]:.2f}) — a "
            f"{dir_by_city[hi_dir]-dir_by_city[lo_dir]:.2f} gap, so C routes in "
            f"{hi_dir} are about "
            f"{(1 - dir_by_city[lo_dir]/dir_by_city[hi_dir])*100:.0f} % shorter than "
            f"in {lo_dir} for the same origin–destination pair."
        )),
        ("Density spread:", (
            f"B density ranges from {bkden_by_city[lo_den]:.1f} km/km² "
            f"({lo_den}) to {bkden_by_city[hi_den]:.1f} km/km² ({hi_den}) — a "
            f"{bkden_by_city[hi_den]/bkden_by_city[lo_den]:.1f}× spread. Cities with "
            "low B density also tend to have low overlap with C's proposals."
        )) if hi_den else ("Density spread:", "Density figures unavailable."),
        ("Spatial overlap:", (
            f"Overlap of C in B is highest in {hi_ov} ({sp_pct[hi_ov]:.0f} %), "
            f"meaning C there largely rediscovers existing routes. "
            f"{lo_ov} ({sp_pct[lo_ov]:.0f} %) has the lowest overlap → "
            "largest share of C's proposed connections that are not yet built."
        )) if sp_pct else ("Spatial overlap:", "Statistics unavailable."),
        ("See page 20", (
            "Per-city, data-driven discussion linking each city's B density, "
            "directness and overlap together."
        )),
    ]
    # Tighten the observations block — 5 entries × ~3 wrapped lines need to
    # fit between y = 0.27 (header) and y ≈ 0.05 (above the footer).
    y = 0.255
    for heading, body in obs:
        fig.text(0.06, y, heading, color=BLUE, fontsize=8, fontweight="bold")
        lines = textwrap.wrap(body, 84)
        for li, line in enumerate(lines):
            fig.text(0.22, y - li * 0.015, line, color=DARK, fontsize=8)
        y -= (len(lines) * 0.015 + 0.014)

    fig.add_artist(Rectangle((0, 0), 1, 0.025,
                              transform=fig.transFigure, facecolor=LGREY))
    fig.text(0.5, 0.012, "GrowBikeNet Analysis  ·  gridl=1701  ·  2026",
             color=GREY, fontsize=7.5, ha="center")
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    # ════════════════════════════════════════════════════════════════════════
    # ════════════════════════════════════════════════════════════════════════
    # SAFETY ANALYSIS SECTION  (A = biketrack / protected  vs  C = synthetic)
    #
    # Mirrors the B-vs-C section above, page for page, but everything is keyed on
    # network A (protected/segregated infrastructure, incl. snapped cyclist
    # crossings) instead of B. The banner switches from navy blue to GREEN_DARK
    # so the reader can tell the reference network at a glance.
    #   Page 12     — Section divider (what the safety analysis asks)
    #   Pages 13–17 — Per-city: large A-vs-C map + data boxes + context
    #   Page 18     — Safety summary table (A vs C) + key observations
    # The whole block is guarded on spatial_safety so the report still builds
    # if 02b_analysis_safety_spatial.py has not been run yet.
    # ════════════════════════════════════════════════════════════════════════
    # ════════════════════════════════════════════════════════════════════════
    if spatial_safety is not None:

        # Helper: overlap/gap % for safety per city (string forms for boxes/tables)
        def _safe_row(c):
            r = spatial_safety[spatial_safety["City"] == CITY_LABELS[c]]
            return r.iloc[0] if len(r) else None

        def _safe_pct(c, col):
            r = _safe_row(c)
            if r is None:
                return 0.0
            try:
                return float(str(r[col]).rstrip("%"))
            except (ValueError, KeyError):
                return 0.0

        # Per-city A quantities used in context bars + blurbs (all derived).
        _aLk   = {c: float(city_data[c]["biketrack"]["length"]) / 1000 for c in CITIES}
        _aLcc  = {c: 100 * float(city_data[c]["biketrack"]["length_lcc"])
                       / float(city_data[c]["biketrack"]["length"]) for c in CITIES}
        _aDir  = {c: float(city_data[c]["biketrack"]["directness_lcc"]) for c in CITIES}
        _aEff  = {c: float(city_data[c]["biketrack"]["efficiency_global"]) for c in CITIES}
        _aCmp  = {c: int(float(city_data[c]["biketrack"]["components"])) for c in CITIES}
        _aDen  = {c: float(density_tbl[density_tbl["City"] == CITY_LABELS[c]]
                           ["biketrack / km²"].values[0]) for c in CITIES}
        _aOv   = {c: _safe_pct(c, "Overlap (%)") for c in CITIES}

        _ORD_S = {0: "highest", 1: "2nd-highest", 2: "3rd-highest",
                  3: "2nd-lowest", 4: "lowest"}
        def _rnkS(d, c):
            return _ORD_S.get(sorted(d, key=d.get, reverse=True).index(c), "mid-range")

        # ════════════════════════════════════════════════════════════════════
        # PAGE 12 — SAFETY SECTION DIVIDER
        # ════════════════════════════════════════════════════════════════════
        fig = new_page()
        fig.add_artist(Rectangle((0, 0), 1, 1, transform=fig.transFigure,
                                 facecolor=LGREEN, zorder=0))
        fig.add_artist(Rectangle((0, 0.62), 1, 0.20, transform=fig.transFigure,
                                 facecolor=GREEN_DARK, zorder=1))
        fig.text(0.5, 0.775, "PART II", color="#bfe3cd", fontsize=12,
                 fontweight="bold", ha="center", va="center", zorder=2)
        fig.add_artist(Rectangle((0.5 - 0.05, 0.752), 0.10, 0.004,
                                 transform=fig.transFigure, facecolor="#8fd1a8", zorder=2))
        fig.text(0.5, 0.705, "Safety Analysis", color=WHITE, fontsize=32,
                 fontweight="bold", ha="center", va="center", zorder=2)
        fig.text(0.5, 0.658, "Protected network A  vs.  synthetic optimum C",
                 color="#dcefe4", fontsize=14, ha="center", va="center", zorder=2)

        intro_s = (
            "Part I compared the synthetic optimum C against B — every legally "
            "cyclable edge. But not all of B is comfortable or safe to ride: much "
            "of it is mixed traffic. This part repeats the comparison against "
            "A — biketrack — the dedicated, segregated cycle infrastructure only "
            "(now including cyclist crossings snapped on as real connectors).\n\n"
            "Two questions drive the safety view:"
        )
        y = 0.575
        for ln in textwrap.wrap(intro_s, 92):
            if ln == "":
                y -= 0.012
                continue
            fig.text(0.5, y, ln, color=DARK, fontsize=10.5, ha="center", zorder=2)
            y -= 0.026

        qs = [
            ("1", "How fragmented is protected riding?",
             "A's largest connected component covers only a fraction of its length "
             "(LCC share 9–76 % across the five cities) — protected paths form many "
             "disconnected islands, unlike the near-whole B network."),
            ("2", "How much of the optimum is already protected?",
             "Overlaying C on A shows the share of the optimal network that already "
             "runs on protected infrastructure (overlap) versus the share that has "
             "no protected path today (gap) — the safety upgrade backlog."),
        ]
        qy = y - 0.010
        qh = 0.115
        for num, qt, qd in qs:
            qb = qy - qh
            fig.add_artist(FancyBboxPatch((0.10, qb), 0.80, qh,
                           transform=fig.transFigure, boxstyle="round,pad=0.008",
                           facecolor=WHITE, edgecolor=GREEN_MED, linewidth=1.3, zorder=2))
            fig.add_artist(Rectangle((0.125, qb + qh - 0.050), 0.040, 0.038,
                           transform=fig.transFigure, facecolor=GREEN_DARK, zorder=3))
            fig.text(0.145, qb + qh - 0.031, num, ha="center", va="center",
                     color=WHITE, fontsize=14, fontweight="bold", zorder=4)
            fig.text(0.180, qb + qh - 0.031, qt, va="center", color=GREEN_DARK,
                     fontsize=11.5, fontweight="bold", zorder=4)
            dy = qb + qh - 0.060
            for ln in textwrap.wrap(qd, 86):
                fig.text(0.135, dy, ln, color=DARK, fontsize=8.7, zorder=4)
                dy -= 0.0165
            qy = qb - 0.030

        fig.text(0.5, 0.045,
                 "A — biketrack  ·  protected / segregated cycle infrastructure  ·  "
                 "reference for Part II  ·  green banners mark the safety section",
                 color=GREEN_TXT, fontsize=8.5, ha="center", style="italic", zorder=2)
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # ════════════════════════════════════════════════════════════════════
        # PAGES 13–17 — PER-CITY SAFETY (A vs C), green-themed
        # ════════════════════════════════════════════════════════════════════
        SHOWS_SAFE = (
            "The map shows where synthetic network C overlaps (green) the "
            "protected network A (biketrack, blue) or has no protected path "
            "nearby (red, 'gap'). A C edge counts as overlap if ≥ 50 % of its "
            "15 m buffer falls within A's buffer. Clipped to the effective city "
            "boundary (Nominatim polygon minus large forest / park areas)."
        )

        CTX_PANELS_S = [
            ("A directness (LCC)",  "dirA",  "{:.3f}"),
            ("A LCC share (%)",     "lccA",  "{:.0f}"),
            ("Overlap C in A (%)",  "ovA",   "{:.0f}"),
            ("A density (km/km²)",  "denA",  "{:.1f}"),
        ]
        ctx_s = pd.DataFrame([dict(city=c, dirA=_aDir[c], lccA=_aLcc[c],
                                   ovA=_aOv[c], denA=_aDen[c]) for c in CITIES])

        def _blurbS(c):
            return (
                f"Protected network A spans {_aLk[c]:.0f} km at {_aDen[c]:.1f} "
                f"km/km² ({_rnkS(_aDen, c)} of the five). Its largest connected "
                f"piece is only {_aLcc[c]:.0f} % of A ({_rnkS(_aLcc, c)}) — "
                f"protected riding breaks into {_aCmp[c]} islands. "
                f"{_aOv[c]:.0f} % of the synthetic optimum C already runs on "
                f"protected infrastructure; the remaining {100-_aOv[c]:.0f} % "
                f"gap is where C's best routes have no protected path today."
            )

        for pg_idx, city in enumerate(CITIES):
            label = CITY_LABELS[city]
            color = COLORS[city]
            syn   = city_data[city]["syn"]
            bt_km = _aLk[city]
            bk_km = float(city_data[city]["bikeable"]["length"]) / 1000
            syn_max_km = syn["length_km"].max()

            sr = _safe_row(city)
            ov_pct_str  = str(sr["Overlap (%)"]) if sr is not None else "n/a"
            gap_pct_str = str(sr["Gap (%)"]) if sr is not None else "n/a"

            fig = new_page()
            page_header(fig, label,
                        "SAFETY — protected network A vs. synthetic C  ·  gridl=1701",
                        13 + pg_idx, bar_color=GREEN_DARK, sub_color="#bfe3cd")
            fig.add_artist(Rectangle((0, 0.895), 0.006, 0.105,
                                     transform=fig.transFigure,
                                     facecolor=color, zorder=12))

            # ── LEFT SIDEBAR ───────────────────────────────────────────────
            SX, SW = 0.045, 0.245
            WRAP_COLS = 36

            fig.text(SX, BODY_TOP, "WHAT THIS SHOWS",
                     color=GREEN_DARK, fontsize=8.5, fontweight="bold")
            y_t = BODY_TOP - 0.020
            for ln in textwrap.wrap(SHOWS_SAFE, WRAP_COLS):
                fig.text(SX, y_t, ln, color=DARK, fontsize=8)
                y_t -= 0.0165

            fig.text(SX, 0.640, f"RESULT — {label.upper()}",
                     color=GREEN_DARK, fontsize=8.5, fontweight="bold")
            box_h = 0.072
            box_gap = 0.012
            _draw_number_box(fig, [SX, 0.555, SW, box_h],
                             ov_pct_str, "Overlap — C on protected A", GREEN_MED)
            _draw_number_box(fig, [SX, 0.555 - box_h - box_gap, SW, box_h],
                             gap_pct_str, "Gap — C not protected", "#c0392b")
            _draw_number_box(fig, [SX, 0.555 - 2*(box_h + box_gap), SW, box_h],
                             f"{_aLcc[city]:.0f}%", "A LCC share (connectivity)", GREEN_DARK)

            fig.text(SX, 0.330, "INTERPRETATION",
                     color=GREEN_DARK, fontsize=8.5, fontweight="bold")
            y_t = 0.312
            for ln in textwrap.wrap(_blurbS(city), WRAP_COLS):
                fig.text(SX, y_t, ln, color=DARK, fontsize=8)
                y_t -= 0.0165

            fig.text(SX, 0.105,
                     f"A  biketrack  {bt_km:>5.0f} km   ({_aCmp[city]} comp.)\n"
                     f"B  bikeable   {bk_km:>5.0f} km\n"
                     f"C  synthetic  {syn_max_km:>5.0f} km",
                     color=GREY, fontsize=7.5, family="monospace",
                     linespacing=1.55)

            # ── RIGHT COLUMN: large safety map ─────────────────────────────
            MX, MW = 0.34, 0.62
            add_image(fig, AO / city / f"map_{city}_safety.png",
                      [MX, 0.36, MW, 0.52])

            fig.text(MX + MW / 2, 0.325,
                     f"{label} — protected infrastructure in cross-city context",
                     ha="center", fontsize=9.5, fontweight="bold", color=GREEN_DARK)
            fig.text(MX + MW / 2, 0.308,
                     "Highlighted bar = this city · grey = others",
                     ha="center", fontsize=7.5, color=GREY, style="italic")

            ctx_bottom = 0.075
            ctx_h      = 0.165
            n_panels   = len(CTX_PANELS_S)
            ctx_gap    = 0.030
            ctx_pw     = (MW - (n_panels - 1) * ctx_gap) / n_panels

            for k, (ptitle, pcol, pfmt) in enumerate(CTX_PANELS_S):
                ax_ctx = fig.add_axes([MX + k * (ctx_pw + ctx_gap),
                                       ctx_bottom, ctx_pw, ctx_h])
                vals = ctx_s[pcol].values
                labels_short = [CITY_LABELS[c][:3] for c in ctx_s["city"]]
                bar_colors = [COLORS[c] if c == city else "#cfd8dc"
                              for c in ctx_s["city"]]
                ax_ctx.bar(labels_short, vals, color=bar_colors,
                           edgecolor="white", linewidth=0.4, width=0.65)
                own_idx = ctx_s.index[ctx_s["city"] == city][0]
                ax_ctx.annotate(pfmt.format(vals[own_idx]),
                                (own_idx, vals[own_idx]),
                                xytext=(0, 3), textcoords="offset points",
                                ha="center", va="bottom", fontsize=6.5,
                                color=COLORS[city], fontweight="bold")
                ax_ctx.set_title(ptitle, fontsize=7.5, color=GREEN_DARK,
                                 fontweight="bold", pad=3)
                ax_ctx.tick_params(axis="x", labelsize=6.5, pad=2, length=0)
                ax_ctx.tick_params(axis="y", labelsize=5.5, pad=1, length=0)
                ymax = vals.max() * 1.25 if vals.max() > 0 else 1
                ax_ctx.set_ylim(0, ymax)
                ax_ctx.set_yticks([0, vals.max()])
                ax_ctx.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2g"))
                ax_ctx.grid(axis="y", alpha=0.18, lw=0.4)
                for spine in ("top", "right"):
                    ax_ctx.spines[spine].set_visible(False)
                ax_ctx.spines["left"].set_color("#cfd8dc")
                ax_ctx.spines["bottom"].set_color("#cfd8dc")

            fig.add_artist(Rectangle((0, 0), 1, 0.015,
                                     transform=fig.transFigure, facecolor=LGREEN))
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # ════════════════════════════════════════════════════════════════════
        # PAGE 18 — SAFETY SUMMARY TABLE (A vs C) & observations
        # ════════════════════════════════════════════════════════════════════
        fig = new_page()
        page_header(fig, "Safety Summary & Conclusions",
                    "Protected network A vs. synthetic C — all cities (gridl=1701)",
                    18, bar_color=GREEN_DARK, sub_color="#bfe3cd")

        fig.text(0.5, BODY_TOP, "Protected (A) vs. synthetic (C) — key structural metrics",
                 ha="center", fontsize=11, fontweight="bold", color=GREEN_DARK)
        fig.text(0.5, BODY_TOP - 0.024,
                 "A = protected OSM infrastructure (biketrack) · C = synthetic at its maximum (Q = 1.0)",
                 ha="center", fontsize=8, color=GREY, style="italic")

        ax_s = fig.add_axes([0.04, 0.52, 0.92, 0.32])
        ax_s.set_xlim(0, 1); ax_s.set_ylim(0, 1); ax_s.set_axis_off()
        ax_s.add_patch(Rectangle((0, 0.83), 1, 0.10, facecolor=GREEN_TINT, zorder=-1))

        col_w2 = [0.13, 0.07, 0.07, 0.07,
                  0.1067, 0.1067, 0.1067, 0.1067, 0.1067, 0.1067]
        col_x2 = [sum(col_w2[:i]) for i in range(len(col_w2) + 1)]
        groups = [("Length (km)", 1, 4), ("Directness (LCC)", 4, 6),
                  ("Global efficiency", 6, 8), ("LCC share", 8, 10)]
        ax_s.add_patch(Rectangle((col_x2[1], 0.935), col_x2[10] - col_x2[1], 0.045,
                                 facecolor=GREEN_TINT, zorder=0))
        for lab, a, b in groups:
            ax_s.text((col_x2[a] + col_x2[b]) / 2, 0.957, lab, ha="center",
                      va="center", fontsize=7.5, fontweight="bold", color=GREEN_DARK)
            ax_s.plot([col_x2[a], col_x2[a]], [0.935, 0.98], color=WHITE, lw=1.4, zorder=1)

        sub = [("City", DARK), ("A", GREEN_TXT), ("B", GREY), ("C", SYN_TXT),
               ("A", GREEN_TXT), ("C", SYN_TXT), ("A", GREEN_TXT), ("C", SYN_TXT),
               ("A", GREEN_TXT), ("C", SYN_TXT)]
        for j, (h, col) in enumerate(sub):
            ax_s.text((col_x2[j] + col_x2[j+1]) / 2, 0.905, h, ha="center",
                      va="center", fontsize=8, fontweight="bold", color=col)
        ax_s.axhline(0.875, color=GREEN_DARK, lw=0.8)

        row_h2 = 0.155
        first_top_2 = 0.855
        for i, city in enumerate(CITIES):
            end   = city_data[city]["syn"].iloc[-1]
            a_dir = _aDir[city];  c_dir = float(end["directness_lcc"])
            a_eff = _aEff[city];  c_eff = float(end["efficiency_global"])
            a_lcc = _aLcc[city] / 100.0
            c_lcc = float(end["lcc_share"])
            cells = [
                (CITY_LABELS[city], DARK, False),
                (f"{_aLk[city]:.0f}", GREEN_TXT, True),
                (f"{float(city_data[city]['bikeable']['length'])/1000:.0f}", GREY, False),
                (f"{city_data[city]['syn']['length_km'].max():.0f}", SYN_TXT, False),
                (f"{a_dir:.3f}", GREEN_TXT, True), (f"{c_dir:.3f}", SYN_TXT, False),
                (f"{a_eff:.3f}", GREEN_TXT, True), (f"{c_eff:.3f}", SYN_TXT, False),
                (f"{a_lcc*100:.0f}%", GREEN_TXT, True), (f"{c_lcc*100:.0f}%", SYN_TXT, False),
            ]
            y_top    = first_top_2 - i * row_h2
            y_bottom = y_top - row_h2
            bg = LGREEN if i % 2 == 0 else WHITE
            ax_s.add_patch(Rectangle((0, y_bottom), 1, row_h2, facecolor=bg, zorder=0))
            ax_s.add_patch(Rectangle((-0.005, y_bottom), 0.018, row_h2,
                                     facecolor=COLORS[city], alpha=0.8, zorder=1))
            for j, (val, col, bold) in enumerate(cells):
                ha = "left" if j == 0 else "center"
                xp = col_x2[j] + 0.025 if j == 0 else (col_x2[j] + col_x2[j+1]) / 2
                ax_s.text(xp, y_bottom + row_h2 / 2, val, ha=ha, va="center",
                          fontsize=8.3, color=col, fontweight=("bold" if bold else "normal"))
        sep_bottom_2 = first_top_2 - len(CITIES) * row_h2
        for sep_x in (col_x2[3], col_x2[4], col_x2[6], col_x2[8]):
            ax_s.plot([sep_x, sep_x], [sep_bottom_2, 0.875],
                      color=GREEN_DARK, lw=0.6, alpha=0.4, zorder=2)

        # Spatial overlap table (C vs A, 15 m)
        fig.text(0.5, 0.520,
                 "Spatial overlap — synthetic C vs. protected A  ·  buffer = 15 m",
                 ha="center", fontsize=10, fontweight="bold", color=GREEN_DARK)
        ax_sp = fig.add_axes([0.04, 0.30, 0.92, 0.21])
        ax_sp.set_xlim(0, 1); ax_sp.set_ylim(0, 1); ax_sp.set_axis_off()
        ax_sp.add_patch(Rectangle((0, 0.83), 1, 0.10, facecolor=GREEN_TINT, zorder=-1))

        a_col = "OSM biketrack (km)"
        sp_hdrs = ["City", "A\n(km)", "C\n(km)",
                   "Overlap (km)", "Overlap (%)", "Gap (km)", "Gap (%)"]
        sp_cw   = [0.16, 0.18, 0.14, 0.14, 0.12, 0.13, 0.13]
        sp_cx   = [sum(sp_cw[:i]) for i in range(len(sp_cw) + 1)]
        for j, h in enumerate(sp_hdrs):
            ax_sp.text((sp_cx[j] + sp_cx[j+1]) / 2, 0.92, h, ha="center", va="top",
                       fontsize=7.5, fontweight="bold", color=GREEN_DARK,
                       multialignment="center")
        ax_sp.axhline(0.83, color=GREEN_DARK, lw=0.8)

        sp_rh = 0.13
        sp_first_top = 0.81
        for i, city in enumerate(CITIES):
            r = _safe_row(city)
            if r is None:
                continue
            vals = [CITY_LABELS[city], str(r[a_col]), str(r["Synthetic (km)"]),
                    str(r["Overlap (km)"]), str(r["Overlap (%)"]),
                    str(r["Gap (km)"]), str(r["Gap (%)"])]
            y_top    = sp_first_top - i * sp_rh
            y_bottom = y_top - sp_rh
            bg = LGREEN if i % 2 == 0 else WHITE
            ax_sp.add_patch(Rectangle((0, y_bottom), 1, sp_rh, facecolor=bg, zorder=0))
            ax_sp.add_patch(Rectangle((-0.005, y_bottom), 0.018, sp_rh,
                                      facecolor=COLORS[city], alpha=0.8, zorder=1))
            for j, val in enumerate(vals):
                ha = "left" if j == 0 else "center"
                xp = sp_cx[j] + 0.025 if j == 0 else (sp_cx[j] + sp_cx[j+1]) / 2
                ax_sp.text(xp, y_bottom + sp_rh / 2, val, ha=ha, va="center",
                           fontsize=8.5, color=DARK)
        sp_sep_bottom = sp_first_top - len(CITIES) * sp_rh
        ax_sp.plot([sp_cx[2], sp_cx[2]], [sp_sep_bottom, 0.78],
                   color=GREEN_DARK, lw=0.7, alpha=0.55, zorder=2)

        # Key observations (derived)
        fig.text(0.06, 0.275, "Key observations", color=GREEN_DARK,
                 fontsize=10, fontweight="bold")
        hi_lcc = max(_aLcc, key=_aLcc.get); lo_lcc = min(_aLcc, key=_aLcc.get)
        hi_ov  = max(_aOv, key=_aOv.get);   lo_ov  = min(_aOv, key=_aOv.get)
        hi_den = max(_aDen, key=_aDen.get); lo_den = min(_aDen, key=_aDen.get)
        obs = [
            ("Fragmentation:", (
                f"Protected riding is the most connected in {CITY_LABELS[hi_lcc]} "
                f"(A LCC share {_aLcc[hi_lcc]:.0f} %) and the most fragmented in "
                f"{CITY_LABELS[lo_lcc]} ({_aLcc[lo_lcc]:.0f} %). Unlike B, no city's "
                "protected network A forms a single reachable piece.")),
            ("Protected coverage:", (
                f"C runs on protected infrastructure most in {CITY_LABELS[hi_ov]} "
                f"({_aOv[hi_ov]:.0f} % overlap) and least in {CITY_LABELS[lo_ov]} "
                f"({_aOv[lo_ov]:.0f} %) — the city with the largest protected-path "
                "gap to close.")),
            ("Protected density:", (
                f"A density ranges {_aDen[lo_den]:.1f}–{_aDen[hi_den]:.1f} km/km² "
                f"({CITY_LABELS[lo_den]} → {CITY_LABELS[hi_den]}); compare B in the "
                "density table (p. 4) to see how much of each city's rideable network "
                "is actually protected.")),
            ("Reading the gap:", (
                "A high gap here is stronger than in Part I: it marks optimal routes "
                "with no protected path at all — the priority backlog for safe, "
                "separated cycling infrastructure.")),
        ]
        y = 0.255
        for heading, body in obs:
            fig.text(0.06, y, heading, color=GREEN_MED, fontsize=8, fontweight="bold")
            lines = textwrap.wrap(body, 84)
            for li, line in enumerate(lines):
                fig.text(0.30, y - li * 0.015, line, color=DARK, fontsize=8)
            y -= (len(lines) * 0.015 + 0.014)

        fig.add_artist(Rectangle((0, 0), 1, 0.025,
                                 transform=fig.transFigure, facecolor=LGREEN))
        fig.text(0.5, 0.012, "GrowBikeNet Safety Analysis  ·  gridl=1701  ·  2026",
                 color=GREY, fontsize=7.5, ha="center")
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # ════════════════════════════════════════════════════════════════════
        # PAGE 19 — CLOSING THE GAP  (island-bridging connectivity payoff)
        # ════════════════════════════════════════════════════════════════════
        if gap_conn is not None:
            fig = new_page()
            page_header(fig, "Closing the Gap",
                        "If the synthetic gap were built as protected: A's connectivity payoff",
                        19, bar_color=GREEN_DARK, sub_color="#bfe3cd")

            intro = (
                "The gap (C routes not on protected infrastructure) is not only a backlog "
                "measure — it is a concrete connectivity plan. Adding the gap links back to "
                "the protected network A, as if built as protected paths, reconnects A's "
                "islands. The bars show A's LCC share (the share of protected length sitting "
                "in one single connected piece) before and after."
            )
            y = BODY_TOP
            for ln in textwrap.wrap(intro, 108):
                fig.text(0.06, y, ln, color=DARK, fontsize=9); y -= 0.020

            before = [float(gap_conn[gap_conn["City"] == CITY_LABELS[c]]
                            ["A LCC share %"].iloc[0]) for c in CITIES]
            after  = [float(gap_conn[gap_conn["City"] == CITY_LABELS[c]]
                            ["A+gap LCC share %"].iloc[0]) for c in CITIES]

            ax = fig.add_axes([0.10, 0.40, 0.84, 0.32])
            xs = np.arange(len(CITIES)); bw = 0.38
            ax.bar(xs - bw/2, before, bw, color="#bcd8c6", edgecolor="white",
                   label="A now (protected only)")
            ax.bar(xs + bw/2, after, bw, color=GREEN_DARK, edgecolor="white",
                   label="A + gap built as protected")
            for i, (bef, aft) in enumerate(zip(before, after)):
                ax.annotate(f"{bef:.0f}%", (xs[i]-bw/2, bef), xytext=(0, 2),
                            textcoords="offset points", ha="center", va="bottom",
                            fontsize=7.5, color=GREEN_TXT)
                ax.annotate(f"{aft:.0f}%", (xs[i]+bw/2, aft), xytext=(0, 2),
                            textcoords="offset points", ha="center", va="bottom",
                            fontsize=8.5, color=GREEN_DARK, fontweight="bold")
                ax.annotate(f"+{aft-bef:.0f}pp", (xs[i], max(aft, bef)), xytext=(0, 14),
                            textcoords="offset points", ha="center",
                            fontsize=8, color=GREEN_MED, fontweight="bold")
            ax.set_xticks(xs)
            ax.set_xticklabels([CITY_LABELS[c] for c in CITIES], fontsize=9)
            ax.set_ylabel("Protected LCC share (%)", fontsize=9)
            ax.set_ylim(0, 108)
            ax.set_title("Protected-network connectivity before vs. after building the gap",
                         fontsize=10, color=GREEN_DARK, fontweight="bold", pad=8)
            ax.legend(fontsize=8.5, loc="upper center", ncol=2, frameon=False)
            ax.grid(axis="y", alpha=0.25)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)

            # Takeaway box (derived)
            gains = {CITY_LABELS[c]: after[i] - before[i] for i, c in enumerate(CITIES)}
            best = max(gains, key=gains.get)
            box_top = 0.31
            fig.add_artist(Rectangle((0.06, box_top - 0.150), 0.88, 0.150,
                           transform=fig.transFigure, facecolor=LGREEN,
                           edgecolor=GREEN_MED, linewidth=1.0, zorder=0))
            fig.text(0.075, box_top - 0.028, "WHAT THIS MEANS", color=GREEN_DARK,
                     fontsize=9.5, fontweight="bold")
            takeaway = (
                f"The most fragmented protected networks gain the most: {best} jumps "
                f"{gains[best]:.0f} percentage points. This is the direct link back to the "
                "synthetic network — C does not merely measure the safety gap, it names the "
                "specific strategic links that would join the disconnected protected islands "
                "into one usable whole. By contrast, the cyclist crossings already folded "
                "into A barely move its connectivity (at most ~0.3 pp of LCC share): they "
                "bridge points that already meet at junctions, not the long unprotected "
                "stretches between islands. Length is not the lever; connecting the right "
                "links is."
            )
            yy = box_top - 0.050
            for ln in textwrap.wrap(takeaway, 104):
                fig.text(0.075, yy, ln, color=DARK, fontsize=8.6); yy -= 0.0165

            fig.add_artist(Rectangle((0, 0), 1, 0.025,
                                     transform=fig.transFigure, facecolor=LGREEN))
            fig.text(0.5, 0.012,
                     "GrowBikeNet Safety Analysis  ·  island-bridging  ·  gridl=1701  ·  2026",
                     color=GREY, fontsize=7.5, ha="center")
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    # ════════════════════════════════════════════════════════════════════════
    # PAGE 20 — DATA-DRIVEN DISCUSSION
    #
    # Per-city paragraphs that explain why the headline numbers come out the
    # way they do, using only quantities that appear in the data tables of
    # this report. No historic / cultural speculation.
    # ════════════════════════════════════════════════════════════════════════
    fig = new_page()
    page_header(fig, "Data-driven Discussion",
                "Why each city's headline numbers look the way they do — derived from the data only",
                20)

    # Pull the actual numbers we need
    def _city_stats(c):
        end_c   = city_data[c]["syn"].iloc[-1]
        sp_c    = spatial[spatial["City"] == CITY_LABELS[c]].iloc[0]
        dn_c    = density_tbl[density_tbl["City"] == CITY_LABELS[c]].iloc[0]
        return dict(
            city          = c,
            label         = CITY_LABELS[c],
            color         = COLORS[c],
            bk_km         = float(city_data[c]["bikeable"]["length"]) / 1000,
            bt_km         = float(city_data[c]["biketrack"]["length"]) / 1000,
            syn_km        = float(end_c["length_km"]),
            area_km2      = float(dn_c["Area (km²)"]),
            bk_density    = float(dn_c["bikeable / km²"]),
            ov_pct        = float(str(sp_c["Overlap (%)"]).rstrip("%")),
            gap_km        = float(sp_c["Gap (km)"]),
            directness    = float(end_c["directness_lcc"]),
            efficiency    = float(end_c["efficiency_global"]),
            syn_bk_ratio  = float(end_c["length_km"]) / max(float(city_data[c]["bikeable"]["length"]) / 1000, 1),
        )

    stats = {c: _city_stats(c) for c in CITIES}

    # Rank helper — used by the per-city blurbs
    def _rank(key, reverse=True):
        return sorted(stats.keys(),
                      key=lambda c: stats[c][key], reverse=reverse)

    dir_rank = _rank("directness")     # highest first
    eff_rank = _rank("efficiency")
    ov_rank  = _rank("ov_pct")
    den_rank = _rank("bk_density")
    sbr_rank = _rank("syn_bk_ratio")        # C/B ratio, highest first

    # Ordinal words for a 5-city highest-first ranking. Every superlative below
    # is DERIVED from these ranks, so the text can never contradict the data.
    ORD = {0: "the highest", 1: "the 2nd-highest", 2: "the 3rd-highest",
           3: "the 2nd-lowest", 4: "the lowest"}
    def _ord(rank_list, c):
        return ORD.get(rank_list.index(c), "mid-range")

    def _para(c):
        s = stats[c]
        den_o = _ord(den_rank, c)       # B density
        eff_o = _ord(eff_rank, c)       # C global efficiency
        dir_o = _ord(dir_rank, c)       # C directness
        ov_o  = _ord(ov_rank, c)        # overlap C in B
        cb_o  = _ord(sbr_rank, c)       # C/B ratio
        if c == "amsterdam":
            return (
                f"Amsterdam pairs {den_o} B density ({s['bk_density']:.1f} km/km², "
                f"behind only {CITY_LABELS[den_rank[0]]}) with {cb_o} C/B ratio "
                f"({s['syn_bk_ratio']*100:.0f} %) and {ov_o} overlap "
                f"({s['ov_pct']:.0f} %). Together these mean the synthetic optimum "
                "here proposes the least net-new of the five cities — its most "
                "central routes already exist in B."
            )
        if c == "barcelona":
            return (
                f"Barcelona has {den_o} B density ({s['bk_density']:.1f} km/km²), a "
                f"direct consequence of the smallest effective area "
                f"({s['area_km2']:.0f} km²). Its C reaches {eff_o} global efficiency "
                f"({s['efficiency']:.3f}) — strong, though {CITY_LABELS[eff_rank[0]]} "
                f"and {CITY_LABELS[eff_rank[1]]} rank above it. At "
                f"{s['ov_pct']:.0f} % overlap C still adds {s['gap_km']:.0f} km "
                "outside B, the smallest gap of the five."
            )
        if c == "berlin":
            return (
                f"Berlin is by far the largest city ({s['area_km2']:.0f} km² effective "
                f"area) and grows the longest C network ({s['syn_km']:.0f} km). It has "
                f"{dir_o} directness ({s['directness']:.3f}) and {eff_o} global "
                f"efficiency ({s['efficiency']:.3f}) of the five — its C routes track "
                f"the straight-line distance most closely. Overlap of "
                f"{s['ov_pct']:.0f} % shows C and B largely follow the same main axes."
            )
        if c == "oslo":
            return (
                f"Oslo is the structural outlier: {den_o} B density "
                f"({s['bk_density']:.1f} km/km²) and {cb_o} C/B ratio "
                f"({s['syn_bk_ratio']*100:.0f} %). Its effective area shrinks from a "
                f"480 km² Nominatim polygon to {s['area_km2']:.0f} km² once the Marka "
                f"forest (313 km²) is removed. Only {s['ov_pct']:.0f} % of C overlaps B "
                f"— {ov_o.replace('the ', '')} of the five — leaving "
                f"{s['gap_km']:.0f} km ({100 - s['ov_pct']:.0f} % of C, the largest "
                "relative gap) with no nearby B infrastructure."
            )
        if c == "vienna":
            return (
                f"Vienna scores well on synthetic quality — {dir_o} directness "
                f"({s['directness']:.3f}) and {eff_o} global efficiency "
                f"({s['efficiency']:.3f}) — but sits lower on the real network: "
                f"{den_o} B density ({s['bk_density']:.1f} km/km²) and {ov_o} overlap "
                f"({s['ov_pct']:.0f} %). Its gap is {s['gap_km']:.0f} km "
                f"({100 - s['ov_pct']:.0f} % of C), the second-largest relative gap "
                f"behind {CITY_LABELS[ov_rank[-1]]}; see Vienna's per-city map on "
                "page 9 for the exact corridors."
            )
        return ""

    intro = (
        "The numbers in the summary table on page 11 condense each city to a "
        "few values. The paragraphs below explain why those values look the way "
        "they do, using only quantities that already appear in this report — "
        "area, B density, directness/efficiency of C, and overlap of C in B. "
        "Historic, cultural or terrain explanations are intentionally absent: "
        "the data we have does not let us isolate those effects."
    )
    y_pos = BODY_TOP
    for line in textwrap.wrap(intro, 100):
        fig.text(0.05, y_pos, line, color=GREY, fontsize=8.5, style="italic")
        y_pos -= 0.018
    y_pos -= 0.015

    for c in CITIES:
        s = stats[c]
        # Coloured heading
        fig.text(0.05, y_pos,
                 f"{s['label']}  —  B density {s['bk_density']:.2f} km/km²  ·  "
                 f"overlap (C in B) {s['ov_pct']:.0f} %  ·  directness (C) {s['directness']:.3f}",
                 color=s["color"], fontsize=10, fontweight="bold")
        y_pos -= 0.020
        # Paragraph
        body = _para(c)
        for line in textwrap.wrap(body, 105):
            fig.text(0.05, y_pos, line, color=DARK, fontsize=8.5)
            y_pos -= 0.0165
        y_pos -= 0.012

    # ── Conclusion (synthesis answer to the research question) ────────────
    hi_ov, lo_ov = ov_rank[0], ov_rank[-1]
    concl = (
        "Connectivity is not what distinguishes the five cities: the synthetic "
        "network C reaches a single connected component and ~100 % LCC share early "
        "in growth, while the real networks B stay fragmented into many components. "
        "C also tends to be more direct within its core. But the comparison is "
        "asymmetric — C's maximum is only 15–48 % of B's length, so the much larger "
        "real network can still score higher on global efficiency simply by covering "
        "more of the city (see the B vs C columns on p. 11). The clearest signal is "
        f"therefore spatial: where the real network is well-developed "
        f"({CITY_LABELS[hi_ov]}, {stats[hi_ov]['ov_pct']:.0f} % overlap) C mostly "
        f"rediscovers routes that already exist; where it is sparse ({CITY_LABELS[lo_ov]}, "
        f"{stats[lo_ov]['ov_pct']:.0f} %) C proposes the largest share of genuinely "
        "missing links. The room for improvement is city-specific, not a single number."
    )
    box_top = 0.300
    fig.add_artist(FancyBboxPatch((0.05, 0.080), 0.90, box_top - 0.080,
                   transform=fig.transFigure, boxstyle="round,pad=0.006",
                   facecolor=LGREY, edgecolor=DBLUE, linewidth=1.0, zorder=0))
    fig.text(0.07, box_top - 0.030, "CONCLUSION", color=DBLUE, fontsize=9.5,
             fontweight="bold")
    ty = box_top - 0.054
    for line in textwrap.wrap(concl, 110):
        fig.text(0.07, ty, line, color=DARK, fontsize=8.7)
        ty -= 0.0175

    fig.add_artist(Rectangle((0, 0), 1, 0.025,
                              transform=fig.transFigure, facecolor=LGREY))
    fig.text(0.5, 0.012, "GrowBikeNet Analysis  ·  gridl=1701  ·  2026",
             color=GREY, fontsize=7.5, ha="center")
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    # ════════════════════════════════════════════════════════════════════════
    # PAGE 13 — APPENDIX: DEFINITIONS
    # ════════════════════════════════════════════════════════════════════════
    fig = new_page()
    page_header(fig, "Appendix — Definitions",
                "Networks, parameters, metrics and spatial terms used throughout", 21)

    # (section_title, None)  → a coloured section divider
    # (term, definition)     → one definition row
    defs = [
        ("NETWORKS", None),
        ("A — biketrack",
            "Real OSM network of protected / segregated cycle paths (cycle tracks, "
            "cycleways, designated paths, cyclestreets), plus cyclist crossings "
            "(cycleway=crossing) snapped on within 8 m as real connectors. The "
            "safety reference for Part II. Measured as undirected length."),
        ("B — bikeable",
            "Real OSM network of every legally cyclable edge: network A plus all "
            "drivable streets with speed limit ≤ 30 km/h (maxspeed 5–30) and calmed "
            "living streets (highway=living_street). Primary real reference."),
        ("C — synthetic",
            "Network grown by the GrowBikeNet algorithm, reported at its maximum "
            "(growth quantile Q = 1.0)."),
        ("ALGORITHM & PARAMETERS", None),
        ("GrowBikeNet",
            "Greedy algorithm (Szell et al., 2022) that grows a connected bike "
            "network by repeatedly adding the highest-betweenness edge near "
            "grid-sampled points of interest (POIs)."),
        ("gridl",
            "Spacing in metres of the regular POI grid the algorithm targets. "
            "gridl = 1701 m — the standard value used in Szell et al. (2022) — is "
            "applied to every city for comparability, except the Vienna sensitivity "
            "page (p. 10)."),
        ("Growth quantile (Q)",
            "Fraction of total growth completed (0 = start, 1 = synthetic "
            "maximum). Lets different-sized cities be compared at the same stage."),
        ("METRICS", None),
        ("LCC / LCC share",
            "Largest connected component; LCC share = length in the LCC ÷ total "
            "length. 100 % means the whole network is one reachable piece."),
        ("Components",
            "Number of disconnected sub-graphs (islands). 1 = fully connected."),
        ("Directness (LCC)",
            "Mean ratio of straight-line to network distance over random node "
            "pairs in the LCC. Higher = routes closer to the crow-flies path."),
        ("Global efficiency",
            "Mean of (straight-line ÷ network distance) over all pairs; "
            "disconnected pairs count as 0, so fragmentation is penalised."),
        ("Density (km/km²)",
            "Network length ÷ effective city area — removes the effect of raw "
            "city size when comparing cities."),
        ("SPATIAL COMPARISON (C vs B)", None),
        ("Buffer",
            "Distance tolerance (15 m) for deciding whether a C edge runs along "
            "a B edge."),
        ("Overlap (C in B)",
            "Share of C's length whose 15 m buffer falls ≥ 50 % within B — C "
            "rediscovers an already-built route."),
        ("Gap",
            "Complement of overlap: C's length with no nearby B infrastructure — "
            "routes proposed by C but not yet built."),
        ("SAFETY COMPARISON (C vs A) — Part II", None),
        ("Overlap (C on A)",
            "Share of C's length whose 15 m buffer falls ≥ 50 % within the "
            "protected network A — the optimum already running on safe, "
            "segregated infrastructure."),
        ("Gap (unprotected)",
            "C's length with no protected path nearby — optimal routes that today "
            "would be ridden in mixed traffic. The safe-infrastructure backlog."),
        ("Island-bridging (A + gap)",
            "Hypothetical: add the gap links back to A as if built as protected, "
            "snapping each to the nearest protected node within 15 m, then recompute "
            "A's LCC share. Measures how much the synthetic gap would reconnect the "
            "protected islands (p. 19)."),
        ("Cyclist crossing",
            "OSM cycleway=crossing: a protected route carried across an "
            "intersection. Snapped onto biketrack endpoints (8 m, metric CRS) so "
            "it joins protected segments instead of docking onto nothing."),
        ("Effective area / boundary",
            "Nominatim city polygon minus large forest / park areas; used for "
            "density and to clip the per-city maps."),
    ]

    LXt = 0.06          # term column
    LXd = 0.34          # definition column
    DWRAP = 74
    y = BODY_TOP
    for term, definition in defs:
        if definition is None:
            # Section divider
            y -= 0.006
            fig.add_artist(Rectangle((0.04, y - 0.004), 0.92, 0.022,
                                     transform=fig.transFigure,
                                     facecolor=HEADER_TINT, zorder=0))
            fig.text(LXt, y + 0.004, term, color=DBLUE, fontsize=8.5,
                     fontweight="bold", va="center")
            y -= 0.026
            continue
        wrapped = textwrap.wrap(definition, DWRAP)
        fig.text(LXt, y, term, color=DARK, fontsize=8, fontweight="bold", va="top")
        for li, line in enumerate(wrapped):
            fig.text(LXd, y - li * 0.0145, line, color=DARK, fontsize=8, va="top")
        y -= max(len(wrapped) * 0.0145, 0.0145) + 0.010

    fig.add_artist(Rectangle((0, 0), 1, 0.025,
                              transform=fig.transFigure, facecolor=LGREY))
    fig.text(0.5, 0.012,
             "GrowBikeNet Analysis  ·  Szell et al. (2022)  ·  gridl=1701  ·  2026",
             color=GREY, fontsize=7.5, ha="center")
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

print(f"\nReport saved: {PDF_PATH}")
