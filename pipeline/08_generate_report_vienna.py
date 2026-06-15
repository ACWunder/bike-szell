"""
Vienna Bicycle Network Report v4 — final
v4 changes vs v3:
  · "Preliminary results" banner removed (the analysis is now final)
  · Cover updated to May 2026
  · NEW p. 8 "Vienna in cross-city context" links to report_multicity.pdf
  · NEW p. 9 "Gap hotspots — Vienna OGD" DBSCAN on the 429 km Variant-C gap
Env: growbikenet  |  Run: conda run -n growbikenet python generate_report_v4.py
"""
from pathlib import Path
import textwrap, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import geopandas as gpd

# ── Paths ──────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent / "bikenwgrowth-data"
OUT  = BASE / "analysis_output"
PDF  = OUT / "report_vienna_v4.pdf"
SYN  = BASE / "results/vienna_2/vienna_poi_grid_betweenness.csv"

syn = pd.read_csv(SYN)
syn["length_km"] = syn["length"] / 1000
syn["lcc_share"] = syn["length_lcc"] / syn["length"]
n = len(syn)
syn["quantile"] = [round(i/n, 3) for i in range(1, n+1)]

# Variant C (real) values
C = dict(
    length=1810, components=546, lcc=0.894,
    directness=0.690, efficiency=0.557,
    coverage=90.7, overlap=56.7, gap=43.3,
    overlap_km=560.9, gap_km=429.1,
)
SYN_MAX = dict(
    length=989, components=1, lcc=1.0,
    directness=0.745, efficiency=0.653, coverage=89.3,
)
# Network A — safe / separated infrastructure (OGD "Getrennte Führung")
A_REAL = dict(length=859, components=2360, lcc=0.18, coverage=80.0)

# ── Colours ────────────────────────────────────────────────────────────────
DBLUE  = "#0d3d6e"
BLUE   = "#1a6faf"
TEAL   = "#0eb6d2"
GREEN  = "#27ae60"
RED    = "#c0392b"
ORANGE = "#e67e22"
DARK   = "#1c1c2e"
GREY   = "#555"
LGREY  = "#f0f4f8"
WHITE  = "#ffffff"
LBLUE  = "#a8c8e8"

plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          9,
    "text.color":         DARK,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
})

PW, PH = 8.27, 11.69   # A4 in inches


def new_page():
    fig = plt.figure(figsize=(PW, PH), dpi=300)
    fig.patch.set_facecolor(WHITE)
    return fig


def hdr(fig, title, subtitle, pn):
    fig.add_artist(Rectangle((0, 0.905), 1, 0.095,
                              transform=fig.transFigure,
                              facecolor=DBLUE, zorder=10))
    fig.text(0.025, 0.957, title, color=WHITE, fontsize=19,
             fontweight="bold", va="center", zorder=11)
    fig.text(0.025, 0.918, subtitle, color=LBLUE, fontsize=8.5,
             va="center", zorder=11)
    fig.text(0.975, 0.937, str(pn), color=LBLUE, fontsize=9,
             ha="right", va="center", zorder=11)


def txt(fig, x, y, text, size=9, bold=False, color=DARK, width=52):
    """Render wrapped paragraph; returns new y."""
    lh = (size + 2.5) / (PH * 72)
    for line in textwrap.wrap(text, width):
        fig.text(x, y, line, fontsize=size, color=color,
                 fontweight="bold" if bold else "normal", va="top")
        y -= lh
    return y - lh * 0.5


def section_title(fig, x, y, text):
    """Blue section label."""
    fig.text(x, y, text.upper(), fontsize=8, fontweight="bold",
             color=BLUE, va="top")
    y2 = y - 9 / (PH * 72)
    ax = fig.add_axes([x, y2, 0.42, 0.0015])
    ax.set_facecolor(BLUE); ax.axis("off")
    return y2 - 0.018


def bullet(fig, x, y, text, width=49):
    lh = 11 / (PH * 72)
    lines = textwrap.wrap(text, width)
    fig.text(x, y, "▸", fontsize=8, color=BLUE, va="top")
    for i, line in enumerate(lines):
        fig.text(x + 0.032, y - i * lh, line, fontsize=8.5, color=DARK, va="top")
    return y - lh * len(lines) - lh * 0.4


def stat_box(fig, lx, by, w, h, value, label, bg=BLUE):
    """Coloured stat box — drawn directly on figure to avoid axes-clipping bugs."""
    fig.add_artist(Rectangle(
        (lx, by), w, h,
        transform=fig.transFigure,
        facecolor=bg, zorder=4, clip_on=False
    ))
    # Value — upper portion
    fig.text(lx + w / 2, by + h * 0.62, value,
             ha="center", va="center",
             fontsize=16, fontweight="bold", color=WHITE, zorder=5)
    # Label — lower portion
    fig.text(lx + w / 2, by + h * 0.22, label,
             ha="center", va="center",
             fontsize=7.5, color=WHITE, zorder=5)


def note_box(fig, x, y, w, text, bg="#e8f4fd", fg="#1a3a5c"):
    """Render a tinted info box; returns new y below the box."""
    char_width = int(w * 130)
    lines  = textwrap.wrap(text, char_width)
    lh_pts = 10
    pad    = 0.012
    h      = pad * 2 + lh_pts / (PH * 72) * len(lines) * 1.35
    ax     = fig.add_axes([x, y - h, w, h])
    ax.set_facecolor(bg); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    lh_f = (lh_pts * 1.35) / (h * PH * 72)
    ty   = 1 - pad / h
    for line in lines:
        ax.text(0.015, ty, line, fontsize=8.5, color=fg, va="top",
                transform=ax.transAxes)
        ty -= lh_f
    return y - h - 0.015


def show_img(fig, lx, by, w, h, path, caption=""):
    ax = fig.add_axes([lx, by, w, h])
    try:
        ax.imshow(mpimg.imread(str(path)))
    except FileNotFoundError:
        ax.set_facecolor(LGREY)
        ax.text(0.5, 0.5, f"[Map not found:\n{Path(path).name}]",
                ha="center", va="center", fontsize=9, color=GREY,
                transform=ax.transAxes)
    ax.axis("off")
    if caption:
        fig.text(lx + w / 2, by - 0.018, caption,
                 ha="center", fontsize=7.5, color=GREY, style="italic")


def two_bar(fig, rect, real_val, syn_val, metric_name, fmt="{:.3f}",
            real_label="Real network (B)", syn_label="Synthetic optimum (C)",
            maxv_override=None):
    """Two-row horizontal bar. Uses a wider left margin for tick labels."""
    lx, by, rw, rh = rect
    ax = fig.add_axes([lx, by, rw, rh])
    ax.set_facecolor(LGREY)
    maxv = maxv_override if maxv_override else max(real_val, syn_val) * 1.35
    bar_h = 0.42
    ax.barh([1], [real_val], bar_h, color=TEAL,  zorder=3)
    ax.barh([0], [syn_val],  bar_h, color=DBLUE, zorder=3)
    ax.set_yticks([0, 1])
    ax.set_yticklabels([syn_label, real_label], fontsize=8.5)
    ax.yaxis.set_tick_params(length=0, pad=6)
    # Push tick labels inside the axes so they don't bleed left
    ax.tick_params(axis='y', direction='in', pad=-6)
    ax.set_xlim(0, maxv)
    ax.set_xlabel(metric_name, fontsize=9)
    ax.grid(axis="x", alpha=0.4, color=WHITE, zorder=2)
    for bh_pos, v in zip([1, 0], [real_val, syn_val]):
        ax.text(v + maxv * 0.02, bh_pos, fmt.format(v),
                va="center", fontsize=11, fontweight="bold", color=DARK)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    return ax


# ══════════════════════════════════════════════════════════════════════════
# PAGE 1 — TITLE
# ══════════════════════════════════════════════════════════════════════════
def page_title(pdf):
    fig = new_page()
    fig.patch.set_facecolor(DARK)

    ax = fig.add_axes([0, 0.36, 1, 0.44])
    ax.set_facecolor(BLUE); ax.axis("off")

    fig.text(0.5, 0.75, "Vienna Bicycle Network",
             ha="center", fontsize=28, fontweight="bold", color=WHITE)
    fig.text(0.5, 0.68, "Structural Analysis — Real vs. Synthetic Optimum",
             ha="center", fontsize=15, color=LBLUE)
    fig.text(0.5, 0.60, "bikenwgrowth algorithm  ·  Szell et al. (2022)",
             ha="center", fontsize=10.5, color="#8899bb", style="italic")
    fig.text(0.5, 0.48, "Arthur Wunder  ·  Interdisciplinary Project  ·  May 2026",
             ha="center", fontsize=10, color="#ccccdd")

    fig.text(0.5, 0.41,
             "Data: official City of Vienna cycle-network data (OGD) — more accurate "
             "than OSM for the focus city.",
             ha="center", fontsize=8.5, color="#8899bb", style="italic")

    # Reference box pointing to the multicity companion report
    fig.add_artist(Rectangle(
        (0.12, 0.18), 0.76, 0.12,
        transform=fig.transFigure,
        facecolor="#0a2742", zorder=4, clip_on=False
    ))
    fig.text(0.5, 0.272, "Companion report",
             ha="center", va="center", fontsize=10, fontweight="bold",
             color="#a8c8e8", zorder=5)
    fig.text(0.5, 0.222,
             "Vienna is also analysed in an international cross-city context\n"
             "(Amsterdam, Barcelona, Berlin, Oslo, Vienna) in report_multicity.pdf.",
             ha="center", va="center", fontsize=9, color="#dde6ee",
             linespacing=1.6, zorder=5)

    fig.text(0.5, 0.10,
             "Fragmentation  ·  Global efficiency  ·  Directness  ·  Coverage  ·  Overlap & Gap",
             ha="center", fontsize=9, color="#7788aa")

    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# PAGE 2 — INTRO
# ══════════════════════════════════════════════════════════════════════════
def page_intro(pdf):
    fig = new_page()
    hdr(fig, "Introduction & Methodology",
        "Algorithm · data sources · networks A / B / C  (A, B real · C synthetic)", 2)

    lh9 = 11.5 / (PH * 72)   # line-height for fontsize 9

    # ── TOP HALF: Research question + Data + Variants ────────────────────
    y = 0.875
    y = section_title(fig, 0.05, y, "Research question")
    y = txt(fig, 0.05, y,
            "How does Vienna's real bicycle network compare structurally to a "
            "theoretically optimal network? The bikenwgrowth algorithm (Szell et al., 2022) "
            "grows a fully connected synthetic network over 40 steps using greedy "
            "triangulation and betweenness-centrality pruning.", width=90)
    y -= 0.010

    y = section_title(fig, 0.05, y, "Data")
    for k, v in [
        ("Real network:", "Radwege_ogd.csv  —  Vienna OGD  —  14,777 segments"),
        ("Synthetic:",    "vienna_poi_grid_betweenness  —  40 growth stages (Q 0.025 to 1.0)"),
        ("Projection:",   "EPSG:31256 (MGI Austria GK East)  —  metric distances"),
    ]:
        fig.text(0.05, y, k, fontsize=9, fontweight="bold", color=DARK,
                 style="italic", va="top")
        fig.text(0.24, y, v, fontsize=9, color=DARK, va="top")
        y -= lh9 * 1.8
    y -= 0.010

    y = section_title(fig, 0.05, y, "The three networks — referred to as A / B / C")
    vcols = [0.05, 0.34, 0.46, 0.60]
    for vname, km, comp, eff, bold in [
        ("A – dedicated tracks (real)", "859 km",   "2,360 comp.", "GE = 0.018", False),
        ("B – full real network",       "1,810 km", "546 comp.",   "GE = 0.557", True),
        ("C – synthetic optimum",       "989 km",   "1 comp.",     "GE = 0.653", False),
    ]:
        col = TEAL if bold else GREY
        fw  = "bold" if bold else "normal"
        for cx, val in zip(vcols, [vname, km, comp, eff]):
            fig.text(cx, y, val, fontsize=9, fontweight=fw, color=col, va="top")
        y -= lh9 * 2.0
    y -= 0.006

    # Note box
    note_box(fig, 0.05, y, 0.90,
             "A and B are real (Vienna OGD); C is synthetic. B — the full rideable "
             "network and primary reference — combines dedicated tracks, marked lanes "
             "and cyclable streets (859 + 596 + … = 1,810 km). A (dedicated tracks "
             "alone) is shown for reference: fragmented and near-zero on most metrics. "
             "All detailed pages compare B against C (synthetic at Q = 1.0).",
             bg="#e8f4fd", fg="#1a3a5c")

    # ── BOTTOM HALF: "B vs C at a glance" table — FIXED absolute y ───
    # Pin the table to a fixed position regardless of what happened above.
    TABLE_TOP = 0.355   # top of section label — always visible

    fig.text(0.05, TABLE_TOP, "B (REAL) vs C (SYNTHETIC) AT A GLANCE",
             fontsize=8, fontweight="bold", color=BLUE, va="top")

    t_x   = 0.05
    t_w   = 0.92
    row_h = 0.038
    # Column centres in absolute figure-fraction
    col_cx = [0.185, 0.455, 0.625, 0.810]
    hdrs   = ["Metric", "B (real)", "C (synthetic)", "Gap"]
    rows   = [
        ("Global efficiency",      "0.557",  "0.653",  "−14.6 %"),
        ("Directness (LCC)",       "0.690",  "0.745",  "−7.4 %"),
        ("LCC share",              "89.4 %", "100 %",  "−10.6 pp"),
        ("Spatial coverage",       "90.7 %", "89.3 %", "+1.4 pp"),
        ("Overlap (C in B)",       "56.7 %", "—",      "43.3 % gap"),
    ]

    ty = TABLE_TOP - 0.022   # top of header row

    # Header background
    fig.add_artist(
        Rectangle((t_x, ty - row_h), t_w, row_h,
                  transform=fig.transFigure,
                  facecolor=DBLUE, zorder=5)
    )
    # Header labels — directly on figure, always on top of rectangle
    hdr_y = ty - row_h / 2
    for cx, h in zip(col_cx, hdrs):
        fig.text(cx, hdr_y, h, ha="center", va="center",
                 fontsize=9, fontweight="bold", color=WHITE, zorder=6)

    # Data rows
    ry = ty - row_h
    for ri, (m, rv, sv, g) in enumerate(rows):
        bg = LGREY if ri % 2 == 0 else WHITE
        fig.add_artist(
            Rectangle((t_x, ry - row_h), t_w, row_h,
                      transform=fig.transFigure,
                      facecolor=bg, zorder=5)
        )
        mid_y = ry - row_h / 2
        for cx, val in zip(col_cx, [m, rv, sv, g]):
            if val == m:
                fc = DARK
            elif "−" in val:
                fc = RED
            elif "+" in val:
                fc = GREEN
            else:
                fc = DARK
            fig.text(cx, mid_y, val, ha="center", va="center",
                     fontsize=8.5, color=fc, zorder=6)
        ry -= row_h

    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# PAGE 3 — FRAGMENTATION MAP
# ══════════════════════════════════════════════════════════════════════════
def page_fragmentation_map(pdf):
    fig = new_page()
    hdr(fig, "Metric 1 — Fragmentation & Connectivity",
        "Number of connected components and LCC share  ·  network B (real)", 3)

    # ── Left strip ────────────────────────────────────────────────────
    lx, sw = 0.03, 0.24

    y = 0.877
    y = section_title(fig, lx, y, "What this measures")
    y = txt(fig, lx, y,
            "Number of disconnected subgraphs (components) and the share of "
            "total network length in the largest connected component (LCC). "
            "A fully connected network has exactly 1 component and 100 % LCC "
            "share; high component counts indicate isolated islands that a "
            "cyclist cannot reach from the main network.",
            width=30)
    y -= 0.008

    y = section_title(fig, lx, y, "Result — network B (real)")

    BOX_H = 0.068
    BOX_GAP = 0.008
    for val, lbl, bg in [
        ("546",     "Connected components", TEAL),
        ("89.4 %",  "LCC share",            GREEN),
        ("1,810 km","Total length",          BLUE),
    ]:
        stat_box(fig, lx, y - BOX_H, sw, BOX_H, val, lbl, bg)
        y -= BOX_H + BOX_GAP
    y -= 0.006

    y = section_title(fig, lx, y, "Interpretation")
    y = txt(fig, lx, y,
            "Despite 546 separate components, the full network is "
            "functionally connected: 89.4 % of total length belongs "
            "to one giant connected component (LCC).",
            width=30)
    y = txt(fig, lx, y,
            "The synthetic network is always a single component — "
            "connectivity is a design principle of the algorithm.",
            width=30)
    y = note_box(fig, lx, y, sw + 0.01,
                 "Network A (dedicated tracks): 2,360 components, only 18 % LCC share. "
                 "Network B's connectivity emerges from combining all "
                 "infrastructure types.",
                 bg="#e8f4fd", fg="#1a3a5c")

    # ── Map ───────────────────────────────────────────────────────────
    show_img(fig, 0.30, 0.07, 0.68, 0.82, OUT / "vienna_ogd" / "frag_C.png",
             "Fragmentation map — network B (real). Dark blue = LCC. "
             "Each colour = one isolated component.")

    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# PAGE 4 — GLOBAL EFFICIENCY
# ══════════════════════════════════════════════════════════════════════════
def page_efficiency(pdf):
    fig = new_page()
    hdr(fig, "Metric 3 — Global Efficiency",
        "Mean inverse shortest-path over all node pairs · Disconnected pairs = 0  ·  network B (real)", 5)

    lx = 0.05

    y = 0.87
    y = section_title(fig, lx, y, "What this measures")
    y = txt(fig, lx, y,
            "Global efficiency (GE) is the mean inverse shortest-path "
            "over all node pairs in the network. "
            "Disconnected pairs score 0, making it sensitive to both "
            "fragmentation and routing quality in one number.", width=46)
    y = txt(fig, lx, y,
            "Scale: 0 = completely disconnected, 1 = perfect star network.", width=46)
    y -= 0.012

    y = section_title(fig, lx, y, "Result — network B (real)")

    bw, bh, bg_gap = 0.18, 0.072, 0.010
    stat_box(fig, lx,                 y - bh, bw, bh, "0.557",   "Real network (B)", TEAL)
    stat_box(fig, lx + bw + bg_gap,   y - bh, bw, bh, "0.653",   "Synthetic (C)",   DBLUE)
    stat_box(fig, lx, y - bh*2 - bg_gap*2, bw*2 + bg_gap, bh,
             "−14.6 %", "Gap to synthetic optimum", RED)
    y -= bh * 2 + bg_gap * 3 + 0.012

    y = section_title(fig, lx, y, "Interpretation")
    y = txt(fig, lx, y,
            "Network B is 14.6 % below the synthetic optimum (C) — "
            "a meaningful but bridgeable gap. The real network is functional "
            "and fairly well connected, but the algorithm's triangulated "
            "structure eliminates redundant detours more effectively.", width=46)
    y = note_box(fig, lx, y, 0.44,
                 "For reference: dedicated tracks alone (A) reach GE = 0.018 — "
                 "the full network (B) is ~31x more efficient.",
                 bg="#fff8e1", fg="#5d3000")

    # Right column — growth curve, full height
    ax = fig.add_axes([0.54, 0.38, 0.42, 0.40])
    ax.set_facecolor(LGREY)
    ax.fill_between(syn["length_km"], syn["efficiency_global"], alpha=0.12, color=BLUE)
    ax.plot(syn["length_km"], syn["efficiency_global"],
            color=BLUE, lw=2, label="Synthetic growth curve")
    ax.axvline(C["length"], color=TEAL, lw=2, ls="--",
               label=f"Real B: {C['length']:.0f} km")
    ax.scatter([C["length"]], [C["efficiency"]], color=TEAL, s=80, zorder=5)
    ax.scatter([SYN_MAX["length"]], [SYN_MAX["efficiency"]],
               color=DBLUE, s=80, zorder=5, marker="D",
               label=f"Syn. max: {SYN_MAX['length']:.0f} km")
    ax.set_xlabel("Network length (km)", fontsize=9)
    ax.set_ylabel("Global efficiency", fontsize=9)
    ax.set_title("GE along synthetic growth — real B marked", fontsize=9, pad=5)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.35, color=WHITE)

    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# PAGE 5 — DIRECTNESS
# ══════════════════════════════════════════════════════════════════════════
def page_directness(pdf):
    fig = new_page()
    hdr(fig, "Metric 4 — Directness",
        "Mean ratio of straight-line to network distance within the LCC  ·  network B (real)", 6)

    lx = 0.05

    y = 0.87
    y = section_title(fig, lx, y, "What this measures")
    y = txt(fig, lx, y,
            "Directness is the mean ratio of straight-line distance to actual "
            "network distance, sampled over 1,000 random node pairs within the "
            "largest connected component (LCC). "
            "A value of 1.0 means zero detour; 0.69 means routes are "
            "roughly 45 % longer than the straight-line distance.", width=46)
    y -= 0.012

    y = section_title(fig, lx, y, "Result — network B (real)")

    bw, bh, bg_gap = 0.18, 0.072, 0.010
    stat_box(fig, lx,                 y - bh, bw, bh, "0.690", "Real network (B)", TEAL)
    stat_box(fig, lx + bw + bg_gap,   y - bh, bw, bh, "0.745", "Synthetic (C)",   DBLUE)
    stat_box(fig, lx, y - bh*2 - bg_gap*2, bw*2 + bg_gap, bh,
             "−7.4 %", "Gap to synthetic optimum", ORANGE)
    y -= bh * 2 + bg_gap * 3 + 0.012

    y = section_title(fig, lx, y, "Interpretation")
    y = txt(fig, lx, y,
            "The 7.4 % directness gap is the smallest of all metrics — "
            "once you are within the LCC, routes are quite direct. "
            "The synthetic algorithm achieves better directness through "
            "its triangulated structure, which minimises unnecessary bends.", width=46)
    y = txt(fig, lx, y,
            "Directness complements global efficiency: GE captures whether "
            "a path exists at all; directness measures how good it is.", width=46)

    # Right column — growth curve, full height
    ax = fig.add_axes([0.54, 0.38, 0.42, 0.40])
    ax.set_facecolor(LGREY)
    ax.fill_between(syn["length_km"], syn["directness_lcc"], alpha=0.12, color=BLUE)
    ax.plot(syn["length_km"], syn["directness_lcc"],
            color=BLUE, lw=2, label="Synthetic growth curve")
    ax.axvline(C["length"], color=TEAL, lw=2, ls="--")
    ax.scatter([C["length"]], [C["directness"]],
               color=TEAL, s=80, zorder=5, label=f"Real B: {C['directness']:.3f}")
    ax.scatter([SYN_MAX["length"]], [SYN_MAX["directness"]],
               color=DBLUE, s=80, zorder=5, marker="D",
               label=f"Syn. max: {SYN_MAX['directness']:.3f}")
    ax.set_xlabel("Network length (km)", fontsize=9)
    ax.set_ylabel("Directness (LCC)", fontsize=9)
    ax.set_title("Directness along synthetic growth — real B marked", fontsize=9, pad=5)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.35, color=WHITE)

    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# PAGE 6 — SPATIAL COVERAGE
# ══════════════════════════════════════════════════════════════════════════
def page_coverage(pdf):
    fig = new_page()
    hdr(fig, "Metric 5 — Spatial Coverage",
        "Area within 500 m buffer of network, clipped to Vienna boundary (414.9 km²)  ·  network B (real)", 7)

    lx = 0.05

    y = 0.87
    y = section_title(fig, lx, y, "What this measures")
    y = txt(fig, lx, y,
            "Coverage is the area (km²) within a 500 m buffer of the network, "
            "clipped to Vienna's city boundary (414.9 km²). "
            "500 m corresponds to roughly a 5-minute walk — it answers: "
            "can I reach a cycle path from here?", width=46)
    y -= 0.012

    y = section_title(fig, lx, y, "Result — network B (real)")

    bw, bh, bg_gap = 0.18, 0.072, 0.010
    stat_box(fig, lx,                 y - bh, bw, bh, "90.7 %", "Real network (B)", TEAL)
    stat_box(fig, lx + bw + bg_gap,   y - bh, bw, bh, "89.3 %", "Synthetic (C)",   DBLUE)
    stat_box(fig, lx, y - bh*2 - bg_gap*2, bw*2 + bg_gap, bh,
             "+1.4 pp", "Real network exceeds synthetic!", GREEN)
    y -= bh * 2 + bg_gap * 3 + 0.012

    y = section_title(fig, lx, y, "Interpretation")
    y = txt(fig, lx, y,
            "Coverage is the one metric where the real network outperforms "
            "the synthetic optimum. At 90.7 % vs. 89.3 %, Vienna's network is "
            "excellently distributed spatially — nearly the entire city area "
            "is within walking distance of a cycle path.", width=46)
    y = note_box(fig, lx, y, 0.44,
                 "Coverage is necessary but not sufficient: a path exists "
                 "nearly everywhere, but is it connected? "
                 "Good coverage combined with 89 % LCC share shows "
                 "Network B is strong on both distribution and connectivity.",
                 bg="#d4edda", fg="#155724")

    # Right column — growth curve, full height
    ax = fig.add_axes([0.54, 0.38, 0.42, 0.40])
    ax.set_facecolor(LGREY)
    cov_pct = syn["coverage"] / 4.149
    ax.fill_between(syn["length_km"], cov_pct, alpha=0.12, color=BLUE)
    ax.plot(syn["length_km"], cov_pct, color=BLUE, lw=2, label="Synthetic growth curve")
    ax.axvline(C["length"], color=TEAL, lw=2, ls="--")
    ax.scatter([C["length"]], [C["coverage"]],
               color=TEAL, s=80, zorder=5, label=f"Real B: {C['coverage']:.1f} %")
    ax.scatter([SYN_MAX["length"]], [SYN_MAX["coverage"]],
               color=DBLUE, s=80, zorder=5, marker="D",
               label=f"Syn. max: {SYN_MAX['coverage']:.1f} %")
    ax.axhline(100, color=GREY, lw=0.8, ls=":", alpha=0.5)
    ax.set_xlabel("Network length (km)", fontsize=9)
    ax.set_ylabel("Coverage (% of Vienna)", fontsize=9)
    ax.set_title("Coverage along synthetic growth — real B marked", fontsize=9, pad=5)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.35, color=WHITE)

    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# PAGE 7 — OVERLAP & GAP (donut)
# ══════════════════════════════════════════════════════════════════════════
def page_overlap_analysis(pdf):
    fig = new_page()
    hdr(fig, "Metric 2 — Spatial Overlap & Gap",
        "Share of synthetic network within 15 m of real network (overlap) vs. missing (gap)  ·  network B (real)", 4)

    lx = 0.05

    y = 0.87
    y = section_title(fig, lx, y, "Method")
    y = txt(fig, lx, y,
            "For each synthetic edge it is checked whether it falls within 15 m "
            "of the real network. Edges with at least 50 % spatial proximity are "
            "classified as 'overlap' (already built); the rest as 'gap'.", width=46)
    y -= 0.012

    y = section_title(fig, lx, y, "Result — network B (real)")

    bw, bh, bg_gap = 0.18, 0.072, 0.010
    stat_box(fig, lx,                  y - bh, bw,              bh, "56.7 %",  "Overlap",              GREEN)
    stat_box(fig, lx + bw + bg_gap,    y - bh, bw,              bh, "43.3 %",  "Gap",                  RED)
    stat_box(fig, lx, y - bh*2 - bg_gap*2, bw*2 + bg_gap,      bh, "429 km",  "Missing connections",  RED)
    y -= bh * 2 + bg_gap * 3 + 0.012

    y = section_title(fig, lx, y, "Interpretation")
    y = txt(fig, lx, y,
            "More than half of the algorithmically optimal network "
            "is already present in Vienna. The remaining 43 % (429 km) "
            "are the connections the algorithm prioritises most but that "
            "have not yet been built.", width=46)
    y = txt(fig, lx, y,
            "These gap segments represent the highest theoretical efficiency "
            "gain per kilometre of new infrastructure.", width=46)
    y = note_box(fig, lx, y, 0.44,
                 "Gap GeoJSON exported for QGIS: gap_C_full.geojson — "
                 "can serve directly as spatial planning input.",
                 bg="#e8f4fd", fg="#1a3a5c")

    # ── Donut chart — fixed clipping by using a dedicated axes ──────────
    ax = fig.add_axes([0.52, 0.18, 0.44, 0.60])
    sizes  = [C["overlap"], C["gap"]]
    colors = [GREEN, RED]

    wedges, _, autotexts = ax.pie(
        sizes, colors=colors, startangle=90,
        autopct="%1.1f%%", pctdistance=0.72,
        wedgeprops={"width": 0.46, "edgecolor": WHITE, "linewidth": 3},
        textprops={"fontsize": 13, "fontweight": "bold"},
    )
    # Force correct colors for text (matplotlib places them on the slice)
    for at, col in zip(autotexts, [WHITE, WHITE]):
        at.set_color(col)

    ax.text(0, 0, f"990 km\ntotal", ha="center", va="center",
            fontsize=11, fontweight="bold", color=DARK)
    ax.set_title("Synthetic network (Q=1.0)\nOverlap vs. Gap — network B (real)",
                 fontsize=10, color=DARK, pad=12)

    handles = [
        mpatches.Patch(color=GREEN, label=f"Overlap — already built ({C['overlap_km']:.0f} km)"),
        mpatches.Patch(color=RED,   label=f"Gap — missing ({C['gap_km']:.0f} km)"),
    ]
    ax.legend(handles=handles, loc="lower center",
              bbox_to_anchor=(0.5, -0.16), fontsize=9.5, frameon=False, ncol=1)

    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# PAGE 7 — OVERLAP MAP (merged with former page 7 content)
# ══════════════════════════════════════════════════════════════════════════
def page_overlap_map(pdf):
    fig = new_page()
    hdr(fig, "Metric 2 — Spatial Overlap & Gap",
        "Green = synthetic already built  ·  Red = missing connections  ·  network B (real)", 4)

    lx, sw = 0.03, 0.24

    y = 0.87
    y = section_title(fig, lx, y, "What this measures")
    y = txt(fig, lx, y,
            "For each synthetic edge it is checked whether it falls within 15 m "
            "of the real network. Edges with at least 50 % spatial proximity are "
            "classified as overlap (already built); the rest as gap (missing). "
            "This identifies which algorithmically prioritised connections "
            "are still absent from Vienna's infrastructure.",
            width=28)
    y -= 0.010

    y = section_title(fig, lx, y, "Result — network B (real)")

    BOX_H, BOX_GAP = 0.068, 0.008
    for val, lbl, bg in [
        ("56.7 %", "Overlap (built)", GREEN),
        ("43.3 %", "Gap (missing)",   RED),
        ("429 km", "Gap length",      RED),
    ]:
        stat_box(fig, lx, y - BOX_H, sw, BOX_H, val, lbl, bg)
        y -= BOX_H + BOX_GAP
    y -= 0.012

    y = section_title(fig, lx, y, "Interpretation")
    y = txt(fig, lx, y,
            "The synthetic optimum (989 km) is leaner by design than the real "
            "network (1,810 km) — an efficient skeleton, not a dense web. The real "
            "network is denser, above all in the inner city, which already contains "
            "the skeleton: high overlap and little gap in the core.",
            width=28)
    y = txt(fig, lx, y,
            "So the 43 % gap is not a density deficit. It is the few strategic links "
            "the optimum would add — concentrated in the outer districts (S/N/SW/W/NE), "
            "not the centre — with the highest efficiency gain per km.",
            width=28)
    y -= 0.006

    y = section_title(fig, lx, y, "Legend")
    for col, lbl in [(BLUE, "Real network"), (GREEN, "Overlap"), (RED, "Gap")]:
        fig.add_artist(Rectangle((lx, y - 0.018), 0.018, 0.015,
                                 transform=fig.transFigure,
                                 facecolor=col, zorder=4))
        fig.text(lx + 0.025, y - 0.010, lbl, fontsize=8.5, color=DARK, va="center")
        y -= 0.028

    show_img(fig, 0.30, 0.07, 0.68, 0.82, OUT / "vienna_ogd" / "map_overlap_C.png")

    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# PAGE 8 — VIENNA IN CROSS-CITY CONTEXT  (NEW in v4)
# ══════════════════════════════════════════════════════════════════════════
def page_cross_city_context(pdf):
    """Place Vienna inside the 5-city multicity comparison."""
    fig = new_page()
    hdr(fig, "Vienna in cross-city context",
        "How does Vienna compare to Amsterdam, Barcelona, Berlin and Oslo?  ·  "
        "see report_multicity.pdf for full detail",
        8)

    # ── Load multicity data ────────────────────────────────────────────────
    COMP = OUT / "comparison"
    summary  = pd.read_csv(COMP / "summary_table.csv")
    spatial  = pd.read_csv(COMP / "spatial_summary.csv")
    density  = pd.read_csv(COMP / "density_table.csv")

    CITIES = ["Amsterdam", "Barcelona", "Berlin", "Oslo", "Vienna"]
    CITY_COLORS = {
        "Amsterdam": "#e41a1c", "Barcelona": "#377eb8",
        "Berlin":    "#4daf4a", "Oslo":      "#984ea3",
        "Vienna":    "#ff7f00",
    }

    def _pull(col, df, key="City"):
        out = {}
        for c in CITIES:
            sub = df[df[key] == c]
            if not len(sub):
                continue
            v = sub[col].values[0]
            if isinstance(v, str) and v.endswith("%"):
                v = float(v.rstrip("%"))
            else:
                v = float(v)
            out[c] = v
        return out

    directness = _pull("Directness @ syn_max", summary)
    efficiency = _pull("Efficiency @ syn_max", summary)
    overlap    = _pull("Overlap (%)", spatial)
    bk_density = _pull("bikeable / km²", density)

    # ── Intro (left column) + key takeaway box (right column) ──────────────
    y = 0.875
    y = section_title(fig, 0.05, y, "What this page shows")
    y = txt(fig, 0.05, y,
            "The companion multicity report places Vienna alongside four other "
            "European cities (Amsterdam, Barcelona, Berlin, Oslo). Vienna's "
            "structural metrics are recomputed against OSM bikeable (not Vienna OGD) "
            "and against the multicity synthetic run (gridl = 1701 m). The bars "
            "below isolate Vienna in that international setting.", width=90)
    y -= 0.012

    y = section_title(fig, 0.05, y, "Headline — Vienna ranks")
    rank_lines = []
    for metric_name, d in [("Directness",  directness),
                            ("Efficiency",  efficiency),
                            ("Overlap %",   overlap),
                            ("Density",     bk_density)]:
        items = sorted(d.items(), key=lambda x: x[1], reverse=True)
        rank = [c for c, _ in items].index("Vienna") + 1
        rank_lines.append(f"{metric_name}: rank {rank} / 5  ({d['Vienna']:.3g})")
    for ln in rank_lines:
        fig.text(0.05, y, "• " + ln, fontsize=9.5, color=DARK)
        y -= 0.022
    y -= 0.010

    # ── Four bar panels with Vienna highlighted ───────────────────────────
    panels = [
        ("Directness (LCC)",       directness, "{:.3f}"),
        ("Global efficiency",      efficiency, "{:.3f}"),
        ("Overlap vs bikeable %",  overlap,    "{:.0f}"),
        ("bikeable density km/km²", bk_density, "{:.1f}"),
    ]
    bar_left, bar_bottom = 0.06, 0.07
    bar_h = 0.32
    n = len(panels)
    gap = 0.025
    bar_pw = (0.88 - (n - 1) * gap) / n

    fig.text(0.5, 0.46, "Vienna highlighted · grey = other cities",
             ha="center", color=GREY, fontsize=8.5, style="italic")

    for k, (title, dct, fmt) in enumerate(panels):
        ax = fig.add_axes([
            bar_left + k * (bar_pw + gap),
            bar_bottom, bar_pw, bar_h
        ])
        labels = [c[:3] for c in CITIES]
        vals   = [dct.get(c, 0) for c in CITIES]
        colors = [CITY_COLORS[c] if c == "Vienna" else "#cfd8dc"
                  for c in CITIES]
        ax.bar(labels, vals, color=colors, edgecolor="white",
               linewidth=0.5, width=0.7)
        ax.set_title(title, fontsize=9, color=DBLUE,
                     fontweight="bold", pad=4)
        # Annotate Vienna's value
        v_idx = CITIES.index("Vienna")
        ax.annotate(fmt.format(vals[v_idx]),
                    (v_idx, vals[v_idx]),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=9, fontweight="bold",
                    color=CITY_COLORS["Vienna"])
        ax.tick_params(axis="x", labelsize=7.5, pad=2, length=0)
        ax.tick_params(axis="y", labelsize=6.5, pad=1, length=0)
        ymax = max(vals) * 1.22
        ax.set_ylim(0, ymax)
        ax.set_yticks([0, max(vals)])
        ax.grid(axis="y", alpha=0.20, lw=0.4)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.spines["left"].set_color("#cfd8dc")
        ax.spines["bottom"].set_color("#cfd8dc")

    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# PAGE 9 — GAP HOTSPOTS (Vienna OGD)  (NEW in v4)
# ══════════════════════════════════════════════════════════════════════════
def page_gap_hotspots(pdf):
    """DBSCAN on the 429 km Variant-C gap edges (OGD reference)."""
    from shapely.geometry import Point
    from sklearn.cluster import DBSCAN

    fig = new_page()
    hdr(fig, "Gap hotspots — Vienna OGD",
        "DBSCAN clusters of the 429 km gap edges (C not in B) (not within 15 m of OGD network)",
        9)

    # Compass helper
    def _dir(dx, dy, central=False):
        if central:
            return "central"
        ang = math.degrees(math.atan2(dy, dx))
        bins = [(-22.5, 22.5, "east"), (22.5, 67.5, "north-east"),
                (67.5, 112.5, "north"), (112.5, 157.5, "north-west"),
                (157.5, 180.1, "west"), (-180, -157.5, "west"),
                (-157.5, -112.5, "south-west"),
                (-112.5, -67.5, "south"),
                (-67.5, -22.5, "south-east")]
        for lo, hi, name in bins:
            if lo <= ang < hi:
                return name
        return "central"

    # Load Variant C gap edges (OGD reference)
    gap_path = OUT / "vienna_ogd" / "gap_C_full.geojson"
    if not gap_path.exists():
        fig.text(0.5, 0.5, f"[{gap_path.name} missing]",
                 ha="center", color=RED, fontsize=11)
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
        return
    gap = gpd.read_file(gap_path)
    crs_p = gap.estimate_utm_crs()
    gap_p = gap.to_crs(crs_p)

    mids = gap_p.geometry.centroid
    coords = np.array([(p.x, p.y) for p in mids])
    lengths = gap_p.geometry.length.values

    centre_x, centre_y = coords[:, 0].mean(), coords[:, 1].mean()
    typ_radius = float(np.percentile(
        ((coords[:, 0] - centre_x) ** 2
         + (coords[:, 1] - centre_y) ** 2) ** 0.5,
        95,
    ))

    db = DBSCAN(eps=300, min_samples=8).fit(coords)
    labels = db.labels_

    rows = []
    for cid in sorted(set(labels)):
        if cid == -1:
            continue
        mask = labels == cid
        cl_coords  = coords[mask]
        cl_lengths = lengths[mask]
        cx = float(cl_coords[:, 0].mean())
        cy = float(cl_coords[:, 1].mean())
        dx, dy = cx - centre_x, cy - centre_y
        dist = math.sqrt(dx ** 2 + dy ** 2)
        is_central = dist / max(typ_radius, 1) < 0.20
        rows.append({
            "cluster_id": int(cid),
            "n_edges":    int(mask.sum()),
            "total_km":   float(cl_lengths.sum() / 1000),
            "cx_m": cx, "cy_m": cy,
            "dist_km": dist / 1000,
            "direction": _dir(dx, dy, central=is_central),
        })
    cl_df = pd.DataFrame(rows).sort_values("total_km", ascending=False)
    top = cl_df.head(5).copy()
    top["rank"] = range(1, len(top) + 1)

    total_gap_km = float(lengths.sum() / 1000)
    top_km = top["total_km"].sum()
    top_share = 100 * top_km / total_gap_km if total_gap_km else 0

    # ── Two columns: map (left) + table (right) ──────────────────────────
    # Map on the left
    ax_m = fig.add_axes([0.03, 0.32, 0.55, 0.55])
    ax_m.set_axis_off()
    gap_p.plot(ax=ax_m, color="#d73027", lw=0.4, alpha=0.30, zorder=1)
    for _, r in top.iterrows():
        mask = labels == r["cluster_id"]
        pts = coords[mask]
        if len(pts) >= 3:
            hull = gpd.GeoSeries(
                [Point(x, y) for x, y in pts], crs=crs_p
            ).unary_union.convex_hull
        else:
            hull = Point(pts[0]).buffer(300)
        gpd.GeoSeries([hull], crs=crs_p).plot(
            ax=ax_m, facecolor="#ff7f00", alpha=0.30,
            edgecolor="#ff7f00", lw=1.5, zorder=2,
        )
        ax_m.annotate(
            f"#{int(r['rank'])}",
            (r["cx_m"], r["cy_m"]),
            color="white", fontsize=12, fontweight="bold",
            ha="center", va="center",
            bbox=dict(boxstyle="circle,pad=0.35",
                      facecolor="#ff7f00", edgecolor="white", linewidth=1.5),
            zorder=5,
        )
    ax_m.set_title(
        f"Top {len(top)} hotspots: {top_km:.0f} km of {total_gap_km:.0f} km "
        f"({top_share:.0f}% of all gap)",
        fontsize=9.5, color=DBLUE, fontweight="bold",
    )

    # Sidebar (right): explanation + table
    y = 0.875
    y = section_title(fig, 0.62, y, "What this shows")
    y = txt(fig, 0.62, y,
            "Each of the 429 km of unbuilt high-priority synthetic edges "
            "is reduced to its midpoint. DBSCAN groups them spatially "
            "(eps = 300 m, min = 8). The five biggest clusters are "
            "shown as orange hulls on the map and listed here.", width=35)
    y -= 0.010

    y = section_title(fig, 0.62, y, "Top 5 hotspots")
    headers = ["#", "km", "edges", "direction"]
    cws  = [0.04, 0.06, 0.07, 0.18]
    cxs  = [0.62 + sum(cws[:i]) for i in range(len(cws) + 1)]
    # Header row
    fig.text(cxs[0], y, headers[0], fontsize=8.5,
             fontweight="bold", color=DBLUE)
    fig.text(cxs[1], y, headers[1], fontsize=8.5,
             fontweight="bold", color=DBLUE)
    fig.text(cxs[2], y, headers[2], fontsize=8.5,
             fontweight="bold", color=DBLUE)
    fig.text(cxs[3], y, headers[3], fontsize=8.5,
             fontweight="bold", color=DBLUE)
    y -= 0.022
    fig.add_artist(Rectangle(
        (0.61, y + 0.008), 0.36, 0.001,
        transform=fig.transFigure, facecolor=DBLUE,
    ))
    for _, r in top.iterrows():
        fig.text(cxs[0], y, f"#{int(r['rank'])}",
                 fontsize=9, fontweight="bold",
                 color="#ff7f00")
        fig.text(cxs[1], y, f"{r['total_km']:.1f}",
                 fontsize=9, color=DARK)
        fig.text(cxs[2], y, f"{int(r['n_edges'])}",
                 fontsize=9, color=DARK)
        fig.text(cxs[3], y, r["direction"],
                 fontsize=9, color=DARK)
        y -= 0.022

    y -= 0.012
    note_box(fig, 0.62, y, 0.34,
             "Cluster polygons can be exported to QGIS by re-running "
             "analysis_gap_clusters.py — the 5 hotspots are exact spatial "
             "priorities for Vienna's next cycling-infrastructure round.",
             bg="#fff2e1", fg="#5a3500")

    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# PAGE 10 — SUMMARY
# ══════════════════════════════════════════════════════════════════════════
def page_summary(pdf):
    fig = new_page()
    hdr(fig, "Summary & Conclusions",
        "Key findings  ·  Vienna bicycle network vs. synthetic optimum", 9)

    y = 0.87
    findings = [
        ("1 · Length does not equal Quality",
         "Vienna has 1,810 km of cycle infrastructure — nearly double the synthetic "
         "maximum of 989 km. Yet length alone is a poor indicator. The full network "
         "only becomes functional when all infrastructure types are combined (network B)."),
        ("2 · Fragmentation is the central problem",
         "The full network (B) achieves 89.4 % LCC share with 546 components — "
         "functionally connected. The restrictive layers alone are near-useless for "
         "navigation: dedicated tracks (A) split into 2,360 components, marked lanes "
         "into 2,520. A cyclist using only dedicated paths has no connected network to ride."),
        ("3 · Global efficiency — the decisive metric",
         "GE(B) = 0.557 is 14.6 % below the synthetic optimum C (0.653). "
         "This is meaningful but bridgeable. The algorithm demonstrates that "
         "a fully connected network with the same 989 km would reach GE = 0.653, "
         "improving by 17 %."),
        ("4 · Coverage is excellent — and one area where Vienna wins",
         "At 90.7 %, network B marginally exceeds the synthetic maximum (89.3 %). "
         "The network is well distributed across all city districts. "
         "This is the one metric where real infrastructure outperforms the algorithm."),
        ("5 · 43 % of the optimal network is missing",
         "Approximately 429 km of algorithmically optimal connections are not yet built. "
         "These gap segments (gap_C_full.geojson) represent the highest "
         "theoretical efficiency gain per kilometre and can serve as direct "
         "planning input for future cycling infrastructure investment."),
    ]

    for heading, body in findings:
        y = section_title(fig, 0.05, y, heading)
        y = txt(fig, 0.05, y, body, width=90)
        y -= 0.010

    y -= 0.005
    y = note_box(fig, 0.05, y, 0.90,
                 "Core finding: Vienna's bicycle network has excellent spatial reach (90.7 % coverage) "
                 "and reasonable connectivity (GE = 0.557). The main deficit is structural: safer "
                 "infrastructure types are highly fragmented. The bikenwgrowth model shows that "
                 "a strategically built network of equal length could achieve 17 % higher efficiency. "
                 "Gap maps provide concrete spatial priorities for future investment.",
                 bg="#d4edda", fg="#155724")

    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# BUILD
# ══════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════
# PAGE 8 — SAFE INFRASTRUCTURE (A): REACH & CONNECTIVITY  (reviewer request)
# ══════════════════════════════════════════════════════════════════════════
def page_safe_infrastructure(pdf):
    fig = new_page()
    hdr(fig, "Safe Infrastructure (A) — Reach & Connectivity",
        "Does the separated cycle-path network reach the city, and does it connect?"
        "  ·  network A", 8)

    lx, sw = 0.03, 0.24
    y = 0.877
    y = section_title(fig, lx, y, "What this measures")
    y = txt(fig, lx, y,
            "Network A is the physically separated / protected cycle-path layer — "
            "the infrastructure most cyclists consider safe. Two questions: spatial "
            "reach (% of Vienna within 500 m of A) and connectivity (share of A in "
            "one connected component).",
            width=30)
    y -= 0.008

    y = section_title(fig, lx, y, "Result — network A (safe)")
    BOX_H, BOX_GAP = 0.068, 0.008
    for val, lbl, bg in [
        (f"{A_REAL['coverage']:.0f} %", "Spatial reach (500 m)", TEAL),
        (f"{A_REAL['lcc']*100:.0f} %",  "LCC share (connected)", RED),
        (f"{A_REAL['components']:,}",   "Components (islands)",   RED),
        (f"{A_REAL['length']:.0f} km",  "Total length",          BLUE),
    ]:
        stat_box(fig, lx, y - BOX_H, sw, BOX_H, val, lbl, bg)
        y -= BOX_H + BOX_GAP
    y -= 0.006

    y = section_title(fig, lx, y, "Interpretation")
    y = txt(fig, lx, y,
            "Safe infrastructure is spatially well distributed — 80 % of Vienna is "
            "within 500 m of a protected path, nearly as much as the full network "
            "(90.7 %).",
            width=30)
    y = note_box(fig, lx, y, sw + 0.01,
                 "But it does not connect: only 18 % of A forms one piece (2,360 "
                 "islands, vs 546 for the full network). A safety-conscious cyclist "
                 "finds a protected path nearby almost everywhere, yet cannot ride a "
                 "continuous safe route — they are repeatedly forced onto unsafe "
                 "streets. Connecting the islands, not adding kilometres, is the key.",
                 bg="#fdecea", fg="#7b241c")

    show_img(fig, 0.30, 0.07, 0.68, 0.82, OUT / "vienna_ogd" / "frag_A.png",
             "Fragmentation map — network A (safe infrastructure). Dark blue = LCC. "
             "Each colour = one isolated component.")

    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


print("Generating report v4 …")
with PdfPages(PDF) as pdf:
    page_title(pdf)              # 1
    page_intro(pdf)               # 2
    page_fragmentation_map(pdf)   # 3
    page_overlap_map(pdf)         # 4
    page_efficiency(pdf)          # 5
    page_directness(pdf)          # 6
    page_coverage(pdf)            # 7
    page_safe_infrastructure(pdf) # 8  reviewer request: safe network A
    page_summary(pdf)             # 9

print(f"Saved: {PDF}")

print(f"\nDone: {PDF}")
try:
    print(f"Size: {PDF.stat().st_size/1024:.0f} KB")
except FileNotFoundError:
    pass