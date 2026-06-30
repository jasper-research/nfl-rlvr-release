"""Publication figure style for the NFL win-probability RLVR paper.

Serif-neutral typography, restrained colorblind-safe palette, vector PDF output.
Ported from the qark-rag paper style so the two papers share a visual language.

Usage:
    import figstyle as fs
    fs.use_style()
    fig, ax = plt.subplots(figsize=(fs.WIDE, 2.4))
    ...
    fs.save(fig, "F3_reliability")   # writes paper/figs/F3_reliability.{pdf,png}
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGS = Path(__file__).resolve().parent / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

# Widths (inches). Single-column arXiv body text is ~6.9in; HALF for side-by-side.
WIDE = 6.9
COL = 3.35

# Restrained, colorblind-safe palette.
INK = "#21303f"        # near-ink: betting-market reference, text
ACCENT = "#b1283a"     # crimson: the direct-RLVR model (calibration lead)
COOL = "#2f6f8f"       # slate blue: the masked-CoT model
GREEN = "#2e7d52"      # green: frontier zero-shot reference
BAR = "#6f93a8"        # muted steel: empirical-rate teacher / histogram fills
NULL = "#9aa3ab"       # grey: untrained base / "everything else"
NULL_L = "#d3d8dd"
GRID = "#e6e8eb"
CEIL = "#caa53d"       # muted gold: the static-information ceiling band


def use_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.06,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "TeX Gyre Heros", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8.5,
        "axes.titleweight": "bold",
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7,
        "legend.frameon": False,
        "axes.linewidth": 0.7,
        "axes.edgecolor": "#3a3f44",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "lines.linewidth": 1.4,
        "lines.markersize": 4,
        "legend.handlelength": 1.4,
        "legend.columnspacing": 1.0,
        "legend.labelspacing": 0.3,
    })


def halo(txt, lw: float = 2.2):
    """White outline so a label stays readable over a busy area."""
    import matplotlib.patheffects as pe
    txt.set_path_effects([pe.withStroke(linewidth=lw, foreground="white")])
    return txt


def panel_label(ax, text: str, dx: float = -0.02, dy: float = 1.0) -> None:
    ax.text(dx, dy + 0.04, text, transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="bottom", ha="right")


def save(fig, name: str) -> None:
    fig.savefig(FIGS / f"{name}.pdf")
    fig.savefig(FIGS / f"{name}.png", dpi=200)
    plt.close(fig)
