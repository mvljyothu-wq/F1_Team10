"""Shared Matplotlib styling so every figure in the report looks consistent.

Colours: validated categorical palette (blue, orange, aqua) in a fixed order:
the conventional estimator is always blue, the ML estimator always orange and
the aggressive K=2 conventional variant always aqua.
"""

import matplotlib

matplotlib.use("Agg")       # render to files; no display needed
import matplotlib.pyplot as plt  # noqa: E402

# Fixed slot order of the validated palette: blue, orange, aqua, yellow, magenta, green.
COLORS = {"rto": "#2a78d6", "ml": "#eb6834", "rto_k2": "#1baf7a", "ml_v2": "#eda100",
          "rto_rack": "#e87ba4", "ml_v2_rack": "#008300", "baseline": "#1baf7a"}
LABELS = {"rto": "Conventional (RFC 6298, K=4)", "ml": "ML estimator v1",
          "rto_k2": "Conventional, K=2 (more aggressive)", "ml_v2": "ML estimator v2",
          "rto_rack": "RFC 6298 + RACK", "ml_v2_rack": "ML v2 + RACK",
          "baseline": "Recent-max baseline"}
MODE_ORDER = ["rto", "ml", "rto_k2", "ml_v2", "rto_rack", "ml_v2_rack"]
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"


def setup():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "axes.titlecolor": TEXT,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 10,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "legend.fontsize": 9, "font.size": 10,
        "lines.linewidth": 2.0, "figure.dpi": 110,
    })


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
