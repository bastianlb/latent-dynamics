"""Matplotlib helpers: headless backend and the paper rcParams (from summary_figs.py)."""


def agg():
    """Select the headless Agg backend; returns matplotlib.pyplot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def paper_style():
    """Apply the paper rcParams (summary_figs.py)."""
    plt = agg()
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 8.5, "legend.fontsize": 7, "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.7, "lines.linewidth": 1.3, "legend.frameon": False,
                         "figure.dpi": 100, "pdf.fonttype": 42, "font.family": "DejaVu Sans", "xtick.major.width": 0.7, "ytick.major.width": 0.7})
    return plt
