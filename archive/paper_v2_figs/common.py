"""Shared helpers for the paper_v2 figure scripts (CPU only, thread-capped)."""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "8")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import csv
import glob
import re
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BIT = "/vol/bitbucket/lbastian/latent-dynamics"
A2 = f"{BIT}/analysis_v2"
CACHE = f"{A2}/cache"
EXT = f"{A2}/ext/ext"
DYN = f"{BIT}/dynamics_v2"
RUNS = f"{BIT}/runs_v2/runs"
RE2 = f"{BIT}/reanalysis_v2"
OUT = f"{BIT}/paper_v2/figures"
os.makedirs(OUT, exist_ok=True)

# Okabe-Ito colour-blind-safe palette
C = dict(blue="#0072B2", orange="#E69F00", green="#009E73", verm="#D55E00",
         sky="#56B4E9", purple="#CC79A7", yellow="#F0E442", black="#000000",
         grey="#777777")

COL_W = 3.25   # ICML column width (in)
PAGE_W = 6.75  # ICML text width (in)


def setup():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 7.5,
        "axes.labelsize": 7.5,
        "axes.titlesize": 8,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "lines.linewidth": 1.1,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "figure.dpi": 150,
    })


def savefig(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, bbox_inches="tight", pad_inches=0.02)
    pv = "/tmp/claude-2292/-homes-lbastian-latent-dynamics/ff433239-7ead-4777-86c4-f8c05d4a450b/scratchpad/prev_" + name.replace(".pdf", ".png")
    fig.savefig(pv, bbox_inches="tight", pad_inches=0.02, dpi=170)
    plt.close(fig)
    print("wrote", p)


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


RUN_RE = re.compile(r"^(mnist|fashion_mnist|cifar10)_probe_none_d(\d+)_s(\d+)_(.+)\.npz$")


def cache_runs():
    """Return {config_tag: {seed: path}} for MNIST caches."""
    out = defaultdict(dict)
    for p in sorted(glob.glob(f"{CACHE}/*.npz")):
        m = RUN_RE.match(os.path.basename(p))
        if m and m.group(1) == "mnist":
            out[m.group(4)][int(m.group(3))] = p
    return out


def panel_label(ax, s, x=-0.18, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="bottom", ha="left")
