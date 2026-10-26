"""Figure: timing of train-test divergence (vs accuracy gap) and of geometry vs test accuracy.
Points: per-config log10 t50 differences (ext set, 69 configs); bars: median and config-bootstrap 95% CI
parsed from analysis_v2/ext/ext/step2_summary.txt."""
from common import *

setup()
tim = read_csv(f"{EXT}/step2_timing_per_config.csv")
summ = open(f"{EXT}/step2_summary.txt").read().splitlines()

METRICS = [("fisher", "Fisher", "Fisher"), ("eff_rank_pr", "eff. rank (PR)", "Effective rank"),
           ("mag_t1", "Mag(t=1)", r"Mag$(t{=}1)$"), ("pmag_fixed", "PMag[.5,10]", r"PMag (fixed window)$^\dagger$"),
           ("probe_acc", "probe acc", r"Probe accuracy$^\dagger$"), ("h1_top5", "H1 top-5", r"$H_1$ top-5$^\dagger$"),
           ("h1_total", "H1 total", r"$H_1$ total$^\dagger$")]
pat = re.compile(r"median ([+-][\d.]+) \[([+-][\d.]+),([+-][\d.]+)\] (?:over (\d+) valid|n=(\d+))")


def parse(section_hdr, prefix):
    out = {}
    on = False
    for ln in summ:
        if ln.startswith("== ("):
            on = ln.startswith(section_hdr)
            continue
        if not on:
            continue
        for key, sname, _ in METRICS:
            if ln.strip().startswith(prefix + sname + " ") or ln.strip().startswith(prefix + sname.ljust(14)):
                m = pat.search(ln)
                if m and key not in out:
                    out[key] = (float(m.group(1)), float(m.group(2)), float(m.group(3)), int(m.group(4) or m.group(5)))
    return out


# (a) vs accuracy gap
ca = parse("== (a)", "")
# (c) vs test acc  (lines start with 'vs test_acc  <name>')
cc = {}
on = False
for ln in summ:
    if ln.startswith("== ("):
        on = ln.startswith("== (c)")
        continue
    if on and ln.strip().startswith("vs test_acc"):
        rest = ln.strip()[len("vs test_acc"):].strip()
        for key, sname, _ in METRICS:
            if rest.startswith(sname) and key not in cc:
                m = pat.search(ln)
                if m:
                    cc[key] = (float(m.group(1)), float(m.group(2)), float(m.group(3)), int(m.group(4) or m.group(5)))
                break


def diffs(a_col, b_col):
    out = {}
    for key, _, _ in METRICS:
        if a_col(key) not in tim[0]:
            continue
        v = []
        for r in tim:
            x = fnum(r[a_col(key)]); y = fnum(r[b_col])
            if np.isfinite(x) and np.isfinite(y):
                v.append(x - y)
        out[key] = np.array(v)
    return out


# use the step-4 bootstrap (same as the timing-sensitivity table) for the medians and CIs
_s4 = {(r["ref"], r["metric"]): r for r in read_csv(f"{A2}/ext2/mnist/step4_timing_summary.csv") if r["variant"] == "all_runs" and abs(float(r["q"]) - 0.5) < 1e-9}
for _ref, _d in [("gap", ca), ("test_acc", cc)]:
    for _k in list(_d):
        _r = _s4.get((_ref, _k))
        if _r:
            _d[_k] = (float(_r["median"]), float(_r["lo"]), float(_r["hi"]), int(_r["n_valid"]))
da = diffs(lambda k: f"t50_d_{k}", "t50_gap")
dc = diffs(lambda k: f"t50_te_{k}", "t50_test_acc")

fig, axes = plt.subplots(1, 2, figsize=(PAGE_W, 2.05), sharey=True, gridspec_kw=dict(wspace=0.06))
rng = np.random.RandomState(0)
titles = [r"(a) train$-$test divergence vs.\ accuracy gap", "(b) test-probe metric vs.\ test accuracy"]
titles = ["train$-$test divergence $\\Delta_m$ vs. accuracy gap", "test-probe metric vs. test accuracy"]
for ax, dd, cs, ttl in zip(axes, [da, dc], [ca, cc], titles):
    ypos = np.arange(len(METRICS))[::-1]
    for y, (key, _, lab) in zip(ypos, METRICS):
        if key not in dd:
            continue
        v = dd[key]
        ax.scatter(v, y + rng.uniform(-0.22, 0.22, len(v)), s=3.5, color=C["sky"], alpha=0.55, lw=0, zorder=1)
        med, lo, hi, n = cs[key]
        col = C["blue"] if hi < 0 else (C["verm"] if lo > 0 else C["black"])
        ax.errorbar([med], [y], xerr=[[med - lo], [hi - med]], fmt="D", color=col, ms=3.6, lw=1.3, capsize=2, zorder=3)
    ax.axvline(0, color="k", lw=0.6, ls="-")
    ax.set_yticks(ypos)
    ax.set_yticklabels([m[2] for m in METRICS])
    ax.set_xlim(-3.2, 3.2)
    ax.set_xlabel(r"$\log_{10} t_{50}(\mathrm{metric}) - \log_{10} t_{50}(\mathrm{reference})$")
    ax.set_title(ttl, fontsize=7.5)
    ax.text(-3.1, len(METRICS) - 0.35, r"$\leftarrow$ metric first", fontsize=6, color=C["blue"], va="bottom")
    ax.text(3.1, len(METRICS) - 0.35, r"metric lags $\rightarrow$", fontsize=6, color=C["verm"], va="bottom", ha="right")
    ax.set_ylim(-0.6, len(METRICS) - 0.0)
panel_label(axes[0], "a", x=-0.55, y=1.0)
panel_label(axes[1], "b", x=-0.03, y=1.0)
savefig(fig, "fig_timing.pdf")
print(ca)
print(cc)
