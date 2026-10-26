"""Draft-3 tables from analysis_v2/ext3 (Amendment 4) and step-4 timing. Writes paper_v2/tables/*.tex and paper_stats3.txt.
Tables: tab_equiv, tab_equiv_full_{mnist,fashion}_body (from ext3 margin_mstar.csv), tab_mstar, tab_resid3, tab_power_body,
tab_power_resid_body, tab_uncert_body, tab_learners, and the 'All' column of tab_timing_{a,c} (step-4 bootstrap)."""
from common import *

TAB = f"{BIT}/paper_v2/tables"
E3 = f"{A2}/ext3"
E2 = f"{A2}/ext2"
CUTS = [10, 30, 100, 300, 1000]
STATS = []


def st(*a):
    s = " ".join(str(x) for x in a)
    STATS.append(s)
    print(s)


def sg(x, nd=3, plus=True):
    if x is None or not np.isfinite(x):
        return "--"
    s = f"{abs(x):.{nd}f}"
    if x < 0 and float(s) != 0:
        return "$-$" + s
    return ("+" if plus else "") + s


def un(x, nd=3):
    return f"{x:.{nd}f}"


def write(name, txt):
    with open(f"{TAB}/{name}.tex", "w") as f:
        f.write(txt + "\n")
    print("wrote", name)


def code(lbl):
    harm = "possible harm" in lbl
    c = "E" if lbl.startswith("equivalent") else ("P" if lbl.startswith("positive") else "I")
    return c + ("$^\\downarrow$" if harm else "")


MS = {ds: {(r["model"], r["target"], int(r["cutoff"])): r for r in read_csv(f"{E3}/{ds}/margin_mstar.csv")} for ds in ["mnist", "fashion"]}
RF = {ds: {(r["target"], int(r["cutoff"])): r for r in read_csv(f"{E3}/{ds}/design_refit.csv")} for ds in ["mnist", "fashion"]}
HO = {ds: {(r["target"], int(r["cutoff"])): r for r in read_csv(f"{E3}/{ds}/multiplicity_holm.csv") if r["kind"] == "iid"} for ds in ["mnist", "fashion"]}
TG = {"acc_15kw": "test acc.", "gap_15kw": "gap", "acc_15k": "test acc.\\ (step 15k)", "gap_15k": "gap (step 15k)"}

# ---------------------------------------------------------------- tab_equiv (window targets)
L = [r"\begin{tabular}{llccccc}", r"\toprule", r"Data, model & Target & 10 & 30 & 100 & 300 & 1000 \\", r"\midrule"]
for ds, model, lab in [("mnist", "ridge", "MNIST, ridge"), ("mnist", "gbm", "MNIST, GBM"), ("fashion", "ridge", "Fashion, ridge"), ("fashion", "gbm", "Fashion, GBM")]:
    for j, tgt in enumerate(["acc_15kw", "gap_15kw"]):
        cells = []
        for c in CUTS:
            r = MS[ds][(model, tgt, c)]
            d = fnum(r["delta"])
            s = f"{abs(d):.3f}"[1:]
            sign = "$-$" if (d < 0 and float(f"{abs(d):.3f}") != 0) else "+"
            cells.append(f"{sign}{s}\\,{code(r['class_iid'])}")
        L.append(f"{lab if j == 0 else ''} & {'test acc.' if j == 0 else 'gap'} & " + " & ".join(cells) + r" \\")
    if ds == "mnist" and model == "gbm":
        L.append(r"\midrule")
L += [r"\bottomrule", r"\end{tabular}"]
write("tab_equiv", "\n".join(L))

# counts
for ds in ["mnist", "fashion"]:
    for model in ["ridge", "gbm"]:
        for tg in [("acc_15kw", "gap_15kw"), ("acc_15k", "gap_15k")]:
            cnt = {"E": 0, "P": 0, "I": 0}
            for t in tg:
                for c in CUTS:
                    cl = MS[ds][(model, t, c)]["class_iid"]
                    cnt["E" if cl.startswith("equivalent") else ("P" if cl.startswith("positive") else "I")] += 1
            st(f"[ext3 class counts] {ds} {model} targets {tg}: {cnt} (P includes 'also within margin' cells)")

# ---------------------------------------------------------------- full equivalence tables
for ds in ["mnist", "fashion"]:
    L = []
    for tgt in TG:
        for c in CUTS:
            f = lambda r: f"{sg(fnum(r['delta']))} [{sg(fnum(r['lo90_iid']))}, {sg(fnum(r['hi90_iid']))}] {code(r['class_iid'])}"
            ro, gb = MS[ds][("ridge", tgt, c)], MS[ds][("gbm", tgt, c)]
            L.append(f"{TG[tgt] if c == 10 else ''} & {c} & {f(ro)} & {fnum(ro['mstar_iid']):.3f} & {f(gb)} & {fnum(gb['mstar_iid']):.3f} \\\\")
        L.append(r"\midrule")
    L[-1] = r"\bottomrule"
    write(f"tab_equiv_full_{ds}_body", "\n".join(L))

# ---------------------------------------------------------------- m* table (ridge, window targets)
L = [r"\begin{tabular}{llccccc}", r"\toprule", r"Data, target & & 10 & 30 & 100 & 300 & 1000 \\", r"\midrule"]
for ds, dl in [("mnist", "MNIST"), ("fashion", "Fashion")]:
    for tgt, tl in [("acc_15kw", "acc."), ("gap_15kw", "gap")]:
        rows = [MS[ds][("ridge", tgt, c)] for c in CUTS]
        dag = [(HO[ds][(tgt, c)]["equivalent_holm"] == "1") for c in CUTS]
        L.append(f"{dl} {tl} & $m^\\ast$ & " + " & ".join(f"{fnum(r['mstar_iid']):.3f}" + ("$^\\dagger$" if d else "") for r, d in zip(rows, dag)) + r" \\")
        L.append(" & $m^\\ast$, refit & " + " & ".join(f"{fnum(RF[ds][(tgt, c)]['mstar_refit']):.3f}" for c in CUTS) + r" \\")
        L.append(" & 0.5-pt RMSE & " + " & ".join(f"{fnum(r['dR2_improve_0p5pt']):.3f}" for r in rows) + r" \\")
        st(f"[mstar] {ds} {tgt} m*: " + ", ".join(f"{fnum(r['mstar_iid']):.3f}" for r in rows) + " | refit: " +
           ", ".join(f"{fnum(RF[ds][(tgt, c)]['mstar_refit']):.3f}" for c in CUTS) + " | 0.5pt: " + ", ".join(f"{fnum(r['dR2_improve_0p5pt']):.4f}" for r in rows) +
           " | RMSE(B1) pts: " + ", ".join(f"{fnum(r['rmse_B1_pts']):.2f}" for r in rows))
        if not (ds == "fashion" and tgt == "gap_15kw"):
            L.append(r"\midrule" if tgt == "gap_15kw" or True else "")
    # holm count
    ne = sum(HO[ds][(t, c)]["equivalent_unadj"] == "1" for t in ["acc_15kw", "gap_15kw"] for c in CUTS)
    nh = sum(HO[ds][(t, c)]["equivalent_holm"] == "1" for t in ["acc_15kw", "gap_15kw"] for c in CUTS)
    st(f"[holm] {ds}: equivalent unadjusted {ne}/10, after Holm {nh}/10")
L += [r"\bottomrule", r"\end{tabular}"]
write("tab_mstar", "\n".join(L))

# ---------------------------------------------------------------- residual table (MNIST), parsed from resid_summary.txt
txt = open(f"{E3}/mnist/resid_summary.txt").read().splitlines()
pat = re.compile(r"c(\d+): ([+-][\d.]+) \[([+-][\d.]+),([+-][\d.]+)\]")
cur = None
RES = {}
for ln in txt:
    m = re.match(r"\[mnist\] (\w+):", ln)
    if m:
        cur = m.group(1)
        continue
    s = ln.strip()
    for key, lab in [("design-residual:lin", "linear"), ("design-residual:quad", "quadratic"), ("design-residual:gbm", "GBM"), ("config-mean", "config mean")]:
        if s.startswith(key):
            RES[(cur, lab)] = {int(a): (float(b), float(c), float(d)) for a, b, c, d in pat.findall(s)}


def cc(x):
    d, lo, hi = x
    f2 = lambda v: ("$-$" if v < 0 and abs(v) >= 0.005 else "") + f"{abs(v):.2f}" if (v < 0 and abs(v) >= 0.005) else f"{abs(v):.2f}" if v > -0.005 else "0.00"
    return f"{sg(d)} {{\\scriptsize [{sg(lo, 2, False)}, {sg(hi, 2, False)}]}}"


L = [r"\begin{tabular}{llccccc}", r"\toprule", r"Target & Design model & 10 & 30 & 100 & 300 & 1000 \\", r"\midrule"]
sd = {}
for r in read_csv(f"{E3}/mnist/resid_results.csv"):
    sd[(r["target"], r["variant"])] = fnum(r["resid_sd"])
sdk = {"linear": "design-residual:lin", "quadratic": "design-residual:quad", "GBM": "design-residual:gbm", "config mean": "config-mean (within-config, centred)"}
for tgt, tl in [("acc_15kw", "test acc."), ("gap_15kw", "gap")]:
    for k, lab in enumerate(["linear", "quadratic", "GBM", "config mean"]):
        v = RES[(tgt, lab)]
        sdv = sd.get((tgt, sdk[lab]))
        if sdv is None:
            for (t2, vv), x in sd.items():
                if t2 == tgt and ("within" in vv or "config" in vv) and lab == "config mean":
                    sdv = x
        L.append(f"{tl if k == 0 else ''} & {lab} & " + " & ".join(cc(v[c]) for c in CUTS) + r" \\")
        st(f"[resid3] mnist {tgt} {lab}: " + "; ".join(f"c{c} {v[c][0]:+.3f} [{v[c][1]:+.2f},{v[c][2]:+.2f}]" for c in CUTS) + f" sd={sdv}")
    if tgt == "acc_15kw":
        L.append(r"\midrule")
L += [r"\bottomrule", r"\end{tabular}"]
write("tab_resid3", "\n".join(L))

# ---------------------------------------------------------------- power v2
pw = read_csv(f"{E3}/mnist/power_summary.csv")
G = {}
for r in pw:
    G.setdefault((r["signal"], r["target"], int(r["cutoff"]), r["variant"]), []).append(r)
SIG = {"geo:te_eff_rank_pr": "eff. rank", "geo:te_fisher": "Fisher"}
L = []
for sg_ in ["geo:te_eff_rank_pr", "geo:te_fisher"]:
    for tgt in ["acc_15kw", "gap_15kw"]:
        for c in [100, 1000]:
            for var in ["full", "compact8"]:
                rr = {round(float(x["level"]), 2): x for x in G[(sg_, tgt, c, var)]}
                r0 = list(rr.values())[0]
                l80 = fnum(r0["level_at_80pct"])
                cells = [f"{float(rr[l]['detection']):.2f} ({sg(float(rr[l]['mean_delta']))})" for l in [0.02, 0.05, 0.1]]
                L.append(f"{SIG[sg_]} & {TG[tgt]} & {c} & {var} & {float(r0['r2_B1']):.3f} & {sg(l80, 3, False) if np.isfinite(l80) else 'not reached'} & " + " & ".join(cells) + r" \\")
                st(f"[power-v2] {SIG[sg_]} {tgt} cut{c} {var}: R2B1 {float(r0['r2_B1']):.3f} 80% level {l80:.3f}")
    L.append(r"\midrule")
L[-1] = r"\bottomrule"
write("tab_power_body", "\n".join(L))
L = []
for tgt in ["acc_15kw", "gap_15kw"]:
    for c in [100, 1000]:
        for var in ["full", "compact8"]:
            rr = {round(float(x["level"]), 2): x for x in G[("resid", tgt, c, var)]}
            r0 = list(rr.values())[0]
            l80 = fnum(r0["level_at_80pct"])
            cells = [f"{float(rr[l]['detection']):.2f} ({sg(float(rr[l]['mean_delta']))})" for l in [0.3, 0.5, 0.7]]
            L.append(f"{TG[tgt]} & {c} & {var} & {float(r0['r2_B1']):.3f} & {sg(l80, 2, False) if np.isfinite(l80) else 'not reached'} & " + " & ".join(cells) + r" \\")
L.append(r"\bottomrule")
write("tab_power_resid_body", "\n".join(L))

# ---------------------------------------------------------------- uncertainty variants
L = []
HOLMTXT = {}
for ds, dl in [("mnist", "MNIST"), ("fashion", "Fashion")]:
    for tgt in ["acc_15kw", "gap_15kw"]:
        for c in CUTS:
            r = MS[ds][("ridge", tgt, c)]
            rf = RF[ds][(tgt, c)]
            h = HO[ds][(tgt, c)]
            ci_ = lambda lo, hi: f"[{sg(float(lo))}, {sg(float(hi))}]"
            L.append(f"{dl + ' ' + TG[tgt] if c == 10 else ''} & {c} & {sg(fnum(r['delta']))} & {ci_(r['lo95_iid'], r['hi95_iid'])} & {ci_(r['lo95_block'], r['hi95_block'])} & "
                     f"{ci_(rf['lo95_refit'], rf['hi95_refit'])} & {float(h['holm_p']):.3f} \\\\")
        L.append(r"\midrule")
L[-1] = r"\bottomrule"
write("tab_uncert_body", "\n".join(L))
# stats: refit CI ranges
for ds in ["mnist", "fashion"]:
    for pre in ["acc", "gap"]:
        v = [fnum(RF[ds][(f"{pre}_15kw", c)]["mstar_refit"]) for c in CUTS]
        lo = [fnum(RF[ds][(f"{pre}_15kw", c)]["lo95_refit"]) for c in CUTS]
        hi = [fnum(RF[ds][(f"{pre}_15kw", c)]["hi95_refit"]) for c in CUTS]
        st(f"[refit] {ds} {pre}: m* refit {min(v):.3f}..{max(v):.3f}; 95% lo min {min(lo):+.3f}, hi max {max(hi):+.3f}")

# ---------------------------------------------------------------- learner / feature-set survival table
surv = {}
for ds in ["mnist", "fashion"]:
    on = False
    for ln in open(f"{E3}/{ds}/learn_summary.txt"):
        if "survival of" in ln:
            on = True
            continue
        if ln.startswith("###") and "survival" not in ln:
            on = False
        m = re.match(r"\s+(leave_one_\w+)\s+(ridge|krr|mlp)\s+(\w+)\s*: hurts\(H\) (\d)/6.*dR2 range \[([+-][\d.]+),([+-][\d.]+)\]", ln)
        if on and m:
            surv[(ds, m.group(1), m.group(2), m.group(3))] = (int(m.group(4)), float(m.group(5)), float(m.group(6)))
cols = [("mnist", "leave_one_n_out"), ("mnist", "leave_one_wd_out"), ("mnist", "leave_one_optimizer_out"), ("fashion", "leave_one_n_out")]
L = [r"\begin{tabular}{l" + "ccc" * 4 + "}", r"\toprule",
     r" & \multicolumn{3}{c}{MNIST leave-one-$n$} & \multicolumn{3}{c}{MNIST leave-one-wd} & \multicolumn{3}{c}{MNIST Adam$\to$SGD} & \multicolumn{3}{c}{Fashion leave-one-$n$} \\",
     r"$G$ variant & ridge & KRR & MLP & ridge & KRR & MLP & ridge & KRR & MLP & ridge & KRR & MLP \\", r"\midrule"]
for var in ["full", "compact4", "compact8", "pca2", "pca4"]:
    cells = []
    for ds, sp in cols:
        for lr in ["ridge", "krr", "mlp"]:
            k = surv[(ds, sp, lr, var)]
            cells.append(f"{k[0]}/6")
    L.append(f"{var.replace('compact', 'compact-').replace('pca', 'PCA-')} & " + " & ".join(cells) + r" \\")
L += [r"\bottomrule", r"\end{tabular}"]
write("tab_learners", "\n".join(L))
for k, v in sorted(surv.items()):
    if k[3] in ("full", "compact4", "pca2") and k[2] == "ridge":
        st(f"[learners] {k}: hurts {v[0]}/6, dR2 range [{v[1]:+.3f},{v[2]:+.3f}]")

# ---------------------------------------------------------------- timing 'All' column from step-4 bootstrap
T4 = {(r["variant"], float(r["q"]), r["ref"], r["metric"]): r for r in read_csv(f"{E2}/mnist/step4_timing_summary.csv")}
ORDER = {"Fisher": "fisher", "Effective rank": "eff_rank_pr", "Mag$(t{=}1)$": "mag_t1", "PMag (fixed window)": "pmag_fixed", "Probe accuracy": "probe_acc",
         "$H_1$ top-5": "h1_top5", "$H_1$ total": "h1_total", "Latent norm": "norm"}
for kind, ref in [("a", "gap"), ("c", "test_acc")]:
    fn = f"{TAB}/tab_timing_{kind}.tex"
    out = []
    for ln in open(fn).read().splitlines():
        lab = ln.split(" & ")[0]
        if lab in ORDER and (("all_runs", 0.5, ref, ORDER[lab]) in T4):
            r = T4[("all_runs", 0.5, ref, ORDER[lab])]
            parts = ln.split(" & ")
            parts[1] = f"{sg(float(r['median']), 2)} [{sg(float(r['lo']), 2)}, {sg(float(r['hi']), 2)}] ({int(r['n_valid'])})"
            ln = " & ".join(parts)
        out.append(ln)
    write(f"tab_timing_{kind}", "\n".join(out))
for ref in ["gap", "test_acc"]:
    for m in ["fisher", "eff_rank_pr", "mag_t1"]:
        r = T4[("all_runs", 0.5, ref, m)]
        st(f"[timing-step4] q=0.5 vs {ref} {m}: {float(r['median']):+.2f} [{float(r['lo']):+.2f},{float(r['hi']):+.2f}] ({r['n_valid']})")

with open("/homes/lbastian/latent-dynamics/paper_v2_figs/paper_stats3.txt", "w") as f:
    f.write("\n".join(STATS) + "\n")
