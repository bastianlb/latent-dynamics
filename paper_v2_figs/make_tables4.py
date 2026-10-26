"""Draft-4 tables from analysis_v2/confirm/results (pre-registered confirmatory study), ext3 (regimes) and ext4 (Amendment 5).
Writes paper_v2/tables/*.tex and paper_stats4.txt. CPU only, no pandas (csv + numpy)."""
from common import *

TAB = f"{BIT}/paper_v2/tables"
CF = f"{A2}/confirm/results"
E3 = f"{A2}/ext3"
E4 = f"{A2}/ext4"
CUTS = [10, 30, 100, 300, 1000]
TG = {"acc_15kw": "Test acc.", "gap_15kw": "Gap"}
MARG = {"acc_15kw": 0.03, "gap_15kw": 0.05}
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


def ci(lo, hi, nd=3):
    return f"[{sg(lo, nd)}, {sg(hi, nd)}]"


def bf(s, on):
    return f"\\textbf{{{s}}}" if on else s


def write(name, text):
    with open(f"{TAB}/{name}.tex", "w") as f:
        f.write(text + "\n")
    print("wrote", name)


def pfmt(p):
    return "<.001" if p < 0.001 else f"{p:.3f}" if p < 0.1 else f"{p:.2f}"


# ================================================================ confirmatory primary table
P = read_csv(f"{CF}/primary_ridge.csv")
PR = {(r["X"], r["target"], int(r["cutoff"])): r for r in P}
GB = {(r["X"], r["target"], int(r["cutoff"])): r for r in read_csv(f"{CF}/secondary_b_gbm.csv")}
L = [r"\begin{tabular}{llcccccccc}", r"\toprule",
     r" & & & \multicolumn{2}{c}{Ridge, full $G$ (18)} & \multicolumn{2}{c}{Ridge, compact-4} & \multicolumn{2}{c}{GBM, full $G$} \\",
     r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
     r"Target & Cutoff & $\Rsq(B_1)$ & $\dR$ [90\% CI] & Holm $p$ & $\dR$ [90\% CI] & Holm $p$ & $\dR$ [90\% CI] & $\Rsq(B_1)$ \\", r"\midrule"]
for tgt in TG:
    for c in CUTS:
        g, c4, gb = PR[("G", tgt, c)], PR[("compact4", tgt, c)], GB[("G", tgt, c)]
        def cell(r):
            lo, hi = fnum(r["lo90"]), fnum(r["hi90"])
            return bf(f"{sg(fnum(r['delta']))} {ci(lo, hi)}", hi < 0)
        eqc = c4["equivalent_holm"] == "1"
        eqm = "$^{\\mathrm{E}}$" if eqc else ""
        L.append(f"{TG[tgt] if c == 10 else ''} & {c} & {fnum(g['r2_B1']):.3f} & {cell(g)} & {pfmt(fnum(g['tost_holm']))} & {cell(c4)} & "
                 f"{bf(pfmt(fnum(c4['tost_holm'])), eqc)}{eqm} & "
                 f"{sg(fnum(gb['delta']))} {ci(fnum(gb['lo90']), fnum(gb['hi90']))} & {fnum(gb['r2_B1']):.3f} \\\\")
    L.append(r"\midrule")
L[-1] = r"\bottomrule"
L.append(r"\end{tabular}")
write("tab_conf_primary", "\n".join(L))
for X in ["G", "compact4"]:
    neq = sum(PR[(X, t, c)]["equivalent_holm"] == "1" for t in TG for c in CUTS)
    nun = sum(fnum(PR[(X, t, c)]["tost_p"]) <= 0.05 for t in TG for c in CUTS)
    npos = sum(PR[(X, t, c)]["positive_holm"] == "1" for t in TG for c in CUTS)
    nbelow = sum(fnum(PR[(X, t, c)]["hi90"]) < 0 for t in TG for c in CUTS)
    nneg = sum(fnum(PR[(X, t, c)]["delta"]) < 0 for t in TG for c in CUTS)
    st(f"[conf primary] ridge {X}: equivalent Holm {neq}/10, unadjusted {nun}/10, positive Holm {npos}/10, "
       f"point<0 {nneg}/10, 90% CI wholly <0 {nbelow}/10")
st("[conf primary] R2(B1) acc", min(fnum(PR[("G", 'acc_15kw', c)]['r2_B1']) for c in CUTS), max(fnum(PR[("G", 'acc_15kw', c)]['r2_B1']) for c in CUTS),
   "gap", min(fnum(PR[("G", 'gap_15kw', c)]['r2_B1']) for c in CUTS), max(fnum(PR[("G", 'gap_15kw', c)]['r2_B1']) for c in CUTS))
st("[conf primary] G delta range", min(fnum(PR[("G", t, c)]['delta']) for t in TG for c in CUTS), max(fnum(PR[("G", t, c)]['delta']) for t in TG for c in CUTS))
st("[conf primary] c4 delta range", min(fnum(PR[("compact4", t, c)]['delta']) for t in TG for c in CUTS), max(fnum(PR[("compact4", t, c)]['delta']) for t in TG for c in CUTS))
st("[conf gbm] equivalent/inconclusive/positive:",
   {X: {lab: sum(GB[(X, t, c)]['label'].startswith(lab) for t in TG for c in CUTS) for lab in ['equivalent', 'inconclusive', 'positive']} for X in ['G', 'compact4']})
for X in ['G', 'compact4']:
    st(f"[conf gbm] {X} delta range", min(fnum(GB[(X, t, c)]['delta']) for t in TG for c in CUTS), max(fnum(GB[(X, t, c)]['delta']) for t in TG for c in CUTS))
    st(f"[conf gbm] {X} gap1000", GB[(X, 'gap_15kw', 1000)]['delta'], GB[(X, 'gap_15kw', 1000)]['lo90'], GB[(X, 'gap_15kw', 1000)]['hi90'])

# ================================================================ confirmatory GBM appendix table (all cells, both blocks)
L = [r"\begin{tabular}{llcccc}", r"\toprule",
     r"Target & Cutoff & $\Rsq(B_1)$ & Block & $\dR$ [90\% CI] & Class \\", r"\midrule"]
for tgt in TG:
    for c in CUTS:
        for X, XL in [("G", "full $G$"), ("compact4", "compact-4")]:
            r = GB[(X, tgt, c)]
            lab_ = r['label'].replace(' (possible harm)', '$^\\downarrow$')
            L.append(f"{TG[tgt] if (c == 10 and X == 'G') else ''} & {c if X == 'G' else ''} & {fnum(r['r2_B1']):.3f} & {XL} & "
                     f"{sg(fnum(r['delta']))} {ci(fnum(r['lo90']), fnum(r['hi90']))} & {lab_} \\\\")
    L.append(r"\midrule")
L[-1] = r"\bottomrule"
L.append(r"\end{tabular}")
write("tab_conf_gbm", "\n".join(L))

# ================================================================ confirmatory placebo appendix table
D = read_csv(f"{CF}/secondary_d_placebo.csv")
L = [r"\begin{tabular}{llllll}", r"\toprule",
     r"Target & Cutoff & Block & $\dR$ [95\% CI] & $\Rsq(B_1{+}G)-\Rsq(B_1{+}\text{block})$ [95\% CI] \\", r"\midrule"]
nneg_rp = nneg_nz = nex_rp = nex_nz = 0
for tgt in TG:
    for c in CUTS:
        for blk, bl in [("RP18", "random proj. (18)"), ("NZ18", "noise (18)")]:
            r = [x for x in D if x["target"] == tgt and int(x["cutoff"]) == c and x["block"] == blk][0]
            gp, glo, ghi = fnum(r["G_minus_placebo"]), fnum(r["GvsP_lo95"]), fnum(r["GvsP_hi95"])
            if blk == "RP18":
                nneg_rp += gp < 0; nex_rp += ghi < 0
            else:
                nneg_nz += gp < 0; nex_nz += ghi < 0
            L.append(f"{TG[tgt] if (c == 10 and blk == 'RP18') else ''} & {c if blk == 'RP18' else ''} & {bl} & "
                     f"{sg(fnum(r['dR2_mean']))} {ci(fnum(r['lo95_mean']), fnum(r['hi95_mean']))} & {bf(sg(gp) + ' ' + ci(glo, ghi), ghi < 0)} \\\\")
    L.append(r"\midrule")
L[-1] = r"\bottomrule"
L.append(r"\end{tabular}")
write("tab_conf_placebo", "\n".join(L))
st(f"[conf placebo] G - RP18 negative in {nneg_rp}/10 cells (95% CI wholly <0 in {nex_rp}); G - NZ18 negative in {nneg_nz}/10 (CI wholly <0 in {nex_nz})")
for blk in ["RP18", "NZ18"]:
    v = [fnum(x["G_minus_placebo"]) for x in D if x["block"] == blk and int(x["cutoff"]) == 1000]
    st(f"[conf placebo] {blk} G-placebo at cutoff 1000 (acc, gap):", v)
    v = [fnum(x["dR2_mean"]) for x in D if x["block"] == blk]
    st(f"[conf placebo] {blk} dR2 mean range:", min(v), max(v))

# ================================================================ config table (from PREREG section 6)
cfgs = []
for line in open(f"{A2}/confirm/PREREG.txt"):
    m = re.match(r"^(c\d\d)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(mlp|cnn)\s*$", line)
    if m:
        cfgs.append(m.groups())
assert len(cfgs) == 28, len(cfgs)


def fmtv(x):
    if x == "full":
        return "full"
    return x


def e(x):  # 2.4e-05 -> LaTeX
    if "e" in x:
        a, b = x.split("e")
        return f"${a}{{\\cdot}}10^{{{int(b)}}}$"
    return x


rows = [f"{c} & {n} & {e(w)} & {ln} & {e(lr)} & {a.upper()}" for c, n, w, ln, lr, a in cfgs]
L = [r"\begin{tabular}{llllll@{\hspace{2.2em}}llllll}", r"\toprule",
     r"ID & $n$ & wd & noise & lr & arch & ID & $n$ & wd & noise & lr & arch \\", r"\midrule"]
for i in range(14):
    L.append(rows[i] + " & " + rows[i + 14] + r" \\")
L += [r"\bottomrule", r"\end{tabular}"]
write("tab_conf_cfg", "\n".join(L))
st("[conf cfg] arch counts:", {a: sum(x[5] == a for x in cfgs) for a in ["mlp", "cnn"]},
   " n range:", min(int(x[1]) if x[1] != "full" else 60000 for x in cfgs), max(int(x[1]) if x[1] != "full" else 60000 for x in cfgs),
   " zero wd:", sum(x[2] == "0" for x in cfgs), " zero noise:", sum(x[3] == "0" for x in cfgs))

# ================================================================ per-architecture table
A = read_csv(f"{CF}/exploratory_per_arch.csv")
L = [r"\begin{tabular}{llccccccc}", r"\toprule",
     r" & & \multicolumn{3}{c}{MLP (42 runs)} & \multicolumn{3}{c}{CNN (42 runs)} \\", r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
     r"Target & Cutoff & $B_1$ & $B_1{+}G$ & $B_1{+}c_4$ & $B_1$ & $B_1{+}G$ & $B_1{+}c_4$ \\", r"\midrule"]
nh = {"mlp": 0, "cnn": 0}
for tgt in TG:
    for c in CUTS:
        m = [x for x in A if x["target"] == tgt and int(x["cutoff"]) == c and x["arch"] == "mlp"][0]
        n = [x for x in A if x["target"] == tgt and int(x["cutoff"]) == c and x["arch"] == "cnn"][0]
        for k, r in (("mlp", m), ("cnn", n)):
            nh[k] += fnum(r["r2_B1G"]) < fnum(r["r2_B1"])
        L.append(f"{TG[tgt] if c == 10 else ''} & {c} & " + " & ".join(f"{fnum(r[k]):.3f}" for r in (m, n) for k in ("r2_B1", "r2_B1G", "r2_B1c4")) + r" \\")
    L.append(r"\midrule")
L[-1] = r"\bottomrule"
L.append(r"\end{tabular}")
write("tab_conf_arch", "\n".join(L))
st(f"[conf arch] cells (of 10) where pooled R2(B1+G) < R2(B1): mlp {nh['mlp']}, cnn {nh['cnn']}")
for k in ("mlp", "cnn"):
    rr = [x for x in A if x["arch"] == k]
    st(f"[conf arch] {k} R2(B1+G)-R2(B1) range:", min(fnum(x['r2_B1G']) - fnum(x['r2_B1']) for x in rr), max(fnum(x['r2_B1G']) - fnum(x['r2_B1']) for x in rr))

# ================================================================ transfer summary
T = read_csv(f"{CF}/secondary_a_transfer.csv")
L = [r"\begin{tabular}{lllcccc}", r"\toprule", r"Learner & Test set & Block & Equivalent & Positive & Inconclusive & Inconclusive, $\dR<0$ \\", r"\midrule"]
for mdl in ["ridge", "gbm"]:
    for sub, sl in [("all", "84 runs"), ("mlp_only", "42 MLP runs")]:
        for X, XL in [("G", "full $G$"), ("compact4", "compact-4")]:
            rr = [x for x in T if x["model"] == mdl and x["subset"] == sub and x["X"] == X]
            ne = sum(x["label"].startswith("equivalent") for x in rr)
            npz = sum(x["label"].startswith("positive") for x in rr)
            ni = sum(x["label"].startswith("inconclusive") for x in rr)
            nin = sum(x["label"].startswith("inconclusive") and fnum(x["delta"]) < 0 for x in rr)
            L.append(f"{mdl.upper() if mdl == 'gbm' else 'Ridge'} & {sl} & {XL} & {ne} & {npz} & {ni} & {nin} \\\\")
            st(f"[conf transfer] {mdl} {sub} {X}: E {ne} P {npz} I {ni} (I with dR2<0: {nin}), n={len(rr)}")
    L.append(r"\midrule" if mdl == "ridge" else "")
L[-1] = r"\bottomrule"
L.append(r"\end{tabular}")
write("tab_conf_transfer", "\n".join(L))
for x in T:
    if x["label"].startswith("positive"):
        st(f"[conf transfer positive] {x['model']} {x['subset']} {x['X']} {x['target']} {x['cutoff']}: R2(B1) {fnum(x['r2_B1']):.3f} dR2 {fnum(x['delta']):.3f} 90% [{fnum(x['lo90']):.3f},{fnum(x['hi90']):.3f}]")

# ================================================================ regimes table (exploratory ridge, MNIST)
MS = {(r["target"], int(r["cutoff"])): r for r in read_csv(f"{E3}/mnist/margin_mstar.csv") if r["model"] == "ridge"}
HO = {(r["target"], int(r["cutoff"])): r for r in read_csv(f"{E3}/mnist/multiplicity_holm.csv") if r["kind"] == "iid"}
RF = {(r["target"], int(r["cutoff"])): r for r in read_csv(f"{E3}/mnist/design_refit.csv")}
L = [r"\begin{tabular}{llccccc|c}", r"\toprule", r"Target (margin) & Regime & 10 & 30 & 100 & 300 & 1000 & Equiv. \\", r"\midrule"]
tot = {"plain": 0, "holm": 0, "refit": 0}
for tgt, lab in [("acc_15kw", "Test acc. ($\\pm.03$)"), ("gap_15kw", "Gap ($\\pm.05$)")]:
    m = MARG[tgt]
    d = [fnum(MS[(tgt, c)]["delta"]) for c in CUTS]
    L.append(f"{lab} & $\\dR$ & " + " & ".join(sg(x) for x in d) + r" & \\")
    pl = [fnum(MS[(tgt, c)]["mstar_iid"]) <= m for c in CUTS]
    L.append(" & plain $m^\\ast$ & " + " & ".join(bf(f"{fnum(MS[(tgt, c)]['mstar_iid']):.3f}", p) for c, p in zip(CUTS, pl)) + f" & {sum(pl)}/5 \\\\")
    hl = [HO[(tgt, c)]["equivalent_holm"] == "1" for c in CUTS]
    L.append(" & Holm $p$ & " + " & ".join(bf(pfmt(fnum(HO[(tgt, c)]["holm_p"])), h) for c, h in zip(CUTS, hl)) + f" & {sum(hl)}/5 \\\\")
    rl = [RF[(tgt, c)]["class_refit"] == "equivalent" for c in CUTS]
    L.append(" & refit $m^\\ast$ & " + " & ".join(bf(f"{fnum(RF[(tgt, c)]['mstar_refit']):.3f}", r) for c, r in zip(CUTS, rl)) + f" & {sum(rl)}/5 \\\\")
    tot["plain"] += sum(pl); tot["holm"] += sum(hl); tot["refit"] += sum(rl)
    L.append(r"\midrule")
L[-1] = r"\bottomrule"
L.append(r"\end{tabular}")
write("tab_regimes", "\n".join(L))
st("[regimes] MNIST ridge equivalent cells (of 10): plain m*<=margin", tot["plain"], " Holm", tot["holm"], " refit (class_refit equivalent)", tot["refit"])
# Fashion
MSF = {(r["target"], int(r["cutoff"])): r for r in read_csv(f"{E3}/fashion/margin_mstar.csv") if r["model"] == "ridge"}
HOF = {(r["target"], int(r["cutoff"])): r for r in read_csv(f"{E3}/fashion/multiplicity_holm.csv") if r["kind"] == "iid"}
RFF = {(r["target"], int(r["cutoff"])): r for r in read_csv(f"{E3}/fashion/design_refit.csv")}
st("[regimes] Fashion ridge equivalent: plain", sum(fnum(MSF[(t, c)]["mstar_iid"]) <= MARG[t] for t in TG for c in CUTS),
   " Holm", sum(HOF[(t, c)]["equivalent_holm"] == "1" for t in TG for c in CUTS),
   " refit", sum(RFF[(t, c)]["class_refit"] == "equivalent" for t in TG for c in CUTS))

# ================================================================ controls table (placebo + compact extrapolation)
PLN = {d: read_csv(f"{E4}/{d}/A_placebo_summary.csv") for d in ["mnist", "fashion"]}


def prow(d, split, base, tgt, cut, size, block):
    rr = [r for r in PLN[d] if r["model"] == "ridge" and r["split"] == split and r["base_set"] == base and r["target"] == tgt and
          int(r["cutoff"]) == cut and r["size"] == size and r["block"] == block]
    assert len(rr) == 1, (d, split, base, tgt, cut, size, block, len(rr))
    return rr[0]


def cellg(r, d=3):
    return f"{sg(fnum(r['dR2']), d)} {ci(fnum(r['lo95']), fnum(r['hi95']), 2)}"


CROWS = [("mnist", "loco", "B0", "acc_15kw", 10, r"$B_0\!\to\!B_0{+}X$, LOCO, test acc., cutoff 10"),
         ("mnist", "loco", "B0", "gap_15kw", 10, r"$B_0\!\to\!B_0{+}X$, LOCO, gap, cutoff 10"),
         ("mnist", "leave_one_n_out", "B1", "acc_15kw", 1000, r"$B_1\!\to\!B_1{+}X$, leave-one-$n$-out, test acc., 1000"),
         ("mnist", "leave_one_n_out", "B1", "gap_15kw", 300, r"$B_1\!\to\!B_1{+}X$, leave-one-$n$-out, gap, 300"),
         ("mnist", "leave_one_wd_out", "B1", "acc_15kw", 1000, r"$B_1\!\to\!B_1{+}X$, leave-one-wd-out, test acc., 1000"),
         ("mnist", "leave_one_wd_out", "B1", "gap_15kw", 1000, r"$B_1\!\to\!B_1{+}X$, leave-one-wd-out, gap, 1000")]
L = [r"\begin{tabular}{lcccc}", r"\toprule", r"Exploratory, MNIST (305 runs) & full $G$ (18) & compact-4 & random proj. (18) & noise (18) \\", r"\midrule"]
for d, sp, bs, tgt, c, lab in CROWS:
    g = prow(d, sp, bs, tgt, c, "full G (18)", "G")
    c4 = prow(d, sp, bs, tgt, c, "compact4", "G")
    rp = prow(d, sp, bs, tgt, c, "full G (18)", "random-projection")
    nz = prow(d, sp, bs, tgt, c, "full G (18)", "gaussian-noise")
    L.append(f"{lab} & {cellg(g)} & {cellg(c4)} & {sg(fnum(rp['dR2']))} & {sg(fnum(nz['dR2']))} \\\\")
    st(f"[controls] {d} {sp} {bs} {tgt} {c}: G {fnum(g['dR2']):.3f} c4 {fnum(c4['dR2']):.3f} RP18 {fnum(rp['dR2']):.3f} NZ18 {fnum(nz['dR2']):.3f}")
L.append(r"\midrule")
L.append(r"Confirmatory (84 runs, 28 new configs) & & & & \\")
for tgt, c, lab in [("acc_15kw", 1000, r"$B_1\!\to\!B_1{+}X$, LOCO, test acc., 1000"), ("gap_15kw", 1000, r"$B_1\!\to\!B_1{+}X$, LOCO, gap, 1000")]:
    g, c4 = PR[("G", tgt, c)], PR[("compact4", tgt, c)]
    rp = [x for x in D if x["target"] == tgt and int(x["cutoff"]) == c and x["block"] == "RP18"][0]
    nz = [x for x in D if x["target"] == tgt and int(x["cutoff"]) == c and x["block"] == "NZ18"][0]
    L.append(f"{lab} & {sg(fnum(g['delta']))} {ci(fnum(g['lo95']), fnum(g['hi95']), 2)} & {sg(fnum(c4['delta']))} {ci(fnum(c4['lo95']), fnum(c4['hi95']), 2)} & {sg(fnum(rp['dR2_mean']))} & {sg(fnum(nz['dR2_mean']))} \\\\")
L += [r"\bottomrule", r"\end{tabular}"]
write("tab_controls", "\n".join(L))
# fashion placebo numbers for the text / appendix
for sp, bs, tgt, c in [("loco", "B0", "acc_15kw", 10), ("loco", "B0", "gap_15kw", 10), ("leave_one_n_out", "B1", "acc_15kw", 1000), ("leave_one_n_out", "B1", "gap_15kw", 300)]:
    g = prow("fashion", sp, bs, tgt, c, "full G (18)", "G"); rp = prow("fashion", sp, bs, tgt, c, "full G (18)", "random-projection"); nz = prow("fashion", sp, bs, tgt, c, "full G (18)", "gaussian-noise")
    st(f"[controls fashion] {sp} {bs} {tgt} {c}: G {fnum(g['dR2']):.3f} RP18 {fnum(rp['dR2']):.3f} NZ18 {fnum(nz['dR2']):.3f}")

# ================================================================ compact extrapolation (appendix)
DC = read_csv(f"{E4}/mnist/D_compact_extrapolation.csv")
L = [r"\begin{tabular}{lllccccc}", r"\toprule", r"Split & Target & Block & 10 & 30 & 100 & 300 & 1000 \\", r"\midrule"]
SPL = [("leave_one_n_out", "leave-one-$n$"), ("leave_one_wd_out", "leave-one-wd"), ("leave_one_optimizer_out", "Adam$\\to$SGD")]
for sp, sl in SPL:
    for tgt in TG:
        for vi, (v, vl) in enumerate([("compact4", "compact-4"), ("compact8", "compact-8"), ("full G", "full $G$")]):
            cells = []
            for c in CUTS:
                r = [x for x in DC if x["model"] == "ridge" and x["split"] == sp and x["target"] == tgt and int(x["cutoff"]) == c and x["variant"] == v and x["fold"] == "POOLED"][0]
                lo, hi = fnum(r["lo95"]), fnum(r["hi95"])
                cells.append(bf(sg(fnum(r["dR2"]), 2), hi < 0))
            L.append(f"{sl if (vi == 0 and tgt == 'acc_15kw') else ''} & {TG[tgt] if vi == 0 else ''} & {vl} & " + " & ".join(cells) + r" \\")
        L.append(r"\addlinespace[1pt]")
    L.append(r"\midrule")
L[-1] = r"\bottomrule"
L.append(r"\end{tabular}")
write("tab_compact_extrap", "\n".join(L))
# counts of cells with CI wholly below 0 at cutoffs >= 100 per variant/split (ridge)
for sp, sl in SPL:
    for v in ["compact4", "compact8", "full G"]:
        n = sum(fnum(r["hi95"]) < 0 for r in DC if r["model"] == "ridge" and r["split"] == sp and r["variant"] == v and r["fold"] == "POOLED" and int(r["cutoff"]) >= 100)
        st(f"[compact extrap] ridge {sp} {v}: cells (of 6, cutoff>=100) with 95% CI wholly < 0: {n}")

# ================================================================ within-config RMSE
WC = read_csv(f"{E4}/mnist/E_within_config_rmse.csv")
WF = read_csv(f"{E4}/fashion/E_within_config_rmse.csv")
L = [r"\begin{tabular}{llccccccc}", r"\toprule", r"Data, target & Cutoff & SD & null RMSE & $B_0^c$ RMSE & $B_0^c{+}G$ RMSE & gain [95\% CI] & explained SD [95\% CI] \\", r"\midrule"]
for ds, dl, W in [("mnist", "MNIST", WC), ("fashion", "Fashion", WF)]:
    for tgt in TG:
        for c in CUTS:
            b = [x for x in W if x["target"] == tgt and int(x["cutoff"]) == c and x["model"] == "B0c"][0]
            g = [x for x in W if x["target"] == tgt and int(x["cutoff"]) == c and x["model"] == "B0c+G"][0]
            gl, gh = fnum(g["rmse_gain_lo95"]), fnum(g["rmse_gain_hi95"])
            esd = ci(fnum(g['explained_sd_lo95']), fnum(g['explained_sd_hi95']), 2).replace('$-$', '-')
            L.append(f"{dl + ' ' + TG[tgt].lower() if c == 10 else ''} & {c} & {fnum(b['within_sd_ddof1_pts']):.2f} & {fnum(b['null_rmse_pts']):.2f} & {fnum(b['rmse_pts']):.2f} & {fnum(g['rmse_pts']):.2f} & "
                     f"{bf(sg(fnum(g['rmse_gain_vs_B0c_pts']), 2) + ' ' + ci(gl, gh, 2), gl > 0)} & {fnum(g['explained_sd_pts']):.2f} {esd} \\\\")
        L.append(r"\midrule")
L[-1] = r"\bottomrule"
L.append(r"\end{tabular}")
write("tab_withincfg", "\n".join(L))
for ds, W in [("mnist", WC), ("fashion", WF)]:
    for tgt in TG:
        for c in CUTS:
            g = [x for x in W if x["target"] == tgt and int(x["cutoff"]) == c and x["model"] == "B0c+G"][0]
            st(f"[within-cfg] {ds} {tgt} {c}: gain {fnum(g['rmse_gain_vs_B0c_pts']):.3f} [{fnum(g['rmse_gain_lo95']):.3f},{fnum(g['rmse_gain_hi95']):.3f}] pts; explained SD {fnum(g['explained_sd_pts']):.2f}")

# ================================================================ hard cases
HC = {d: read_csv(f"{E4}/{d}/B_hardcase.csv") for d in ["mnist", "fashion"]}
L = [r"\begin{tabular}{lllcccc}", r"\toprule", r"Data (configs) & Learner & Block & $\Rsq(B_1)$ & gap, cutoff 1000 & gap, cutoff 100 & acc., cutoff 100 \\", r"\midrule"]
for ds, dl in [("mnist", "MNIST (23)"), ("fashion", "Fashion (12)")]:
    for mdl in ["ridge", "gbm"]:
        for v, vl in [("G full", "full $G$"), ("compact4", "compact-4"), ("placebo RP18 (mean 3 draws)", "random proj. (18)")]:
            cells = []
            for tgt, c in [("gap_15kw", 1000), ("gap_15kw", 100), ("acc_15kw", 100)]:
                r = [x for x in HC[ds] if x["subset"] == "hard_tercile_B1err" and x["model"] == mdl and x["target"] == tgt and int(x["cutoff"]) == c and x["variant"] == v][0]
                cells.append(f"{sg(fnum(r['dR2']), 2)} {ci(fnum(r['lo95']), fnum(r['hi95']), 2)}")
                if tgt == "gap_15kw" and c == 1000:
                    r2 = fnum(r["r2_B1"])
            L.append(f"{dl if (mdl == 'ridge' and v == 'G full') else ''} & {mdl.upper() if mdl == 'gbm' else 'Ridge'} & {vl} & {r2:.2f} & " + " & ".join(cells) + r" \\")
    L.append(r"\midrule")
L[-1] = r"\bottomrule"
L.append(r"\end{tabular}")
write("tab_hardcase", "\n".join(L))

with open("/homes/lbastian/latent-dynamics/paper_v2_figs/paper_stats4.txt", "w") as f:
    f.write("\n".join(STATS) + "\n")

# ================================================================ pre-registered sentences (all 20 cells) for the appendix
def tex_sentence(s):
    s = s.replace("_", r"\_").replace("+-", r"$\pm$").replace("R^2", r"$R^2$").replace("B1+full G", r"$B_1{+}$full $G$").replace("B1+compact4", r"$B_1{+}$compact4")
    s = s.replace("of B1 and", "of $B_1$ and").replace("%", r"\%")
    s = re.sub(r"(?<![\w$])-(?=0\.)", "$-$", s)
    return s


L = []
for X, XL in [("G", "Family F (full $G$)"), ("compact4", "Family C (compact-4)")]:
    L.append(r"\textbf{" + XL + r".}" + "\n" + r"\begin{itemize}\setlength{\itemsep}{1pt}")
    for tgt in TG:
        for c in CUTS:
            L.append(r"\item " + tex_sentence(PR[(X, tgt, c)]["sentence"]))
    L.append(r"\end{itemize}")
write("conf_sentences", "\n".join(L))
