"""Draft-5 tables from analysis_v2/ext5 (Amendment 6, post hoc). Writes paper_v2/tables/tab_conf_ctrl.tex (main text),
tab_ext5.tex (appendix) and paper_stats5.txt; also patches two draft-4 tables (GBM bold cell, placebo column count).
CPU only, csv + numpy."""
from common import *

TAB = f"{BIT}/paper_v2/tables"
E5 = f"{A2}/ext5"
STATS = []


def st(*a):
    s = " ".join(str(x) for x in a)
    STATS.append(s)
    print(s)


def sg(x, nd=3):
    s = f"{abs(x):.{nd}f}"
    if x < 0 and float(s) != 0:
        return "$-$" + s
    return "+" + s if float(s) != 0 else s


def R(name):
    return list(csv.DictReader(open(f"{E5}/{name}")))


def f(r, k):
    return float(r[k])


def write(name, text):
    open(f"{TAB}/{name}.tex", "w").write(text + "\n")
    print("wrote", name)


# ---------------------------------------------------------------- B, C, E: blocks vs B1 and contrasts with G
H = R("B_harm_pvalues.csv")
C = R("C_placebo_contrast.csv")
E = R("E_curve_block_placebo.csv")


def agg(rows, dk, pk):
    d = [f(r, dk) for r in rows]
    return np.mean(d), sum(x < 0 for x in d), sum(f(r, pk) < 0.05 for r in rows)


rowsG = [r for r in H if r["X"] == "G"]
rows4 = [r for r in H if r["X"] == "compact4"]
rp = [r for r in C if r["placebo"] == "RP18"]
nz = [r for r in C if r["placebo"] == "NZ18"]
cvb = [r for r in E if r["contrast"] == "CV18 vs B1"]
gcv = [r for r in E if r["contrast"] == "G minus CV18"]
lines = []
spec = [("Full $G$ (18)", agg(rowsG, "delta", "p_neg_holm"), None),
        ("Compact-4 (4)", agg(rows4, "delta", "p_neg_holm"), None),
        ("Random proj. (18)", agg(rp, "placebo_dR2_vs_B1", "placebo_p_neg_holm"), agg(rp, "G_minus_placebo", "p_G_worse_holm")),
        ("iid noise (18)", agg(nz, "placebo_dR2_vs_B1", "placebo_p_neg_holm"), agg(nz, "G_minus_placebo", "p_G_worse_holm")),
        ("Curve stats CV18 (18)", agg(cvb, "delta", "p_neg_holm"), agg(gcv, "delta", "p_neg_holm"))]
for name, (m, n, h), con in spec:
    if con is None:
        cs = "--- & --- & ---"
    else:
        cm, cn, ch = con
        if name.startswith("Curve"):  # G minus CV18: delta is G - CV18; "G worse" = negative
            cs = f"{sg(cm)} & {cn}/10 & {ch}/10"
        else:
            cs = f"{sg(cm)} & {cn}/10 & {ch}/10"
    lines.append(f"{name} & {sg(m)} & {n}/10 & {h}/10 & {cs} \\\\")
    st("[ctrl]", name, f"mean {m:.4f} neg {n} holm {h}", None if con is None else f"contrast mean {con[0]:.4f} neg {con[1]} holm {con[2]}")
L = [r"\begin{tabular}{lcccccc}", r"\toprule",
     r" & \multicolumn{3}{c}{Block vs.\ $B_1$ (ridge)} & \multicolumn{3}{c}{$G$ minus block} \\",
     r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
     r"Block (columns) & Mean $\dR$ & $<0$ & Holm & Mean & $G$ worse & Holm \\", r"\midrule"] + lines + [r"\bottomrule", r"\end{tabular}"]
write("tab_conf_ctrl", "\n".join(L))

# ---------------------------------------------------------------- A: subsampling
S = R("A_subsample_summary.csv")


def sub(K, b):
    rows = [r for r in S if r["K"] == str(K) and r["block"] == b]
    return dict(mean=np.mean([f(r, "mean") for r in rows]), med=np.mean([f(r, "median") for r in rows]),
                neg=np.mean([f(r, "frac_neg") for r in rows]), p5=np.mean([f(r, "p5") for r in rows]), p95=np.mean([f(r, "p95") for r in rows]))


s28G, s28c, s28n = sub(28, "G"), sub(28, "c4"), sub(28, "NZ18")
V = R("A_conf_vs_old27.csv")
pctG = [f(r, "frac_old_draws_le_conf") for r in V if r["block"] == "G"]
st("[sub] K=28 G", s28G, "c4", s28c, "NZ18", s28n, "conf percentiles", min(pctG), max(pctG), np.mean(pctG))
st("[sub] K=14,49,69 G mean:", [round(sub(k, "G")["mean"], 3) for k in (14, 49, 69)], "NZ18:", [round(sub(k, "NZ18")["mean"], 3) for k in (14, 28, 49, 69)])
acc_sp = [(f(r, "p5"), f(r, "p95")) for r in S if r["K"] == "28" and r["block"] == "G" and r["target"] == "acc_15kw"]
gap_sp = [(f(r, "p5"), f(r, "p95")) for r in S if r["K"] == "28" and r["block"] == "G" and r["target"] == "gap_15kw"]
st("[sub] K=28 G 5-95% mean over cells acc", np.mean(acc_sp, 0), "gap", np.mean(gap_sp, 0))

# within architecture
W = R("A_conf_i_within_arch.csv")


def wa(fit, X):
    rows = [r for r in W if r["fit"] == fit and r["X"] == X]
    return np.median([f(r, "delta") for r in rows]), sum(f(r, "delta") < 0 for r in rows)


mlpG, cnnG, mlp4, cnn4 = wa("within_mlp", "full G"), wa("within_cnn", "full G"), wa("within_mlp", "compact4"), wa("within_cnn", "compact4")
ixG, ix4 = wa("pooled_interaction_B1x", "full G"), wa("pooled_interaction_B1x", "compact4")
st("[arch]", mlpG, cnnG, mlp4, cnn4, ixG, ix4)

# low-dim
Ld = R("A_conf_ii_lowdim.csv")
low = {b: np.mean([f(r, "delta") for r in Ld if r["block"] == b]) for b in sorted(set(r["block"] for r in Ld))}
lowpos = sum(f(r, "lo90") > 0 for r in Ld)
st("[lowdim]", {k: round(v, 3) for k, v in low.items()}, "cells lo90>0:", lowpos)

# alpha
A = R("A_conf_iii_alpha.csv")
wideG = np.mean([f(r, "delta") for r in A if r["alpha"] == "nested" and r["X"] == "full G"])
wide4 = np.mean([f(r, "delta") for r in A if r["alpha"] == "nested" and r["X"] == "compact4"])
st("[alpha] wide nested mean G", wideG, "c4", wide4)

# precision
P = R("D_precision.csv")
D = R("D_extrapolation.csv")
hwG = [f(r, "halfwidth90") for r in P if r["X"] == "G"]
hwA = [f(r, "halfwidth90") for r in P if r["X"] == "G" and r["target"] == "acc_15kw"]
hwP = [f(r, "halfwidth90") for r in P if r["X"] == "G" and r["target"] == "gap_15kw"]
nbelow4 = sum(f(r, "halfwidth90") < f(r, "margin") for r in P if r["X"] == "compact4")
Kn = [f(r, "K_needed_fit") for r in D if r["X"] == "G"]
st("[prec] G hw", min(hwG), max(hwG), "acc", min(hwA), max(hwA), "gap", min(hwP), max(hwP), "c4 cells below margin", nbelow4, "K needed median", np.median(Kn), "range", min(Kn), max(Kn))
SH = R("D_sim_halfwidth.csv")
simk = {k: np.mean([f(r, "hw90_mean") for r in SH if r["X"] == "G" and r["K"] == str(k)]) for k in (28, 56, 112, 224, 448)}
st("[prec] simulated mean G half-width by K", {k: round(v, 3) for k, v in simk.items()})

hp = lambda rows, X: ", ".join(f"{r['target'][:3]} {r['cutoff']}: {f(r, 'p_neg_holm'):.3f}" for r in rows if f(r, "p_neg_holm") < 0.05)
st("[harm] G", hp(rowsG, "G"), "| c4", hp(rows4, "c4"))
st("[harm] unadjusted p<.05 G", sum(f(r, "p_neg") < .05 for r in rowsG), "c4", sum(f(r, "p_neg") < .05 for r in rows4),
   "prereg p+ Holm min", min(f(r, "pos_holm_prereg") for r in H))

# ---------------------------------------------------------------- appendix table
def row(a, b):
    return f"{a} & {b} \\\\"


T = [r"\begin{tabular}{p{2.9cm}p{12.6cm}}", r"\toprule", r"Analysis & Result (ridge, mean $\dR$ over the ten cells unless stated) \\", r"\midrule"]
T.append(row("Subsampled old grid, 28 configurations (480 draws; 27 training configurations per fit)",
             f"Full $G$ mean {sg(s28G['mean'])} (mean of cell medians {sg(s28G['med'])}); {100 * s28G['neg']:.0f}\\% of draws below zero; 5--95\\% range of the draws, narrowest to widest cell, "
             f"[{sg(max(a for a, b in acc_sp), 2)}, {sg(min(b for a, b in acc_sp), 2)}] to [{sg(min(a for a, b in acc_sp), 2)}, {sg(max(b for a, b in acc_sp), 2)}] (accuracy cells) and [{sg(max(a for a, b in gap_sp), 2)}, {sg(min(b for a, b in gap_sp), 2)}] to [{sg(min(a for a, b in gap_sp), 2)}, {sg(max(b for a, b in gap_sp), 2)}] (gap cells). "
             f"Compact-4 {sg(s28c['mean'])}, iid noise block {sg(s28n['mean'])} ({100 * s28n['neg']:.0f}\\% below zero). "
             f"The confirmatory value {sg(np.mean([f(r, 'conf_dR2') for r in V if r['block'] == 'G']))} lies at the {100 * min(pctG):.0f}th to {100 * max(pctG):.0f}th percentile of the draws in the ten cells."))
T.append(row("Same, other sizes", f"Full $G$ mean {sg(sub(14, 'G')['mean'])} (14 configurations), {sg(sub(49, 'G')['mean'])} (49), {sg(sub(69, 'G')['mean'])} (all 69); noise block {sg(sub(14, 'NZ18')['mean'])}, {sg(sub(49, 'NZ18')['mean'])}, {sg(sub(69, 'NZ18')['mean'])}."))
T.append(row("Within architecture (14 configurations, 42 runs; median over cells)",
             f"Full $G$: MLP {sg(mlpG[0])} ({mlpG[1]}/10 negative), CNN {sg(cnnG[0])} ({cnnG[1]}/10); compact-4: MLP {sg(mlp4[0])}, CNN {sg(cnn4[0])}. "
             f"Pooled fit with architecture-by-curve interactions in $B_1$: full $G$ {sg(ixG[0])}, compact-4 {sg(ix4[0])}."))
T.append(row("Low-dimensional blocks (28 configurations)",
             f"Test-probe Fisher {sg(low['te_fisher'])}, effective rank {sg(low['te_eff_rank_pr'])}, Mag$(t{{=}}1)$ {sg(low['te_mag_t1'])}, $H_1$ top-5 {sg(low['te_h1_top5'])}; "
             f"Fisher plus effective rank {sg(low['fisher+eff_rank'])}; in-fold PCA of $G$: 1 component {sg(low['PCA1(G,infold)'])}, 2 components {sg(low['PCA2(G,infold)'])}; "
             f"1 component of compact-4 {sg(low['PCA1(compact4,infold)'])}. A 90\\% lower bound above zero in {lowpos} of 80 cells ($+0.003$)."))
T.append(row("Ridge penalty",
             f"Nested grid widened to $10^{{-4}}$ to $10^{{6}}$: full $G$ {sg(wideG)} (unchanged, negative in 10/10), compact-4 {sg(wide4)}. The top penalty of the original grid was chosen in 0 of 28 folds in every cell."))
T.append(row("Harm tests (post hoc, one-sided, Holm over ten cells)",
             f"Full $G$: $p<.05$ in {sum(f(r, 'p_neg_holm') < .05 for r in rowsG)}/10 (accuracy at cutoffs 300 and 1000, gap at 300; {sum(f(r, 'p_neg') < .05 for r in rowsG)}/10 unadjusted). "
             f"Compact-4: {sum(f(r, 'p_neg_holm') < .05 for r in rows4)}/10 (accuracy at 30 and 300; {sum(f(r, 'p_neg') < .05 for r in rows4)}/10 unadjusted). "
             f"The pre-registered one-sided tests of a positive increment have Holm $p=1.0$ in all 20 cells."))
T.append(row("Placebo contrast (post hoc, draw-mean bootstrap)",
             f"$G$ minus random projections is negative in 10/10 cells ({sum(f(r, 'hi95') < 0 for r in rp)}/10 with a 95\\% CI below zero, {sum(f(r, 'p_G_worse_holm') < .05 for r in rp)}/10 after Holm); "
             f"minus noise negative in {sum(f(r, 'G_minus_placebo') < 0 for r in nz)}/10 ({sum(f(r, 'p_G_worse_holm') < .05 for r in nz)}/10 after Holm). The placebos themselves lower $\\Rsq$: noise in 10/10 cells after Holm, random projections in {sum(f(r, 'placebo_p_neg_holm') < .05 for r in rp)}/10."))
T.append(row("Precision at 28 configurations (plug-in resampling, post hoc)",
             f"90\\% half-width of full $G$: {min(hwA):.2f} to {max(hwA):.2f} (accuracy, margin 0.03) and {min(hwP):.2f} to {max(hwP):.2f} (gap, margin 0.05), never below the margin. "
             f"Simulated mean half-width at 28, 56, 112, 224 configurations: {simk[28]:.3f}, {simk[56]:.3f}, {simk[112]:.3f}, {simk[224]:.3f}. "
             f"Configurations needed for half-width below the margin (power-law fit, true increment zero): {min(Kn):.0f} to {max(Kn):.0f}, median about 115 ({np.median(Kn):.1f}). "
             f"Compact-4 half-width is below the margin in {nbelow4} cells."))
T.append(row("Curve-statistics block CV18",
             f"Eighteen columns from the train and test loss and accuracy curves (value at one tenth of the cutoff, slope and standard deviation over snapshots up to the cutoff), without latents: "
             f"$B_1{{\\to}}B_1{{+}}\\mathrm{{CV18}}$ {sg(spec[4][1][0])}, negative in 10/10 cells, Holm $p<.05$ in {spec[4][1][2]}/10; $G$ minus CV18 {sg(spec[4][2][0])}, with $G$ worse in {spec[4][2][1]}/10 and Holm-significant only for accuracy at cutoff 1000 ($-0.062$)."))
T += [r"\bottomrule", r"\end{tabular}"]
write("tab_ext5", "\n".join(T))

# ---------------------------------------------------------------- patch draft-4 tables
p = f"{TAB}/tab_conf_primary.tex"
s = open(p).read()
old = "$-$0.053 [$-$0.124, $-$0.017]"
if old in s and "\\textbf{" + old not in s:
    s = s.replace(old, "\\textbf{" + old + "}")
    open(p, "w").write(s)
    print("bolded GBM acc cutoff-300 cell")
p = f"{TAB}/tab_conf_placebo.tex"
s = open(p).read()
if s.startswith("\\begin{tabular}{llllll}"):
    s = s.replace("\\begin{tabular}{llllll}", "\\begin{tabular}{lllll}", 1)
    open(p, "w").write(s)
    print("fixed placebo column spec")
open(f"{os.path.dirname(os.path.abspath(__file__))}/paper_stats5.txt", "w").write("\n".join(str(x) for x in STATS) + "\n")
