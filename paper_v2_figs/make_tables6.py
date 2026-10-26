"""Draft-6 tables and statistics: both pre-registered confirmatory studies (MNIST: analysis_v2/confirm, Fashion-MNIST: analysis_v2/confirm_fm),
ext5 patches (numbers_check5 F1, F4, W8), and the ext6 results (MDE80, within-configuration, low-dimensional blocks).
Writes paper_v2/tables/*.tex and paper_stats6.txt. CPU only, csv + numpy. Reads results files only; never modifies them."""
from common import *

TAB = f"{BIT}/paper_v2/tables"
CF = f"{A2}/confirm/results"
FM = f"{A2}/confirm_fm/results"
E5 = f"{A2}/ext5"
E6 = f"{A2}/ext6"
CUTS = [10, 30, 100, 300, 1000]
TG = {"acc_15kw": "Test acc.", "gap_15kw": "Gap"}
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


def sz(x, nd=3):
    """signed, no leading zero (compact tables)"""
    s = f"{abs(x):.{nd}f}"
    z = float(s) == 0
    s = s.lstrip("0") if s.startswith("0.") else s
    if z:
        return s
    return ("$-$" if x < 0 else "+") + s


def ci(lo, hi, nd=3, z=False):
    f_ = sz if z else sg
    return f"[{f_(lo, nd)}, {f_(hi, nd)}]"


def bf(s, on):
    return f"\\textbf{{{s}}}" if on else s


def write(name, text):
    with open(f"{TAB}/{name}.tex", "w") as f:
        f.write(text + "\n")
    print("wrote", name)


def pfmt(p):
    return "<.001" if p < 0.001 else f"{p:.3f}" if p < 0.1 else f"{p:.2f}"


def R(path):
    return read_csv(path)


def F(r, k):
    return float(r[k])


PM = {(r["X"], r["target"], int(r["cutoff"])): r for r in R(f"{CF}/primary_ridge.csv")}
PF = {(r["X"], r["target"], int(r["cutoff"])): r for r in R(f"{FM}/primary_ridge.csv")}
GM = {(r["X"], r["target"], int(r["cutoff"])): r for r in R(f"{CF}/secondary_b_gbm.csv")}
GF = {(r["X"], r["target"], int(r["cutoff"])): r for r in R(f"{FM}/secondary_b_gbm.csv")}
cells = [(t, c) for t in TG for c in CUTS]

# ============================================================ side-by-side primary table (main text)
L = [r"\begin{tabular}{llccccccccc}", r"\toprule",
     r" & & \multicolumn{3}{c}{MNIST (28 configurations, 84 runs)} & & \multicolumn{3}{c}{Fashion-MNIST (30 configurations, 90 runs)} \\",
     r"\cmidrule(lr){3-5}\cmidrule(lr){7-9}",
     r"Target & Cutoff & $\Rsq(B_1)$ & Full $G$ & Compact-4 & & $\Rsq(B_1)$ & Full $G$ & Compact-4 \\", r"\midrule"]


def cellf(r):
    lo, hi = F(r, "lo90"), F(r, "hi90")
    mark = ""
    if r["equivalent_holm"] == "1":
        mark = "$^{\\mathrm{E}}$"
    elif lo > 0:
        mark = "$^{\\dagger}$"
    return bf(f"{sz(F(r, 'delta'))} {ci(lo, hi, z=True)}", hi < 0) + mark


for tgt in TG:
    for c in CUTS:
        a, b, d, e = PM[("G", tgt, c)], PM[("compact4", tgt, c)], PF[("G", tgt, c)], PF[("compact4", tgt, c)]
        L.append(f"{TG[tgt] if c == 10 else ''} & {c} & {F(a, 'r2_B1'):.2f} & {cellf(a)} & {cellf(b)} & & {F(d, 'r2_B1'):.2f} & {cellf(d)} & {cellf(e)} \\\\")
    L.append(r"\midrule")
L[-1] = r"\bottomrule"
L.append(r"\end{tabular}")
write("tab_conf_both", "\n".join(L))


def summ(P, X, tag):
    neq = sum(P[(X, t, c)]["equivalent_holm"] == "1" for t, c in cells)
    nun = sum(F(P[(X, t, c)], "tost_p") <= 0.05 for t, c in cells)
    npos = sum(P[(X, t, c)]["positive_holm"] == "1" for t, c in cells)
    npu = sum(F(P[(X, t, c)], "p_pos") <= 0.05 for t, c in cells)
    nlo = sum(F(P[(X, t, c)], "lo90") > 0 for t, c in cells)
    nneg = sum(F(P[(X, t, c)], "delta") < 0 for t, c in cells)
    nbel = sum(F(P[(X, t, c)], "hi90") < 0 for t, c in cells)
    nharm = sum(P[(X, t, c)]["possible_harm"] == "1" for t, c in cells)
    d = [F(P[(X, t, c)], "delta") for t, c in cells]
    minp = min(F(P[(X, t, c)], "pos_holm") for t, c in cells)
    st(f"[{tag} primary] ridge {X}: equiv Holm {neq}/10, equiv unadj {nun}/10, positive Holm {npos}/10, positive unadj(p+<.05) {npu}/10, lo90>0 {nlo}/10, "
       f"point<0 {nneg}/10, hi90<0 {nbel}/10, possible-harm flag {nharm}/10, dR2 range {min(d):.3f} .. {max(d):.3f}, mean {np.mean(d):.4f}, min Holm p+ {minp:.4f}")
    return dict(neq=neq, nun=nun, npos=npos, npu=npu, nlo=nlo, nneg=nneg, nbel=nbel, mean=np.mean(d))


for tag, P in (("MNIST", PM), ("FM", PF)):
    for X in ("G", "compact4"):
        summ(P, X, tag)
    for t in TG:
        v = [F(P[("G", t, c)], "r2_B1") for c in CUTS]
        st(f"[{tag} primary] R2(B1) {t}: {min(v):.3f} .. {max(v):.3f}; per cutoff", [round(x, 3) for x in v])
    for X in ("G", "compact4"):
        for t in TG:
            hw = [(F(P[(X, t, c)], "hi90") - F(P[(X, t, c)], "lo90")) / 2 for c in CUTS]
            st(f"[{tag} primary] 90% half-width {X} {t}: {min(hw):.3f} .. {max(hw):.3f}")
        eqc = [(t, c, F(P[(X, t, c)], 'delta'), F(P[(X, t, c)], 'lo90'), F(P[(X, t, c)], 'hi90'), F(P[(X, t, c)], 'tost_holm')) for t, c in cells if P[(X, t, c)]["equivalent_holm"] == "1"]
        st(f"[{tag} primary] {X} equivalence-confirmed cells:", eqc)
        un = [(t, c, round(F(P[(X, t, c)], 'delta'), 3), round(F(P[(X, t, c)], 'p_pos'), 4), round(F(P[(X, t, c)], 'pos_holm'), 3)) for t, c in cells if F(P[(X, t, c)], "p_pos") <= 0.05]
        st(f"[{tag} primary] {X} unadjusted p+<=.05 cells (t,c,dR2,p+,Holm p+):", un)
# fashion G cells with hi90<0
st("[FM primary] G hi90<0 cells:", [(t, c, round(F(PF[("G", t, c)], 'delta'), 3)) for t, c in cells if F(PF[("G", t, c)], "hi90") < 0])
st("[FM primary] G hi95<0 cells:", [(t, c) for t, c in cells if F(PF[("G", t, c)], "hi95") < 0])
st("[FM primary] G positive point estimates:", [(t, c, round(F(PF[("G", t, c)], 'delta'), 3)) for t, c in cells if F(PF[("G", t, c)], "delta") > 0])
st("[FM primary] c4 point estimates:", [(t, c, round(F(PF[("compact4", t, c)], 'delta'), 3)) for t, c in cells])
st("[FM primary] G delta per cell:", [(t[:3], c, round(F(PF[("G", t, c)], 'delta'), 3)) for t, c in cells])

# ============================================================ Fashion appendix primary table (Holm and GBM)
L = [r"\begin{tabular}{llcccccccc}", r"\toprule",
     r" & & & \multicolumn{2}{c}{Ridge, full $G$ (18)} & \multicolumn{2}{c}{Ridge, compact-4} & \multicolumn{2}{c}{GBM, full $G$} \\",
     r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
     r"Target & Cutoff & $\Rsq(B_1)$ & $\dR$ [90\% CI] & Holm $p$ & $\dR$ [90\% CI] & Holm $p$ & $\dR$ [90\% CI] & $\Rsq(B_1)$ \\", r"\midrule"]
for tgt in TG:
    for c in CUTS:
        g, c4, gb = PF[("G", tgt, c)], PF[("compact4", tgt, c)], GF[("G", tgt, c)]

        def cell(r):
            lo, hi = F(r, "lo90"), F(r, "hi90")
            return bf(f"{sg(F(r, 'delta'))} {ci(lo, hi)}", hi < 0)
        eqc = c4["equivalent_holm"] == "1"
        L.append(f"{TG[tgt] if c == 10 else ''} & {c} & {F(g, 'r2_B1'):.3f} & {cell(g)} & {pfmt(F(g, 'tost_holm'))} & {cell(c4)} & "
                 f"{bf(pfmt(F(c4, 'tost_holm')), eqc)} & {sg(F(gb, 'delta'))} {ci(F(gb, 'lo90'), F(gb, 'hi90'))} & {F(gb, 'r2_B1'):.3f} \\\\")
    L.append(r"\midrule")
L[-1] = r"\bottomrule"
L.append(r"\end{tabular}")
write("tab_fm_primary", "\n".join(L))

# Fashion GBM full table
L = [r"\begin{tabular}{llcccc}", r"\toprule", r"Target & Cutoff & $\Rsq(B_1)$ & Block & $\dR$ [90\% CI] & Class \\", r"\midrule"]
for tgt in TG:
    for c in CUTS:
        for X, XL in (("G", "full $G$"), ("compact4", "compact-4")):
            r = GF[(X, tgt, c)]
            lab_ = r["label"].replace(" (possible harm)", "$^\\downarrow$")
            L.append(f"{TG[tgt] if (c == 10 and X == 'G') else ''} & {c if X == 'G' else ''} & {F(r, 'r2_B1'):.3f} & {XL} & {sg(F(r, 'delta'))} {ci(F(r, 'lo90'), F(r, 'hi90'))} & {lab_} \\\\")
    L.append(r"\midrule")
L[-1] = r"\bottomrule"
L.append(r"\end{tabular}")
write("tab_fm_gbm", "\n".join(L))
for tag, G_ in (("MNIST", GM), ("FM", GF)):
    for X in ("G", "compact4"):
        cnt = {lab: sum(G_[(X, t, c)]["label"].startswith(lab) for t, c in cells) for lab in ("equivalent", "inconclusive", "positive")}
        d = [F(G_[(X, t, c)], "delta") for t, c in cells]
        st(f"[{tag} gbm] {X}: {cnt}, dR2 range {min(d):.3f} .. {max(d):.3f}")
for X in ("G", "compact4"):
    for t, c in cells:
        r = GF[(X, t, c)]
        if r["label"].startswith("positive") or r["label"].startswith("equivalent"):
            st(f"[FM gbm] {X} {t} {c}: {r['label']} dR2 {F(r, 'delta'):.3f} [{F(r, 'lo90'):.3f},{F(r, 'hi90'):.3f}] p+ {F(r, 'p_pos'):.4f}")
for X in ("G", "compact4"):
    for tag, G_, P_ in (("MNIST", GM, PM), ("FM", GF, PF)):
        g, p = G_[(X, "gap_15kw", 1000)], P_[(X, "gap_15kw", 1000)]
        st(f"[{tag} gap1000] {X}: GBM {F(g, 'delta'):+.3f} [{F(g, 'lo90'):+.3f},{F(g, 'hi90'):+.3f}] p+ {F(g, 'p_pos'):.3f}; ridge {F(p, 'delta'):+.3f} [{F(p, 'lo90'):+.3f},{F(p, 'hi90'):+.3f}] p+ {F(p, 'p_pos'):.3f}, R2(B1) ridge {F(p, 'r2_B1'):.3f} gbm {F(g, 'r2_B1'):.3f}")

# ============================================================ Fashion pre-registered sentences
L = []
for X, XL in (("G", "Family F (full $G$)."), ("compact4", "Family C (compact-4).")):
    L += [f"\\textbf{{{XL}}}", r"\begin{itemize}\setlength{\itemsep}{1pt}"]
    for t, c in cells:
        s = PF[(X, t, c)]["sentence"]
        s = s.replace("_", "\\_").replace("±", "$\\pm$").replace("+-", "$\\pm$")
        s = re.sub(r"(?<![\w$])-(\d)", r"$-$\1", s)
        s = s.replace("B1+full G", "$B_1{+}$full $G$").replace("B1+compact4", "$B_1{+}$compact-4").replace("B1 and", "$B_1$ and").replace("90% CI", "90\\% CI")
        s = s.replace("R^2", "$R^2$").replace("%", "\\%").replace("\\\\%", "\\%")
        L.append("\\item " + s)
    L.append(r"\end{itemize}")
write("fm_sentences", "\n".join(L))

# ============================================================ Fashion config table
cfgs = []
for line in open(f"{A2}/confirm_fm/PREREG.txt"):
    m = re.match(r"^(c\d\d)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(mlp|cnn)\s*$", line)
    if m:
        cfgs.append(m.groups())
assert len(cfgs) == 30, len(cfgs)


def e(x):
    if "e" in x:
        a, b = x.split("e")
        return f"${a}{{\\cdot}}10^{{{int(b)}}}$"
    return x


rows = [f"{c} & {n} & {e(w)} & {ln} & {e(lr)} & {a.upper()}" for c, n, w, ln, lr, a in cfgs]
L = [r"\begin{tabular}{llllll@{\hspace{2.2em}}llllll}", r"\toprule", r"ID & $n$ & wd & noise & lr & arch & ID & $n$ & wd & noise & lr & arch \\", r"\midrule"]
for i in range(15):
    L.append(rows[i] + " & " + rows[i + 15] + r" \\")
L += [r"\bottomrule", r"\end{tabular}"]
write("tab_fm_cfg", "\n".join(L))
st("[FM cfg] arch counts:", {a: sum(x[5] == a for x in cfgs) for a in ("mlp", "cnn")}, "n range:",
   min(int(x[1]) if x[1] != "full" else 60000 for x in cfgs), max(int(x[1]) if x[1] != "full" else 60000 for x in cfgs),
   "zero wd:", sum(x[2] == "0" for x in cfgs), "zero noise:", sum(x[3] == "0" for x in cfgs), "full-set configs:", sum(x[1] == "full" for x in cfgs))

# ============================================================ controls: placebo / OPT contrasts
PL = R(f"{FM}/secondary_d_placebo.csv")
OP = R(f"{FM}/secondary_e_opt.csv")
CON = {}
for r in PL + OP:
    CON[(r["contrast"], r["target"], int(r["cutoff"]))] = r


def con(name):
    return [CON[(name, t, c)] for t, c in cells]


def cagg(name):
    rows = con(name)
    d = [F(r, "delta") for r in rows]
    return dict(mean=np.mean(d), neg=sum(x < 0 for x in d), hi90neg=sum(F(r, "hi90") < 0 for r in rows), lo90pos=sum(F(r, "lo90") > 0 for r in rows),
                holm_neg=sum(F(r, "p_neg_holm") < 0.05 for r in rows), holm_pos=sum(F(r, "p_pos_holm") < 0.05 for r in rows), rows=rows)


# MNIST (ext5)
Hh = R(f"{E5}/B_harm_pvalues.csv")
Cc = R(f"{E5}/C_placebo_contrast.csv")
Ee = R(f"{E5}/E_curve_block_placebo.csv")


def magg(rows, dk, pk):
    d = [F(r, dk) for r in rows]
    return np.mean(d), sum(x < 0 for x in d), sum(F(r, pk) < 0.05 for r in rows)


mG = magg([r for r in Hh if r["X"] == "G"], "delta", "p_neg_holm")
m4 = magg([r for r in Hh if r["X"] == "compact4"], "delta", "p_neg_holm")
mrp, mnz = [r for r in Cc if r["placebo"] == "RP18"], [r for r in Cc if r["placebo"] == "NZ18"]
mcv_b = [r for r in Ee if r["contrast"] == "CV18 vs B1"]
mcv_g = [r for r in Ee if r["contrast"] == "G minus CV18"]
M = {"Full $G$ (18)": (mG, None), "Compact-4 (4)": (m4, None),
     "Random proj. (18)": (magg(mrp, "placebo_dR2_vs_B1", "placebo_p_neg_holm"), magg(mrp, "G_minus_placebo", "p_G_worse_holm")),
     "iid noise (18)": (magg(mnz, "placebo_dR2_vs_B1", "placebo_p_neg_holm"), magg(mnz, "G_minus_placebo", "p_G_worse_holm")),
     "Curve statistics CV18 (18)": (magg(mcv_b, "delta", "p_neg_holm"), magg(mcv_g, "delta", "p_neg_holm"))}
# Fashion
fG = [F(PF[("G", t, c)], "delta") for t, c in cells]
f4 = [F(PF[("compact4", t, c)], "delta") for t, c in cells]
FMrows = {"Full $G$ (18)": ((np.mean(fG), sum(x < 0 for x in fG), None), None),
          "Compact-4 (4)": ((np.mean(f4), sum(x < 0 for x in f4), None), None)}
for lab, vs, gm in (("Random proj. (18)", "RP18 vs B1", "G minus RP18"), ("iid noise (18)", "NZ18 vs B1", "G minus NZ18"),
                    ("Curve statistics CV18 (18)", "CV18 vs B1", "G minus CV18"),
                    ("Optimizer stats OPT18 (18)", "OPT18 vs B1", "G minus OPT18"), ("Optimizer stats OPT4 (4)", "OPT4 vs B1", "G minus OPT4")):
    a, b = cagg(vs), cagg(gm)
    FMrows[lab] = ((a["mean"], a["neg"], a["holm_neg"]), (b["mean"], sum(F(r, "delta") < 0 for r in b["rows"]), b["holm_neg"]))
    st(f"[FM ctrl] {vs}: mean {a['mean']:.4f}, <0 {a['neg']}/10, hi90<0 {a['hi90neg']}/10, Holm harm p<.05 {a['holm_neg']}/10; "
       f"{gm}: mean {b['mean']:.4f}, G worse {sum(F(r, 'delta') < 0 for r in b['rows'])}/10, G better (delta>0) {sum(F(r, 'delta') > 0 for r in b['rows'])}/10, "
       f"Holm p+ <.05 (G better) {b['holm_pos']}/10, Holm p- <.05 (G worse) {b['holm_neg']}/10, lo90>0 {b['lo90pos']}/10")
for lab, gm in (("c4 - RP18", "compact4 minus RP18"), ("c4 - CV18", "compact4 minus CV18"), ("c4 - OPT18", "compact4 minus OPT18"), ("c4 - OPT4", "compact4 minus OPT4"),
                ("G - OPT4", "G minus OPT4")):
    b = cagg(gm)
    cellsb = [(t[:3], c) for (t, c), r in zip(cells, b["rows"]) if F(r, "p_pos_holm") < 0.05]
    st(f"[FM ctrl] {gm}: mean {b['mean']:.4f}, better (delta>0) {sum(F(r, 'delta') > 0 for r in b['rows'])}/10, Holm p+<.05 {b['holm_pos']}/10 cells {cellsb}, Holm p-<.05 {b['holm_neg']}/10")
for gm in ("G minus RP18", "G minus NZ18", "G minus CV18", "G minus OPT18"):
    b = cagg(gm)
    cellsb = [(t[:3], c) for (t, c), r in zip(cells, b["rows"]) if F(r, "p_pos_holm") < 0.05]
    st(f"[FM ctrl] {gm}: Holm p+<.05 cells {cellsb}")
# also numbers for text
a = cagg("OPT18 vs B1")
st("[FM ctrl] OPT18 vs B1 per cell:", [(t[:3], c, round(F(r, 'delta'), 3), r["p_neg_holm"][:5]) for (t, c), r in zip(cells, a["rows"])])
a = cagg("OPT4 vs B1")
st("[FM ctrl] OPT4 vs B1 per cell:", [(t[:3], c, round(F(r, 'delta'), 3), round(F(r, 'p_neg_holm'), 3)) for (t, c), r in zip(cells, a["rows"])])
a = cagg("RP18 vs B1")
st("[FM ctrl] RP18 vs B1 per cell:", [(t[:3], c, round(F(r, 'delta'), 3), round(F(r, 'p_neg_holm'), 3)) for (t, c), r in zip(cells, a["rows"])])
a = cagg("G minus OPT18")
st("[FM ctrl] G minus OPT18 per cell:", [(t[:3], c, round(F(r, 'delta'), 3), round(F(r, 'p_pos_holm'), 3)) for (t, c), r in zip(cells, a["rows"])])
a = cagg("compact4 minus RP18")
st("[FM ctrl] c4 minus RP18 per cell:", [(t[:3], c, round(F(r, 'delta'), 3), round(F(r, 'p_pos_holm'), 3)) for (t, c), r in zip(cells, a["rows"])])

# main-text control table
L = [r"\begin{tabular}{lcccccc@{\hspace{1.4em}}cccccc}", r"\toprule",
     r" & \multicolumn{5}{c}{MNIST (28 configurations)} & & \multicolumn{5}{c}{Fashion-MNIST (30 configurations)} \\",
     r"\cmidrule(lr){2-6}\cmidrule(lr){8-12}",
     r" & \multicolumn{2}{c}{Block vs.\ $B_1$} & \multicolumn{3}{c}{$G$ minus block} & & \multicolumn{2}{c}{Block vs.\ $B_1$} & \multicolumn{3}{c}{$G$ minus block} \\",
     r"\cmidrule(lr){2-3}\cmidrule(lr){4-6}\cmidrule(lr){8-9}\cmidrule(lr){10-12}",
     r"Block (columns) & Mean $\dR$ & $<0$ & Mean & $G$ worse & Holm & & Mean $\dR$ & $<0$ & Mean & $G$ worse & Holm \\", r"\midrule"]
order = ["Full $G$ (18)", "Compact-4 (4)", "Random proj. (18)", "iid noise (18)", "Curve statistics CV18 (18)", "Optimizer stats OPT18 (18)", "Optimizer stats OPT4 (4)"]


def side(v, hasholm=True):
    (m, n, h), con_ = v
    if con_ is None:
        return f"{sz(m)} & {n}/10 & --- & --- & ---"
    cm, cn, ch = con_
    return f"{sz(m)} & {n}/10 & {sz(cm)} & {cn}/10 & {ch}/10"


for lab in order:
    ms = side(M[lab]) if lab in M else "n/a & n/a & n/a & n/a & n/a"
    L.append(f"{lab} & {ms} & & {side(FMrows[lab])} \\\\")
L += [r"\bottomrule", r"\end{tabular}"]
write("tab_conf_ctrl6", "\n".join(L))
for k, v in M.items():
    st("[MNIST ctrl]", k, v)

# ============================================================ Fashion detailed control table (appendix)
def tabrow(name, lab):
    rows = con(name)
    cells_ = []
    for r in rows:
        cells_.append(r)
    return rows


L = [r"\begin{tabular}{llccccc}", r"\toprule",
     r"Contrast & Quantity & Mean & Cells $<0$ & 90\% CI wholly $<0$ & 90\% CI wholly $>0$ & Holm $p<.05$ (harm / better) \\", r"\midrule"]
NAMES = [("RP18 vs B1", "$B_1\\to B_1{+}$RP18"), ("NZ18 vs B1", "$B_1\\to B_1{+}$noise"), ("CV18 vs B1", "$B_1\\to B_1{+}$CV18"),
         ("OPT18 vs B1", "$B_1\\to B_1{+}$OPT18"), ("OPT4 vs B1", "$B_1\\to B_1{+}$OPT4"),
         ("G minus RP18", "$G$ minus RP18"), ("G minus NZ18", "$G$ minus noise"), ("G minus CV18", "$G$ minus CV18"),
         ("G minus OPT18", "$G$ minus OPT18"), ("G minus OPT4", "$G$ minus OPT4"),
         ("compact4 minus RP18", "compact-4 minus RP18"), ("compact4 minus CV18", "compact-4 minus CV18"),
         ("compact4 minus OPT18", "compact-4 minus OPT18"), ("compact4 minus OPT4", "compact-4 minus OPT4")]
for i, (nm, lab) in enumerate(NAMES):
    a = cagg(nm)
    if i in (5, 10):
        L.append(r"\midrule")
    if "vs B1" in nm:
        hp = f"{a['holm_neg']} / ---"
    else:
        hp = f"{a['holm_neg']} / {a['holm_pos']}"
    qty = "mean $\\dR$" if "vs B1" in nm else "mean difference"
    L.append(f"{lab} & {qty} & {sg(a['mean'])} & {a['neg']}/10 & {a['hi90neg']}/10 & {a['lo90pos']}/10 & {hp} \\\\")
L += [r"\bottomrule", r"\end{tabular}"]
write("tab_fm_ctrl", "\n".join(L))

# ============================================================ low-dim (Fashion f) and within-config (Fashion g, MNIST ext6 B)
LD = R(f"{FM}/secondary_f_lowdim.csv")
blocks = []
for b in dict.fromkeys(r["block"] for r in LD):
    rows = [r for r in LD if r["block"] == b]
    blocks.append((b, rows))
L = [r"\begin{tabular}{lcccccc}", r"\toprule",
     r"Block (1 column) & Mean $\dR$ & Cells $<0$ & 90\% CI $>0$ & Holm $p_+<.05$ & Min Holm $p_+$ & Largest cell \\", r"\midrule"]
for b, rows in blocks:
    d = [F(r, "delta") for r in rows]
    big = max(rows, key=lambda r: F(r, "delta"))
    L.append(f"{b.replace('_', chr(92) + '_')} & {sg(np.mean(d))} & {sum(x < 0 for x in d)}/10 & {sum(F(r, 'lo90') > 0 for r in rows)}/10 & {sum(F(r, 'p_pos_holm') < 0.05 for r in rows)}/10 & "
             f"{F(big, 'p_pos_holm') if False else min(F(r, 'p_pos_holm') for r in rows):.3f} & {('Acc.' if big['target'].startswith('acc') else 'Gap')} {big['cutoff']}: {sg(F(big, 'delta'))} {ci(F(big, 'lo90'), F(big, 'hi90'))} \\\\")
    st(f"[FM lowdim] {b}: mean {np.mean(d):.4f}, <0 {sum(x < 0 for x in d)}, lo90>0 {sum(F(r, 'lo90') > 0 for r in rows)}, Holm<.05 {sum(F(r, 'p_pos_holm') < 0.05 for r in rows)}, "
       f"min Holm p+ {min(F(r, 'p_pos_holm') for r in rows):.3f}, max cell {big['target']} {big['cutoff']} {F(big, 'delta'):.3f} [{F(big, 'lo90'):.3f},{F(big, 'hi90'):.3f}] p+ {F(big, 'p_pos'):.4f}")
    for r in rows:
        if F(r, "lo90") > 0:
            st(f"   [FM lowdim] {b} {r['target']} {r['cutoff']}: {F(r, 'delta'):.3f} [{F(r, 'lo90'):.3f},{F(r, 'hi90'):.3f}] p+ {F(r, 'p_pos'):.4f} Holm {F(r, 'p_pos_holm'):.3f}")
# Fashion ridge R2(B1) at gap 1000
st("[FM lowdim] total cells with lo90>0:", sum(F(r, "lo90") > 0 for r in LD), "of", len(LD), "; Holm p+<.05:", sum(F(r, "p_pos_holm") < 0.05 for r in LD),
   "min Holm", min(F(r, "p_pos_holm") for r in LD))
mld = R(f"{E6}/C_lowdim_cells.csv")
st("[MNIST lowdim ext6 C] columns", list(mld[0].keys())[:12], "n rows", len(mld))
write("tab_fm_lowdim", "\n".join(L + [r"\bottomrule", r"\end{tabular}"]))

# within-config
WF = R(f"{FM}/secondary_g_within.csv")
WM = R(f"{E6}/B_within_config_confirmatory.csv")
L = [r"\begin{tabular}{llcccccc}", r"\toprule",
     r"Data & Cutoff & Target & $B_0^c$ RMSE & $+G$ & $+$compact-4 & $+$OPT18 & $+$noise \\", r"\midrule"]
for tgt in TG:
    for c in CUTS:
        def pick(rows, model_prefix):
            rr = [r for r in rows if r["target"] == tgt and int(r["cutoff"]) == c and r["model"].startswith(model_prefix)]
            return rr[0] if rr else None
        for dat, rows in (("Fashion-MNIST", WF), ("MNIST", WM)):
            g, c4, o, nz = pick(rows, "B0c+G"), pick(rows, "B0c+c4"), pick(rows, "B0c+OPT18"), pick(rows, "B0c+NZ18")

            def cc(r):
                if r is None:
                    return "---"
                lo, hi = F(r, "dR2c_lo95"), F(r, "dR2c_hi95")
                return bf(f"{sz(F(r, 'dR2c'))} {ci(lo, hi, 2, z=True)}", lo > 0)
            b0 = F(g, "B0c_rmse_pts") if "B0c_rmse_pts" in g else float("nan")
            L.append(f"{dat} & {c} & {'Acc.' if tgt.startswith('acc') else 'Gap'} & {b0:.2f} & {cc(g)} & {cc(c4)} & {cc(o)} & {cc(nz)} \\\\")
    L.append(r"\midrule")
L[-1] = r"\bottomrule"
L.append(r"\end{tabular}")
write("tab_within6", "\n".join(L))
for nm, rows in (("FM", WF), ("MNIST", WM)):
    for mdl in ("B0c+G", "B0c+c4", "B0c+OPT18", "B0c+NZ18"):
        rr = [r for r in rows if r["model"].startswith(mdl)]
        if not rr:
            continue
        st(f"[{nm} within] {mdl}: dR2c range {min(F(r, 'dR2c') for r in rr):.3f} .. {max(F(r, 'dR2c') for r in rr):.3f}, lo95>0 cells "
           f"{[(r['target'][:3], r['cutoff'], round(F(r, 'dR2c'), 3), round(F(r, 'dR2c_lo95'), 3), round(F(r, 'dR2c_hi95'), 3), round(F(r, 'p_pos_holm'), 3)) for r in rr if F(r, 'dR2c_lo95') > 0]}, "
           f"min Holm p+ {min(F(r, 'p_pos_holm') for r in rr):.3f}")
    g = [r for r in rows if r["model"].startswith("B0c+G") and r["target"] == "acc_15kw"]
    for r in g:
        k = "within_sd_pts" if "within_sd_pts" in r else "within_sd_ddof1_pts"
        st(f"[{nm} within] acc cutoff {r['cutoff']} within-SD {F(r, k):.2f} null RMSE {F(r, 'null_rmse_pts') if 'null_rmse_pts' in r else float('nan'):.2f} B0c RMSE {F(r, 'B0c_rmse_pts'):.2f} "
           f"G dR2c {F(r, 'dR2c'):+.3f} [{F(r, 'dR2c_lo95'):+.3f},{F(r, 'dR2c_hi95'):+.3f}] RMSE gain {F(r, 'rmse_gain_pts'):+.3f} Holm p+ {F(r, 'p_pos_holm'):.3f}")

# ============================================================ transfer (Fashion a)
T = R(f"{FM}/secondary_a_transfer.csv")
L = [r"\begin{tabular}{lllcccc}", r"\toprule", r"Learner & Test set & Block & Equivalent & Positive & Inconclusive & Inconclusive, $\dR<0$ \\", r"\midrule"]
for mdl in ("ridge", "gbm"):
    for sub, sl in (("all", "90 runs"), ("mlp_only", "45 MLP runs")):
        for X, XL in (("G", "full $G$"), ("compact4", "compact-4")):
            rr = [x for x in T if x["model"] == mdl and x["subset"] == sub and x["X"] == X]
            ne = sum(x["label"].startswith("equivalent") for x in rr)
            npz = sum(x["label"].startswith("positive") for x in rr)
            ni = sum(x["label"].startswith("inconclusive") for x in rr)
            nin = sum(x["label"].startswith("inconclusive") and F(x, "delta") < 0 for x in rr)
            L.append(f"{'GBM' if mdl == 'gbm' else 'Ridge'} & {sl} & {XL} & {ne} & {npz} & {ni} & {nin} \\\\")
            st(f"[FM transfer] {mdl} {sub} {X}: E {ne} P {npz} I {ni} (I with dR2<0: {nin}), n={len(rr)}")
    if mdl == "ridge":
        L.append(r"\midrule")
L += [r"\bottomrule", r"\end{tabular}"]
write("tab_fm_transfer", "\n".join(L))
st("[FM transfer] total positive cells:", sum(x["label"].startswith("positive") for x in T), "of", len(T))
for x in T:
    if x["label"].startswith("positive"):
        st(f"   [FM transfer] positive: {x['model']} {x['subset']} {x['X']} {x['target']} {x['cutoff']} dR2 {F(x, 'delta'):+.3f} [{F(x, 'lo90'):+.3f},{F(x, 'hi90'):+.3f}] R2(B1) {F(x, 'r2_B1'):.2f}")

# ============================================================ per-architecture (Fashion, exploratory)
A = R(f"{FM}/exploratory_per_arch.csv")
L = [r"\begin{tabular}{llccccccc}", r"\toprule", r" & & \multicolumn{3}{c}{MLP (45 runs)} & \multicolumn{3}{c}{CNN (45 runs)} \\", r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
     r"Target & Cutoff & $B_1$ & $B_1{+}G$ & $B_1{+}c_4$ & $B_1$ & $B_1{+}G$ & $B_1{+}c_4$ \\", r"\midrule"]
nh = {"mlp": 0, "cnn": 0}
nh4 = {"mlp": 0, "cnn": 0}
for tgt in TG:
    for c in CUTS:
        m = [x for x in A if x["target"] == tgt and int(x["cutoff"]) == c and x["arch"] == "mlp"][0]
        n = [x for x in A if x["target"] == tgt and int(x["cutoff"]) == c and x["arch"] == "cnn"][0]
        for k, r in (("mlp", m), ("cnn", n)):
            nh[k] += F(r, "r2_B1G") < F(r, "r2_B1")
            nh4[k] += F(r, "r2_B1c4") < F(r, "r2_B1")
        L.append(f"{TG[tgt] if c == 10 else ''} & {c} & " + " & ".join(f"{F(r, k):.3f}" for r in (m, n) for k in ("r2_B1", "r2_B1G", "r2_B1c4")) + r" \\")
    L.append(r"\midrule")
L[-1] = r"\bottomrule"
L.append(r"\end{tabular}")
write("tab_fm_arch", "\n".join(L))
st(f"[FM arch] cells (of 10) with pooled R2(B1+G) < R2(B1): {nh}; R2(B1+c4) < R2(B1): {nh4}")
for k in ("mlp", "cnn"):
    rr = [x for x in A if x["arch"] == k]
    st(f"[FM arch] {k} R2(B1+G)-R2(B1) range:", round(min(F(x, 'r2_B1G') - F(x, 'r2_B1') for x in rr), 3), round(max(F(x, 'r2_B1G') - F(x, 'r2_B1') for x in rr), 3),
       " c4:", round(min(F(x, 'r2_B1c4') - F(x, 'r2_B1') for x in rr), 3), round(max(F(x, 'r2_B1c4') - F(x, 'r2_B1') for x in rr), 3),
       "G>B1 cells:", [(x['target'][:3], x['cutoff']) for x in rr if F(x, 'r2_B1G') > F(x, 'r2_B1')], "c4>B1 cells:", [(x['target'][:3], x['cutoff']) for x in rr if F(x, 'r2_B1c4') > F(x, 'r2_B1')])

# ============================================================ ext6: power (MNIST)
mde = R(f"{E6}/A_power_mde.csv")
txt = open(f"{E6}/A_power_summary.txt").read().split("\n")
anycell = {}
cur = None
for ln in txt:
    m = re.match(r"=== signal (\S+), test block (\S+) ===", ln)
    if m:
        cur = (m.group(1), m.group(2))
    if ln.startswith("level:") and cur:
        lv = [float(x) for x in ln.split()[1:]]
    if ln.startswith("any-cell Holm rejection:") and cur:
        anycell[cur] = dict(zip(lv, [float(x) for x in ln.split(":")[1].split()]))
L = [r"\begin{tabular}{llccccc}", r"\toprule",
     r"Signal & Block & MDE80 (Holm), median [min, max] & Not reached & \multicolumn{3}{c}{Any-cell Holm detection at true $+$} \\",
     r"\cmidrule(lr){5-7}", r" & & & & 0.02 & 0.03 & 0.05 \\", r"\midrule"]
for sig, sl in (("te_fisher", "Fisher ratio"), ("te_eff_rank_pr", "effective rank")):
    for var, vl in (("full", "full $G$ (18)"), ("compact4", "compact-4"), ("1dim", "1 column")):
        rr = [r for r in mde if r["feat"] == sig and r["variant"] == var]
        v = [F(r, "mde80_holm") for r in rr]
        nr = sum(not np.isfinite(x) for x in v)
        vv = [x for x in v if np.isfinite(x)]
        ac = anycell[(sig, {"full": "full", "compact4": "compact4", "1dim": "1dim"}[var])]
        L.append(f"{sl if var == 'full' else ''} & {vl} & {np.median(vv):.3f} [{min(vv):.3f}, {max(vv):.3f}] & {nr}/10 & {ac[0.02]:.2f} & {ac[0.03]:.2f} & {ac[0.05]:.2f} \\\\")
        unad = [F(r, "mde80_unadj") for r in rr]
        st(f"[MNIST power] {sig} {var}: MDE80 Holm median {np.median(vv):.3f} [{min(vv):.3f},{max(vv):.3f}] not reached {nr}/10; unadj median "
           f"{np.nanmedian(unad):.3f}; any-cell Holm at +.02/.03/.05/.10: {ac[0.02]:.2f}/{ac[0.03]:.2f}/{ac[0.05]:.2f}/{ac[0.1]:.2f}; size at 0 (any-cell Holm) {ac[0.0]:.2f}")
    if sig == "te_fisher":
        L.append(r"\midrule")
L += [r"\bottomrule", r"\end{tabular}"]
write("tab_power6", "\n".join(L))

# ============================================================ ext5 patches (numbers_check5 F1, F4, W8)
S = R(f"{E5}/A_subsample_summary.csv")
sub28 = [r for r in S if r["K"] == "28" and r["block"] == "G"]
cellsub = {}
for tgt in TG:
    rr = [(F(r, "p5"), F(r, "p95"), r["cutoff"]) for r in sub28 if r["target"] == tgt]
    nar = min(rr, key=lambda x: x[1] - x[0])
    wid = max(rr, key=lambda x: x[1] - x[0])
    cellsub[tgt] = (nar, wid)
    st(f"[sub K=28 G {tgt}] narrowest cell cutoff {nar[2]} [{nar[0]:.3f},{nar[1]:.3f}], widest cutoff {wid[2]} [{wid[0]:.3f},{wid[1]:.3f}]; p5 range {min(x[0] for x in rr):.3f}..{max(x[0] for x in rr):.3f}; p95 range {min(x[1] for x in rr):.3f}..{max(x[1] for x in rr):.3f}")
x = open(f"{TAB}/tab_ext5.tex").read()
if "narrowest to widest cell" in x:  # idempotent patch
    old_start = x.index("5--95\\% range of the draws, narrowest to widest cell,")
    old_end = x.index("Compact-4 $-$0.001")
    (na, wa), (ng, wg) = cellsub["acc_15kw"], cellsub["gap_15kw"]
    new = (f"5--95\\% range of the draws {ci(na[0], na[1], 3)} in the narrowest accuracy cell (cutoff {na[2]}) and {ci(wa[0], wa[1], 3)} in the widest (cutoff {wa[2]}); "
           f"for the gap {ci(ng[0], ng[1], 3)} (cutoff {ng[2]}) and {ci(wg[0], wg[1], 3)} (cutoff {wg[2]}). ")
    x = x[:old_start] + new + x[old_end:]
    assert "A 90\\% lower bound above zero in 1 of 80 cells ($+0.003$)." in x
    x = x.replace("A 90\\% lower bound above zero in 1 of 80 cells ($+0.003$).",
                  "A 90\\% lower bound above zero in 1 of 80 cells (increment $+0.003$, lower bound $+0.001$).")
    open(f"{TAB}/tab_ext5.tex", "w").write(x)
print("patched tab_ext5")
h = open(f"{TAB}/tab_hardcase.tex").read()
if "+0.23 [+0.00, +0.65]" in h:
    h = h.replace("+0.23 [+0.00, +0.65]", "+0.23 [$-$0.00, +0.65]")
    open(f"{TAB}/tab_hardcase.tex", "w").write(h)
    print("patched tab_hardcase")
# sanity: low-dim lower bound
lowd = R(f"{E5}/A_conf_ii_lowdim.csv")
pos = [r for r in lowd if F(r, "lo90") > 0]
st("[ext5 lowdim] cells lo90>0:", [(r["block"], r["target"], r["cutoff"], round(F(r, "delta"), 4), round(F(r, "lo90"), 4)) for r in pos])


# ============================================================ MNIST low-dimensional table (ext6 C)
CL = R(f"{E6}/C_lowdim_table.csv")
L = [r"\begin{tabular}{lcccccc}", r"\toprule", r"Block & Columns & Mean $\dR$ & Cells 90\% CI $<0$ & Cells 90\% CI $>0$ & Min Holm $p_+$ & Largest cell \\", r"\midrule"]
for r in CL:
    nm = r["X"].replace("_", "\\_")
    L.append(f"{nm} & {r['n_dims']} & {sg(F(r, 'mean_dR2'))} & {r['n_cells_hi90_below0']}/10 & {r['n_cells_lo90_above0']}/10 & {F(r, 'min_p_pos_holm'):.2f} & "
             f"{r['max_cell'].replace('acc_15kw', 'acc.').replace('gap_15kw', 'gap')}: {sg(F(r, 'max_dR2'))} {ci(F(r, 'max_lo90'), F(r, 'max_hi90'))} \\\\")
L += [r"\bottomrule", r"\end{tabular}"]
write("tab_mnist_lowdim", "\n".join(L))

# ============================================================ MNIST confirmatory: percentiles, other paper numbers retained
open("/homes/lbastian/latent-dynamics/paper_v2_figs/paper_stats6.txt", "w").write("\n".join(STATS) + "\n")
print("stats written")
