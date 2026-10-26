"""Generate LaTeX table fragments for paper_v2 directly from the analysis CSV/summary files."""
from common import *
from collections import OrderedDict

TAB = f"{BIT}/paper_v2/tables"
os.makedirs(TAB, exist_ok=True)


def sg(x, nd=3, plus=True):
    if x is None or not np.isfinite(x):
        return "--"
    s = f"{abs(x):.{nd}f}"
    if x < 0 and float(s) != 0:
        return "$-$" + s
    return ("+" if plus else "") + s


def ci(d, lo, hi, nd=3):
    return f"{sg(d, nd)} [{sg(lo, nd)}, {sg(hi, nd)}]"


def write(name, txt):
    with open(f"{TAB}/{name}.tex", "w") as f:
        f.write(txt)
    print("wrote", name)


# ------------------------------------------------------------------ prediction tables (ext, 305 runs)
S3 = read_csv(f"{EXT}/step3_results.csv")
IDX = {}
for r in S3:
    IDX[(r["tag"], r["target"], int(r["cutoff"]), r["model"], r["kind"], r["set"], r["base"])] = r


def g(tag, tgt, cut, model, kind, s, base=""):
    return IDX[(tag, tgt, cut, model, kind, s, base)]


CUTS = [10, 30, 100, 300, 1000]
TN = {"acc_15kw": "test acc., 12k--18k median", "gap_15kw": "gap, 12k--18k median",
      "acc_15k": "test acc., step 15k", "gap_15k": "gap, step 15k",
      "acc_finw": "test acc., final window", "gap_finw": "gap, final window",
      "acc_fin": "test acc., final step", "gap_fin": "gap, final step"}

# main compact table
lines = [r"\begin{tabular}{llrrrll}", r"\toprule",
         r"Target & Cutoff & $R^2(B_1)$ & $R^2(B_1{+}G)$ & $R^2(G)$ & $\Delta R^2$ ridge [95\% CI] & $\Delta R^2$ GBM [95\% CI] \\", r"\midrule"]
for tgt in ["acc_15kw", "gap_15kw"]:
    for c in [10, 100, 1000]:
        b1 = fnum(g("main", tgt, c, "ridge", "set", "B1")["r2"])
        b1g = fnum(g("main", tgt, c, "ridge", "set", "B1+G")["r2"])
        ga = fnum(g("main", tgt, c, "ridge", "set", "G")["r2"])
        ir = g("main", tgt, c, "ridge", "increment", "B1+G", "B1")
        ig = g("main", tgt, c, "gbm", "increment", "B1+G", "B1")
        lines.append(f"{'test accuracy' if tgt.startswith('acc') else 'gap'} & {c} & {b1:.2f} & {b1g:.2f} & {ga:.2f} & "
                     f"{ci(fnum(ir['delta']), fnum(ir['ci_lo']), fnum(ir['ci_hi']))} & {ci(fnum(ig['delta']), fnum(ig['ci_lo']), fnum(ig['ci_hi']))} \\\\")
    if tgt == "acc_15kw":
        lines.append(r"\midrule")
lines += [r"\bottomrule", r"\end{tabular}"]
write("tab_predict_main", "\n".join(lines))

# full appendix tables (body rows only; the longtable header lives in paper.tex), one per model
for model in ["ridge", "gbm"]:
    L = []
    for tgt in ["acc_15kw", "gap_15kw", "acc_15k", "gap_15k", "acc_finw", "gap_finw", "acc_fin", "gap_fin"]:
        for c in CUTS:
            R = lambda s: fnum(g("main", tgt, c, model, "set", s)["r2"])
            i0 = g("main", tgt, c, model, "increment", "B0+G", "B0")
            i1 = g("main", tgt, c, model, "increment", "B1+G", "B1")
            L.append(f"{TN[tgt] if c == 10 else ''} & {c} & {R('B0'):.2f} & {R('B1'):.2f} & {R('G'):.2f} & {R('B1+G'):.2f} & "
                     f"{ci(fnum(i0['delta']), fnum(i0['ci_lo']), fnum(i0['ci_hi']), 2)} & {ci(fnum(i1['delta']), fnum(i1['ci_lo']), fnum(i1['ci_hi']))} \\\\")
        L.append(r"\midrule")
    L[-1] = r"\bottomrule"
    write(f"tab_predict_full_{model}_body", "\n".join(L))

# secondary targets / sensitivity
L = [r"\begin{tabular}{llllll}", r"\toprule", r"Subset / target & Model & Cutoff 10 & Cutoff 100 & Cutoff 1000 & $n$ runs \\", r"\midrule"]
SENS = [("pre_covered", "acc_pre", "pre-slingshot test acc., any event"), ("pre_covered", "gap_pre", "pre-slingshot gap, any event"),
        ("pre_covered", "acc_preN", "pre-slingshot test acc., type N"), ("pre_covered", "gap_preN", "pre-slingshot gap, type N"),
        ("sens_seeds012", "acc_15kw", "seeds 0--2 only, test acc., 12k--18k median"), ("sens_seeds012", "gap_15kw", "seeds 0--2 only, gap, 12k--18k median"),
        ("sens_excl_slingshot_le15k", "acc_15kw", "excl.\\ onset $\\leq$15k, test acc., 12k--18k median"), ("sens_excl_slingshot_le15k", "gap_15kw", "excl.\\ onset $\\leq$15k, gap, 12k--18k median"),
        ("final40k", "acc_finw", "40k-step runs, final-window acc."), ("final40k", "gap_finw", "40k-step runs, final-window gap")]
for tag, tgt, lab in SENS:
    for model in ["ridge", "gbm"]:
        cells = []
        nr = ""
        ok = True
        for c in [10, 100, 1000]:
            k = (tag, tgt, c, model, "increment", "B1+G", "B1")
            if k not in IDX:
                ok = False
                break
            r = IDX[k]
            cells.append(ci(fnum(r["delta"]), fnum(r["ci_lo"]), fnum(r["ci_hi"]), 2))
            nr = IDX[(tag, tgt, c, model, "set", "B1", "")]["n_runs"]
        if ok:
            L.append(f"{lab if model == 'ridge' else ''} & {model} & " + " & ".join(cells) + f" & {nr} \\\\")
    L.append(r"\midrule")
L[-1] = r"\bottomrule"
L.append(r"\end{tabular}")
write("tab_predict_sens", "\n".join(L))

# ------------------------------------------------------------------ audit table (original 105 runs)
au2 = open(f"{A2}/ext/audit/audit2.txt").read().splitlines()
au3 = open(f"{A2}/ext/audit/audit3.txt").read().splitlines()
pat = re.compile(r"(\w+) cut=(\d+) (.*?)\s+R2 base ([+-][\d.]+) comp ([+-][\d.]+)\s+dR2 ([+-][\d.]+) \[([+-][\d.]+),([+-][\d.]+)\]")
pat_rep = re.compile(r"(\w+) cut=(\d+): now dR2 ([+-][\d.]+) \[([+-][\d.]+),([+-][\d.]+)\]")
A = {}
for ln in au2:
    m = pat.search(ln)
    if m and ("[B-compact]" in ln or "[D-deltatarget]" in ln or "[C-alpha]" in ln or "[C-gbm]" in ln or "[E-pos-resid]" in ln or "[E-neg]" in ln):
        tag = ln.strip().split("]")[0] + "]"
        A[(tag, m.group(1), int(m.group(2)), m.group(3).strip())] = tuple(float(m.group(i)) for i in (6, 7, 8))
    m = pat_rep.search(ln)
    if m and "now dR2" in ln:
        A[("[A]", m.group(1), int(m.group(2)), "reproduction")] = tuple(float(m.group(i)) for i in (3, 4, 5))
A3 = {}
pat3 = re.compile(r"(\w+) cut=(\d+) noise ([\d.]+)sd copy alone: R2 [+-][\d.]+->[+-][\d.]+ dR2 ([+-][\d.]+) \[([+-][\d.]+),([+-][\d.]+)\]")
for ln in au3:
    m = pat3.search(ln)
    if m:
        A3[(m.group(1), int(m.group(2)), m.group(3))] = tuple(float(m.group(i)) for i in (4, 5, 6))
# permuted G / shuffled lines
perm = {}
for ln in au2:
    m = re.search(r"\[E-neg\] (\w+) cut=(\d+) G permuted across configs \(5 draws\): dR2 mean ([+-][\d.]+) range \[([+-][\d.]+),([+-][\d.]+)\]", ln)
    if m:
        perm[(m.group(1), int(m.group(2)))] = (float(m.group(3)), float(m.group(4)), float(m.group(5)))
shuf = {}
for ln in au2:
    m = re.search(r"\[E-neg\] (\w+) cut=(\d+) target shuffled across configs: R2\(B1\)=([+-][\d.]+) R2\(B1\+G\)=([+-][\d.]+)", ln)
    if m:
        shuf[(m.group(1), int(m.group(2)))] = (float(m.group(3)), float(m.group(4)))


def cell(key):
    v = A.get(key)
    return ci(*v, nd=2) if v else "--"


def row4(lab, mk):
    return lab + " & " + " & ".join(cell(mk(t, c)) for t, c in [("acc_15k", 100), ("acc_15k", 1000), ("gap_15k", 100), ("gap_15k", 1000)]) + r" \\"


L = [r"\begin{tabular}{lcccc}", r"\toprule", r" & \multicolumn{2}{c}{Test accuracy} & \multicolumn{2}{c}{Gap} \\",
     r"Check (ridge unless noted; original 105 runs) & cutoff 100 & cutoff 1000 & cutoff 100 & cutoff 1000 \\", r"\midrule",
     r"\multicolumn{5}{l}{\emph{Is the null an artefact of the pipeline?}} \\"]
L.append(row4("Reproduction of the primary fit (all 18 geometry features)", lambda t, c: ("[A]", t, c, "reproduction")))
L.append(row4("Compact $G$: Fisher, rank, Mag$(t{=}1)$, $H_1$ top-5, test and difference (8 features)",
              lambda t, c: ("[B-compact]", t, c, "compact: te+delta of F,rank,Mag1,H1top5 (8)")))
L.append(row4("Fisher only (test and difference)", lambda t, c: ("[B-compact]", t, c, "Fisher te+delta (2)")))
L.append(row4("$H_1$ top-5 only (test and difference)", lambda t, c: ("[B-compact]", t, c, "H1 top-5 te+delta (2)")))
L.append(row4("Wider ridge grid ($\\alpha\\in[10^{-6},10^{6}]$, 25 values)", lambda t, c: ("[C-alpha]", t, c, "alpha 1e-6..1e6 (25)")))
L.append(row4("Delta target ($y$ minus value at cutoff), full $G$", lambda t, c: ("[D-deltatarget]", t, c, "y - current value; full G")))
L.append(row4("Gradient boosting, depth 1", lambda t, c: ("[C-gbm]", t, c, "GBM d1 300t lr.05")))
L.append(row4("Gradient boosting, depth 3", lambda t, c: ("[C-gbm]", t, c, "GBM d3 300t lr.03")))
L.append(row4("Random forest, 200 trees", lambda t, c: ("[C-gbm]", t, c, "RF 200 trees mindleaf3")))
L.append(r"\midrule \multicolumn{5}{l}{\emph{Negative controls (expected: no gain)}} \\")
L.append("18 Gaussian noise columns in place of $G$ & " + " & ".join(cell(("[E-neg]", t, c, "18 gaussian noise cols")) for t, c in [("acc_15k", 100), ("acc_15k", 1000), ("gap_15k", 100), ("gap_15k", 1000)]) + r" \\")
L.append("$G$ permuted across configs (mean of 5 draws) & " + " & ".join(sg(perm[(t, c)][0], 2) for t, c in [("acc_15k", 100), ("acc_15k", 1000), ("gap_15k", 100), ("gap_15k", 1000)]) + r" \\")
L.append("Target shuffled across configs: $R^2(B_1)$, $R^2(B_1{+}G)$ & " + " & ".join(f"{sg(shuf[(t, c)][0], 2, False)}, {sg(shuf[(t, c)][1], 2, False)}" for t, c in [("acc_15k", 100), ("acc_15k", 1000), ("gap_15k", 100), ("gap_15k", 1000)]) + r" \\")
L.append(r"\midrule \multicolumn{5}{l}{\emph{Positive controls (expected: detected)}} \\")
for nz in ["0.05", "0.2", "0.4"]:
    L.append(f"Noisy copy of the target (noise {nz} s.d.), added to $B_1$ & " + " & ".join(
        ci(*A3[(t, c, nz)], nd=2) for t, c in [("acc_15k", 100), ("acc_15k", 1000), ("gap_15k", 100), ("gap_15k", 1000)]) + r" \\")
L.append("Feature with correlation 0.5 to out-of-fold residual & " + " & ".join(cell(("[E-pos-resid]", t, c, "feature corr 0.5 with OOF residual")) for t, c in [("acc_15k", 100), ("acc_15k", 1000), ("gap_15k", 100), ("gap_15k", 1000)]) + r" \\")
L += [r"\bottomrule", r"\end{tabular}"]
write("tab_audit", "\n".join(L))

# ------------------------------------------------------------------ step-2 timing tables (3 sets)
METR = [("Fisher", "Fisher"), ("eff. rank (PR)", "Effective rank"), ("Mag(t=1)", "Mag$(t{=}1)$"), ("PMag[.5,10]", "PMag (fixed window)"),
        ("probe acc", "Probe accuracy"), ("H1 top-5", "$H_1$ top-5"), ("H1 total", "$H_1$ total"), ("raw norm", "Latent norm")]
pat_t = re.compile(r"median ([+-][\d.]+) \[([+-][\d.]+),([+-][\d.]+)\] (?:over (\d+) valid|n=(\d+))")


def parse_a(path):
    out = {}
    on = False
    for ln in open(path).read().splitlines():
        if ln.startswith("== ("):
            on = ln.startswith("== (a)")
            continue
        if on:
            for nm, _ in METR:
                if ln.strip().startswith(nm + " ") and nm not in out:
                    m = pat_t.search(ln)
                    if m:
                        out[nm] = (float(m.group(1)), float(m.group(2)), float(m.group(3)), int(m.group(4)))
    return out


def parse_c(path):
    out = {}
    on = False
    for ln in open(path).read().splitlines():
        if ln.startswith("== ("):
            on = ln.startswith("== (c)")
            continue
        if on and ln.strip().startswith("vs test_acc"):
            rest = ln.strip()[len("vs test_acc"):].strip()
            for nm, _ in METR:
                if rest.startswith(nm + " ") and nm not in out:
                    m = pat_t.search(ln)
                    if m:
                        out[nm] = (float(m.group(1)), float(m.group(2)), float(m.group(3)), int(m.group(5)))
                    break
    return out


SETS = [("All (69 configs)", f"{EXT}/step2_summary.txt"), ("Adam (63)", f"{A2}/ext/ext_adam/step2_summary.txt"), ("SGD (6)", f"{A2}/ext/sgd/step2_summary.txt")]
for kind, fn in [("a", parse_a), ("c", parse_c)]:
    P = [fn(p) for _, p in SETS]
    L = [r"\begin{tabular}{l" + "c" * len(SETS) + "}", r"\toprule", "Metric & " + " & ".join(n for n, _ in SETS) + r" \\", r"\midrule"]
    for nm, lab in METR:
        cells = []
        for p in P:
            if nm in p:
                m, lo, hi, n = p[nm]
                cells.append(f"{sg(m, 2)} [{sg(lo, 2)}, {sg(hi, 2)}] ({n})")
            else:
                cells.append("--")
        L.append(f"{lab} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write(f"tab_timing_{kind}", "\n".join(L))

# ------------------------------------------------------------------ redundancy
red = read_csv(f"{EXT}/step2_redundancy.csv")
NAMES = OrderedDict([("test_acc", "Test accuracy"), ("gap", "Gap"), ("norm", "Latent norm"), ("raw norm", "Latent norm"), ("probe_acc", "Probe accuracy"),
                     ("probe acc", "Probe accuracy"), ("eff_rank_pr", "Effective rank (PR)"), ("eff. rank (PR)", "Effective rank (PR)"),
                     ("h0_total", "$H_0$ total"), ("h1_total", "$H_1$ total"), ("h1_top5", "$H_1$ top-5"), ("h1_top10", "$H_1$ top-10"),
                     ("h1_entropy", "$H_1$ entropy"), ("mag_t1", "Mag$(t{=}1)$"), ("pmag_fixed", "PMag (fixed window)")])
L = [r"\begin{tabular}{lrrrr}", r"\toprule", r"Metric & $\rho$ with Fisher [95\% CI] & $\rho$ with test acc. & $\rho$ with gap & partial $\rho$(gap $\mid$ Fisher) \\", r"\midrule"]
seen = set()
for r in red:
    nm = r["metric"]
    lab = NAMES.get(nm)
    if lab is None or lab in seen:
        continue
    seen.add(lab)
    L.append(f"{lab} & {ci(fnum(r['rho_with_fisher']), fnum(r['ci_lo']), fnum(r['ci_hi']), 2)} & {sg(fnum(r['rho_testacc']), 2)} & {sg(fnum(r['rho_gap']), 2)} & {sg(fnum(r['partial_rho_gap|fisher']), 2)} \\\\")
L += [r"\bottomrule", r"\end{tabular}"]
write("tab_redundancy", "\n".join(L))
print("redundancy metrics seen:", [r["metric"] for r in red])

# ------------------------------------------------------------------ CIFAR
txt = open(f"{A2}/step2_summary.txt").read()
cif = {}
for m in re.finditer(r"^\s+(n10000|nfull)\|wd0\.0005\|ln([\d.]+): fisher te=([\d.]+) tr=([\d.]+); probe_acc te=([\d.]+) tr=([\d.]+); h1_top5 te=([\d.]+) tr=([\d.]+); gap=([\d.]+) test_acc=([\d.]+)", txt, re.M):
    cif[(m.group(1), m.group(2))] = [float(m.group(i)) for i in range(3, 11)]
ff = read_csv(f"{A2}/step2_cifar_fixed_fraction.csv")
fin = {}
for r in ff:
    if abs(fnum(r["frac_of_training"]) - 1.0) < 1e-9:
        cfg = r["config"].split("|")
        fin[(cfg[1], cfg[3][2:], r["metric(delta)"])] = fnum(r["value"])
L = [r"\begin{tabular}{llrrrrrrr}", r"\toprule", r"$n$ & Noise & Test acc. & Train acc. & Gap & $\Delta$Fisher & $\Delta$probe acc. & $\Delta$eff.\ rank & $\Delta$Mag$(t{=}1)$ \\", r"\midrule"]
for n in ["n10000", "nfull"]:
    for ln in ["0", "0.2", "0.4"]:
        v = cif[(n, ln)]
        te = v[7]; gap = v[6]
        L.append(f"{'10{,}000' if n == 'n10000' else '50{,}000'} & {ln} & {te:.3f} & {te + gap:.3f} & {gap:.3f} & {sg(fin[(n, ln, 'fisher')], 2)} & {sg(fin[(n, ln, 'probe_acc')], 2)} & {sg(fin[(n, ln, 'eff_rank_pr')], 2)} & {sg(fin[(n, ln, 'mag_t1')], 2)} \\\\")
    if n == "n10000":
        L.append(r"\midrule")
L += [r"\bottomrule", r"\end{tabular}"]
write("tab_cifar", "\n".join(L))
print("cifar keys", sorted(set(k[2] for k in fin)))

# sign agreement with MNIST matched-gap bins
mm = read_csv(f"{A2}/step2_cifar_vs_mnist_matched_gap.csv")
agree = {}
for r in mm:
    d, med = fnum(r["delta"]), fnum(r["mnist_median"])
    if not np.isfinite(med):
        continue
    agree.setdefault(r["metric"], []).append((np.sign(d) == np.sign(med)))
for k, v in agree.items():
    print("sign agreement", k, sum(v), "/", len(v))
L = [r"\begin{tabular}{lc}", r"\toprule", r"Train$-$test divergence & CIFAR configs matching MNIST sign \\", r"\midrule"]
LAB = {"fisher": "Fisher", "probe acc": "Probe accuracy", "probe_acc": "Probe accuracy", "eff. rank (PR)": "Effective rank", "eff_rank_pr": "Effective rank",
       "h1_top5": "$H_1$ top-5", "H1 top-5": "$H_1$ top-5", "mag_t1": "Mag$(t{=}1)$", "Mag(t=1)": "Mag$(t{=}1)$", "pmag_fixed": "PMag (fixed window)", "PMag[.5,10]": "PMag (fixed window)",
       "Fisher": "Fisher"}
for k, v in agree.items():
    L.append(f"{LAB.get(k, k)} & {sum(v)} of {len(v)} \\\\")
L += [r"\bottomrule", r"\end{tabular}"]
write("tab_cifar_sign", "\n".join(L))

# ------------------------------------------------------------------ l2 temperature
l2 = read_csv(f"{A2}/l2t/l2t_summary.csv")
rr = read_csv(f"{A2}/l2t/l2t_ratio_vs_none.csv")
VN = {"none": "none", "s1": r"$\ell_2$, $s{=}1$", "s10": r"$\ell_2$, $s{=}10$", "learn10": r"$\ell_2$, learned ($s{\approx}11$)"}
L = [r"\begin{tabular}{llrrrrr}", r"\toprule", r"Dataset & Normalization & Fisher (test probe) & Ratio vs.\ none & Offline probe acc. & Test acc. & Train acc. \\", r"\midrule"]
for ds, dn in [("mnist", "MNIST"), ("fashion_mnist", "Fashion-MNIST")]:
    for v in ["none", "s1", "s10", "learn10"]:
        r = [q for q in l2 if q["dataset"] == ds and q["variant"] == v and q["target"] == "final"][0]
        ra = [q for q in rr if q["dataset"] == ds and q["variant"] == v and q["target"] == "final"]
        rat = f"{fnum(ra[0]['ratio_mean']):.2f}$\\times$" if ra else "1"
        L.append(f"{dn if v == 'none' else ''} & {VN[v]} & {fnum(r['fisher_test_mean']):.2f} & {rat} & {fnum(r['offline_acc_test_mean']):.3f} & {fnum(r['test_acc_mean']):.3f} & {fnum(r['train_acc_mean']):.3f} \\\\")
    if ds == "mnist":
        L.append(r"\midrule")
L += [r"\bottomrule", r"\end{tabular}"]
write("tab_l2t", "\n".join(L))

# ------------------------------------------------------------------ PMag window
pm = read_csv(f"{RE2}/pmag_l2_ratio.csv")
L = [r"\begin{tabular}{lrrrrr}", r"\toprule", r"Dataset & per-snapshot window & per-run final window & dataset-shared window & wide window & universal $[0.5,10]$ \\", r"\midrule"]
for ds, dn in [("mnist", "MNIST"), ("fashion_mnist", "Fashion-MNIST"), ("cifar10", "CIFAR-10")]:
    r = [q for q in pm if q["dataset"] == ds and q["config"] == "probe/l2"][0]
    L.append(f"{dn} & " + " & ".join(f"{fnum(r[k]):.2f}$\\times$" for k in ["pmag_csv_ratio", "A_ratio", "B_ratio", "Wd_ratio", "U_ratio"]) + r" \\")
L += [r"\bottomrule", r"\end{tabular}"]
write("tab_pmag_ratio", "\n".join(L))

# ------------------------------------------------------------------ slingshot tables
adam = [r for r in read_csv(f"{DYN}/compare_adam_sgd/adam_vs_sgd.csv") if fnum(r["wd"]) in (0.0, 1e-4)]
L = [r"\begin{tabular}{rrrrrrrr}", r"\toprule", r" & & \multicolumn{3}{c}{Adam} & \multicolumn{3}{c}{SGD (momentum 0.9)} \\", r"$n$ & wd & test acc. & events/run & first event & test acc. & events/run & final Fisher \\", r"\midrule"]
ph_a = {}
for r in read_csv(f"{DYN}/sl40k/phase_grid.csv"):
    ph_a[(int(fnum(r["N"])), fnum(r["wd"]))] = r
ph_s = {}
for r in read_csv(f"{DYN}/sgd40k/phase_grid.csv"):
    ph_s[(int(fnum(r["N"])), fnum(r["wd"]))] = r
for r in adam:
    n, wd = int(fnum(r["N"])), fnum(r["wd"])
    ons = fnum(r["adam_slingshot_onset_step_mean"])
    L.append(f"{n} & {'0' if wd == 0 else '$10^{-4}$'} & {fnum(r['adam_final_test_acc_mean']):.3f} & {fnum(r['adam_slingshot_n_events_mean']):.1f} & "
             f"{('%d' % ons) if np.isfinite(ons) else '--'} & {fnum(r['sgd_final_test_acc_mean']):.3f} & {fnum(r['sgd_slingshot_n_events_mean']):.1f} & "
             f"{fnum(ph_s[(n, wd)]['final_fisher_mean']):.2f} (Adam {fnum(ph_a[(n, wd)]['final_fisher_mean']):.2f}) \\\\")
L += [r"\bottomrule", r"\end{tabular}"]
write("tab_sling_adam_sgd", "\n".join(L))

# events by wd and type, 20k grid + 40k grid
ev = read_csv(f"{DYN}/slingshot_events.csv") + read_csv(f"{DYN}/sl40k/slingshot_events.csv")
ps = read_csv(f"{DYN}/phase_summary.csv") + read_csv(f"{DYN}/sl40k/phase_summary.csv")
by = {}
for r in ps:
    if fnum(r["label_noise"]) != 0:
        continue
    k = fnum(r["wd"])
    by.setdefault(k, {"runs": 0, "N": 0, "F": 0})
    by[k]["runs"] += 1
    typ = (r.get("slingshot_type") or "").strip()
    nN = int(fnum(r.get("slingshot_n_events_N") or 0))
    nF = int(fnum(r.get("slingshot_n_events_F") or 0))
    by[k]["N"] += 1 if nN > 0 else 0
    by[k]["F"] += 1 if nF > 0 else 0
L = [r"\begin{tabular}{rrrr}", r"\toprule", r"Weight decay & Runs & Runs with type N & Runs with type F \\", r"\midrule"]
def wdlab(k):
    if k == 0:
        return "0"
    e = int(np.floor(np.log10(k) + 1e-9))
    mnt = k / 10 ** e
    return "$10^{%d}$" % e if abs(mnt - 1) < 1e-6 else "$%d{\\cdot}10^{%d}$" % (round(mnt), e)


for k in sorted(by):
    L.append(wdlab(k) + f" & {by[k]['runs']} & {by[k]['N']} & {by[k]['F']} \\\\")
L += [r"\bottomrule", r"\end{tabular}"]
write("tab_sling_wd", "\n".join(L))
print(by)

# ------------------------------------------------------------------ label noise crossover
cs = read_csv(f"{A2}/config_summary.csv")
L = [r"\begin{tabular}{rrrrr}", r"\toprule", r"$n$ & noise & peak test acc. & peak step & crossover step \\", r"\midrule"]
rowsn = []
for r in cs:
    m = re.match(r"mnist\|n(\w+)\|wd0\|ln(0\.\d)$", r["config_id"])
    if m:
        n = 60000 if m.group(1) == "full" else int(m.group(1))
        rowsn.append((float(m.group(2)), n, r))
for ln, n, r in sorted(rowsn):
    L.append(f"{n:,} & {ln} & {fnum(r['peak_test_acc']):.3f} & {fnum(r['peak_step']):.0f} & {fnum(r['cross_step_mean']):.0f} \\\\".replace(",", "{,}"))
L += [r"\bottomrule", r"\end{tabular}"]
write("tab_noise", "\n".join(L))
