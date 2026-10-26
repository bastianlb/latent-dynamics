"""Draft-2 tables: equivalence classes, extrapolation splits, residual targets, B2 baseline, power, timing sensitivity,
multi-step redundancy, PMag windows across datasets, Fashion replication. Reads analysis_v2/ext2/{mnist,fashion}/*.csv|txt.
Also writes paper_v2_figs/paper_stats.txt with every aggregate number quoted in paper.tex that is not a single table cell."""
from common import *
from collections import Counter, OrderedDict

TAB = f"{BIT}/paper_v2/tables"
E2 = f"{A2}/ext2"
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


def ci(d, lo, hi, nd=3):
    return f"{sg(d, nd)} [{sg(lo, nd)}, {sg(hi, nd)}]"


def write(name, txt):
    with open(f"{TAB}/{name}.tex", "w") as f:
        f.write(txt + "\n")
    print("wrote", name)


def load(ds, name):
    return read_csv(f"{E2}/{ds}/{name}")


CUTS = [10, 30, 100, 300, 1000]
TGT = {"acc_15kw": "test acc.", "gap_15kw": "gap", "acc_15k": "test acc.\\ (step 15k)", "gap_15k": "gap (step 15k)"}


def code(lbl):
    harm = "possible harm" in lbl
    if lbl.startswith("equivalent"):
        c = "E"
    elif lbl.startswith("positive"):
        c = "P"
    else:
        c = "I"
    return c + ("$^\\downarrow$" if harm else "")


def idx_main(ds):
    d = {}
    for r in load(ds, "step4_main_results.csv"):
        d[(r["kind"], r["target"], int(r["cutoff"]), r["model"], r["set"], r["base"])] = r
    return d


M = {"mnist": idx_main("mnist"), "fashion": idx_main("fashion")}


def inc(ds, tgt, c, model, comp="B1+G", base="B1"):
    return M[ds][("increment", tgt, c, model, comp, base)]


# ---------------------------------------------------------------- counts (stats)
for ds in ["mnist", "fashion"]:
    for model in ["ridge", "gbm"]:
        for comp, base in [("B1+G", "B1"), ("B2+G", "B2")]:
            cnt = Counter()
            cells = []
            for tgt in TGT:
                for c in CUTS:
                    r = inc(ds, tgt, c, model, comp, base)
                    cl = r["label"]
                    k = "E" if cl.startswith("equivalent") else ("P" if cl.startswith("positive") else "I")
                    cnt[k] += 1
                    cells.append((tgt, c, k, fnum(r["delta"]), fnum(r["lo95"]), fnum(r["hi95"]), fnum(r["lo90"]), fnum(r["hi90"]), r["label"]))
            st(f"[equiv-count] {ds} {model} {base}->{comp} (4 targets x 5 cutoffs): E={cnt['E']} P={cnt['P']} I={cnt['I']}")
            for t, c, k, d, lo, hi, lo9, hi9, lb in cells:
                if k != "E":
                    st(f"    non-E: {t} cut{c} d={d:+.3f} 95%[{lo:+.3f},{hi:+.3f}] 90%[{lo9:+.3f},{hi9:+.3f}] {lb}")
            if comp == "B1+G" and ds == "mnist":
                for pre in ["acc", "gap"]:
                    dd = [x for x in cells if x[0].startswith(pre)]
                    st(f"    {pre} dR2 range {min(x[3] for x in dd):+.3f}..{max(x[3] for x in dd):+.3f}; max 95% upper {max(x[5] for x in dd):+.3f}; "
                       f"cells with 95% CI wholly above 0: {[(x[0], x[1]) for x in dd if x[4] > 0]}; wholly below 0: {[(x[0], x[1]) for x in dd if x[5] < 0]}")

# R2(B1) ranges
for ds in ["mnist", "fashion"]:
    for pre in ["acc", "gap"]:
        v = [fnum(M[ds][("set", t, c, "ridge", "B1", "")]["r2"]) for t in TGT if t.startswith(pre) for c in CUTS]
        st(f"[R2(B1)] {ds} {pre} ridge range {min(v):.3f}..{max(v):.3f}")
# min margin m* for gap cut1000
for ds in ["mnist", "fashion"]:
    r = inc(ds, "gap_15kw", 1000, "ridge")
    st(f"[m*] {ds} gap_15kw cut1000 ridge d={fnum(r['delta']):+.3f} 95%[{fnum(r['lo95']):+.3f},{fnum(r['hi95']):+.3f}] m*={fnum(r['min_margin']):.3f}")
    r = inc(ds, "gap_15kw", 1000, "gbm")
    st(f"[m*] {ds} gap_15kw cut1000 gbm d={fnum(r['delta']):+.3f} 95%[{fnum(r['lo95']):+.3f},{fnum(r['hi95']):+.3f}] m*={fnum(r['min_margin']):.3f}")

# ---------------------------------------------------------------- main equivalence-class table
L = [r"\begin{tabular}{llccccc}", r"\toprule", r"Data, model & Target & 10 & 30 & 100 & 300 & 1000 \\", r"\midrule"]
ROWS = [("mnist", "ridge", "MNIST, ridge"), ("mnist", "gbm", "MNIST, GBM"), ("fashion", "ridge", "Fashion, ridge"), ("fashion", "gbm", "Fashion, GBM")]
for ds, model, lab in ROWS:
    for j, tgt in enumerate(["acc_15kw", "gap_15kw"]):
        cells = []
        for c in CUTS:
            r = inc(ds, tgt, c, model)
            d = fnum(r["delta"])
            ds_ = f"{abs(d):.3f}"[1:] if abs(d) < 1 else f"{abs(d):.2f}"
            sign = "$-$" if (d < 0 and float(f"{abs(d):.3f}") != 0) else "+"
            cells.append(f"{sign}{ds_}\\,{code(r['label'])}")
        L.append(f"{lab if j == 0 else ''} & {'test acc.' if j == 0 else 'gap'} & " + " & ".join(cells) + r" \\")
    if model == "gbm" and ds == "mnist":
        L.append(r"\midrule")
L += [r"\bottomrule", r"\end{tabular}"]
write("tab_equiv", "\n".join(L))

# ---------------------------------------------------------------- full equivalence tables
for ds in ["mnist", "fashion"]:
    L = []
    for tgt in TGT:
        for c in CUTS:
            ro, gb = inc(ds, tgt, c, "ridge"), inc(ds, tgt, c, "gbm")
            f = lambda r: f"{ci(fnum(r['delta']), fnum(r['lo90']), fnum(r['hi90']))} {code(r['label'])}"
            L.append(f"{TGT[tgt] if c == 10 else ''} & {c} & {f(ro)} & {fnum(ro['min_margin']):.3f} & {f(gb)} & {fnum(gb['min_margin']):.3f} \\\\")
        L.append(r"\midrule")
    L[-1] = r"\bottomrule"
    write(f"tab_equiv_full_{ds}_body", "\n".join(L))

# ---------------------------------------------------------------- B2 table (MNIST)
L = []
for model in ["ridge", "gbm"]:
    for tgt in ["acc_15kw", "gap_15kw"]:
        for c in CUTS:
            b1 = fnum(M["mnist"][("set", tgt, c, model, "B1", "")]["r2"])
            b2 = fnum(M["mnist"][("set", tgt, c, model, "B2", "")]["r2"])
            r1 = inc("mnist", tgt, c, model, "B2", "B1")
            r2 = inc("mnist", tgt, c, model, "B2+G", "B2")
            L.append(f"{model if (tgt == 'acc_15kw' and c == 10) else ''} & {TGT[tgt] if c == 10 else ''} & {c} & {b1:.2f} & {b2:.2f} & "
                     f"{ci(fnum(r1['delta']), fnum(r1['lo95']), fnum(r1['hi95']))} & {ci(fnum(r2['delta']), fnum(r2['lo90']), fnum(r2['hi90']))} {code(r2['label'])} \\\\")
        L.append(r"\midrule")
L[-1] = r"\bottomrule"
write("tab_b2_body", "\n".join(L))
for tgt in ["acc_15kw", "gap_15kw"]:
    dd = [fnum(inc("mnist", tgt, c, "ridge", "B2", "B1")["delta"]) for c in CUTS]
    st(f"[B2] ridge B1->B2 {tgt}: {['%+.3f' % x for x in dd]}")

# ---------------------------------------------------------------- extrapolation splits
SPL = OrderedDict([("leave_one_n_out", "leave-one-$n$-out"), ("leave_one_wd_out", "leave-one-wd-out"), ("leave_one_optimizer_out", "Adam$\\to$SGD")])
sp = {}
for ds in ["mnist", "fashion"]:
    for r in load(ds, "step4_splits_results.csv"):
        if r["fold"] in ("POOLED", "test=SGD"):
            if r["split"] == "leave_one_optimizer_out" and r["fold"] == "POOLED":
                pass
            sp[(ds, r["split"], r["target"], int(r["cutoff"]), r["model"])] = r
L = []
for ds, splits in [("mnist", list(SPL)), ("fashion", ["leave_one_n_out"])]:
    for s in splits:
        for tgt in ["acc_15kw", "gap_15kw"]:
            for model in ["ridge", "gbm"]:
                cells = []
                for c in CUTS:
                    r = sp[(ds, s, tgt, c, model)]
                    cells.append("\\begin{tabular}[c]{@{}c@{}}" + ci(fnum(r["delta"]), fnum(r["lo95"]), fnum(r["hi95"]), 2) + "\\\\ {\\scriptsize " +
                                 f"{fnum(r['r2_B1']):.2f}$\\to${fnum(r['r2_B1G']):.2f}" + "}\\end{tabular}")
                lab = ("Fashion " if ds == "fashion" else "") + SPL[s]
                L.append(f"{lab if (tgt == 'acc_15kw' and model == 'ridge') else ''} & {'test acc.' if tgt.startswith('acc') else 'gap'} & {model} & " + " & ".join(cells) + r" \\")
        L.append(r"\midrule")
L[-1] = r"\bottomrule"
write("tab_extrap_body", "\n".join(L))

for ds in ["mnist", "fashion"]:
    for s in (SPL if ds == "mnist" else ["leave_one_n_out"]):
        for tgt in ["acc_15kw", "gap_15kw"]:
            for model in ["ridge", "gbm"]:
                rr = [sp[(ds, s, tgt, c, model)] for c in CUTS]
                pos = [c for c, r in zip(CUTS, rr) if fnum(r["lo95"]) > 0]
                neg = [c for c, r in zip(CUTS, rr) if fnum(r["hi95"]) < 0]
                st(f"[extrap] {ds} {s} {tgt} {model}: dR2 " + " ".join(f"{fnum(r['delta']):+.3f}" for r in rr) +
                   f" | 95% CI wholly >0 at cutoffs {pos}; wholly <0 at {neg}")
# ridge+gbm across splits: how many (split,target,model) cells at cutoff>=100 have point estimate <0
cnt = Counter()
for s in SPL:
    for tgt in ["acc_15kw", "gap_15kw"]:
        for model in ["ridge", "gbm"]:
            for c in [100, 300, 1000]:
                cnt["neg" if fnum(sp[("mnist", s, tgt, c, model)]["delta"]) < 0 else "pos"] += 1
st(f"[extrap] MNIST splits x targets x models x cutoffs>=100: dR2<0 in {cnt['neg']} of {cnt['neg'] + cnt['pos']} cells")
cnt = Counter()
for s in SPL:
    for tgt in ["acc_15kw", "gap_15kw"]:
        for model in ["ridge"]:
            for c in [100, 300, 1000]:
                cnt["neg" if fnum(sp[("mnist", s, tgt, c, model)]["delta"]) < 0 else "pos"] += 1
st(f"[extrap] MNIST ridge only, cutoffs>=100: dR2<0 in {cnt['neg']} of {cnt['neg'] + cnt['pos']} cells")
cnt = Counter()
for s in ["leave_one_n_out", "leave_one_wd_out"]:
    for tgt in ["acc_15kw", "gap_15kw"]:
        for c in [100, 300, 1000]:
            cnt["neg" if fnum(sp[("mnist", s, tgt, c, "ridge")]["delta"]) < 0 else "pos"] += 1
st(f"[extrap] MNIST ridge LONO+LOWO, cutoffs>=100: dR2<0 in {cnt['neg']} of {cnt['neg'] + cnt['pos']} cells")
cnt = Counter()
for s in ["leave_one_n_out", "leave_one_wd_out"]:
    for tgt in ["acc_15kw", "gap_15kw"]:
        for c in [10, 30]:
            cnt["pos" if fnum(sp[("mnist", s, tgt, c, "ridge")]["delta"]) > 0 else "neg"] += 1
st(f"[extrap] MNIST ridge LONO+LOWO, cutoffs 10,30: dR2>0 in {cnt['pos']} of {cnt['neg'] + cnt['pos']} cells")

# ---------------------------------------------------------------- residual targets
RV = OrderedDict([("3a_within_config", ("within-config seed residual", "B0c -> B0c+Gc")), ("3b_design_residual|lin", ("design residual, linear", "lin: B0 -> B0+G")),
                  ("3b_design_residual|quad", ("design residual, quadratic", "quad: B0 -> B0+G"))])
res = {}
for ds in ["mnist", "fashion"]:
    for r in load(ds, "step4_resid_results.csv"):
        res[(ds, r["analysis"], r["variant"], r["target"], int(r["cutoff"]))] = r
L = []
for ds in ["mnist", "fashion"]:
    for key, (lab, var) in RV.items():
        an = key.split("|")[0]
        for tgt in ["acc_15kw", "gap_15kw"]:
            cells = []
            for c in CUTS:
                r = res[(ds, an, var, tgt, c)]
                cells.append("\\begin{tabular}[c]{@{}c@{}}" + sg(fnum(r["delta"]), 2) + "\\\\ {\\scriptsize [" + sg(fnum(r["lo95"]), 2) + ", " + sg(fnum(r["hi95"]), 2) + "]}\\end{tabular}")
            sd = fnum(res[(ds, an, var, tgt, 100)]["within_cfg_sd" if an.startswith("3a") else "resid_sd"])
            L.append(f"{('MNIST' if ds == 'mnist' else 'Fashion') if (key.startswith('3a') and tgt == 'acc_15kw') else ''} & {lab if tgt == 'acc_15kw' else ''} & "
                     f"{'test acc.' if tgt.startswith('acc') else 'gap'} & {sd:.4f} & " + " & ".join(cells) + r" \\")
    L.append(r"\midrule")
L[-1] = r"\bottomrule"
write("tab_resid_body", "\n".join(L))
for ds in ["mnist", "fashion"]:
    for key, (lab, var) in RV.items():
        an = key.split("|")[0]
        for tgt in ["acc_15kw", "gap_15kw"]:
            st(f"[resid] {ds} {key} {tgt}: " + " ".join(f"c{c}:{fnum(res[(ds, an, var, tgt, c)]['delta']):+.3f}[{fnum(res[(ds, an, var, tgt, c)]['lo95']):+.3f},{fnum(res[(ds, an, var, tgt, c)]['hi95']):+.3f}]" for c in CUTS))
st("[resid] within-config SD (mnist acc/gap, fashion acc/gap):", [round(fnum(res[(ds, "3a_within_config", "B0c -> B0c+Gc", t, 100)]["within_cfg_sd"]), 4) for ds in ["mnist", "fashion"] for t in ["acc_15kw", "gap_15kw"]])
st("[resid] residual SD design lin/quad mnist acc:", [round(fnum(res[("mnist", "3b_design_residual", v, "acc_15kw", 100)]["resid_sd"]), 4) for v in ["lin: B0 -> B0+G", "quad: B0 -> B0+G"]],
   "gap:", [round(fnum(res[("mnist", "3b_design_residual", v, "gap_15kw", 100)]["resid_sd"]), 4) for v in ["lin: B0 -> B0+G", "quad: B0 -> B0+G"]])
st("[resid] quad design max dR2 mnist:", max(fnum(res[("mnist", "3b_design_residual", "quad: B0 -> B0+G", t, c)]["delta"]) for t in ["acc_15kw", "gap_15kw"] for c in CUTS))

# ---------------------------------------------------------------- power
pw = load("mnist", "step4_power_summary.csv")
L = []
seen = []
for r in pw:
    k = (r["target"], r["cutoff"], r["variant"])
    if k not in seen:
        seen.append(k)
for k in seen:
    rows = {fnum(r["rho"]): r for r in pw if (r["target"], r["cutoff"], r["variant"]) == k}
    mde = rows[0.0]["mde_80"]
    mde_s = mde if mde and mde.replace(".", "").isdigit() else "not reached"
    cells = " & ".join(f"{fnum(rows[x]['detection_prob']):.2f} ({sg(fnum(rows[x]['mean_delta']), 3)})" for x in [0.3, 0.5, 0.7])
    L.append(f"{TGT[k[0]]} & {k[1]} & {k[2].replace(' + s', '')} & {fnum(rows[0.0]['r2_B1']):.3f} & {mde_s} & {cells} \\\\")
L.append(r"\bottomrule")
write("tab_power_body", "\n".join(L))
st("[power] MDE per cell:", [(k, [r['mde_80'] for r in pw if (r['target'], r['cutoff'], r['variant']) == k][0]) for k in seen])

# ---------------------------------------------------------------- timing sensitivity
ts = load("mnist", "step4_timing_summary.csv")
MET = [("fisher", "Fisher", "robust: lags"), ("eff_rank_pr", "Effective rank", "robust: leads"), ("mag_t1", "Mag$(t{=}1)$", "robust: leads"),
       ("probe_acc", "Probe accuracy", "sign flips"), ("h1_top5", "$H_1$ top-5", "sign flips"), ("pmag_fixed", "PMag (fixed window)", "leads, weak CI"), ("h1_total", "$H_1$ total", "sign flips")]
tk = {}
for r in ts:
    tk[(r["variant"], r["q"], r["ref"], r["metric"])] = r
st("[timing-keys]", sorted(set(r["metric"] for r in ts)), sorted(set(r["q"] for r in ts)), sorted(set(r["variant"] for r in ts)))
L = []
for mk, lab, _ in MET:
    cells = []
    for var, q in [("all_runs", "0.25"), ("all_runs", "0.5"), ("all_runs", "0.75"), ("excl_slingshot", "0.25"), ("excl_slingshot", "0.5"), ("excl_slingshot", "0.75")]:
        r = tk.get((var, q, "gap", mk))
        if r is None:
            cells.append("--")
        else:
            cells.append(f"{ci(fnum(r['median']), fnum(r['lo']), fnum(r['hi']), 2)} ({r['n_valid']})")
    L.append(f"{lab} & " + " & ".join(cells) + r" \\")
L.append(r"\bottomrule")
write("tab_timing_sens_body", "\n".join(L))
for mk, lab, _ in MET:
    vals = []
    for var, q in [("all_runs", "0.25"), ("all_runs", "0.5"), ("all_runs", "0.75"), ("excl_slingshot", "0.25"), ("excl_slingshot", "0.5"), ("excl_slingshot", "0.75")]:
        r = tk.get((var, q, "gap", mk))
        vals.append((fnum(r["median"]), fnum(r["lo"]), fnum(r["hi"])))
    allpos = all(v[1] > 0 for v in vals); allneg = all(v[2] < 0 for v in vals)
    st(f"[timing-robust] {mk}: CI wholly >0 in all 6 variants={allpos}; wholly <0 in all 6={allneg}; medians {[round(v[0], 2) for v in vals]}")

# undefined timing table
und = []
for ln in open(f"{E2}/mnist/step4_timing_summary.txt").read().splitlines():
    m = re.match(r"\s+q=0\.5 (.+?)\s+defined (\d+), undefined (\d+) \| median log10 n def ([\d.]+) vs undef ([\d.]+) \(MW p=([\d.]+)\); .*?label-noise>0: def ([\d.]+) undef ([\d.]+); SGD: def ([\d.]+) undef ([\d.]+)", ln)
    if m:
        und.append(m.groups())
NM = {"Fisher": "Fisher", "probe acc": "Probe accuracy", "eff. rank (PR)": "Effective rank", "H1 top-5": "$H_1$ top-5", "Mag(t=1)": "Mag$(t{=}1)$",
      "PMag[.5,10]": "PMag (fixed window)", "H1 total": "$H_1$ total", "raw norm": "Latent norm"}
L = []
for g_ in und:
    nm = NM.get(g_[0].strip(), g_[0].strip())
    L.append(f"{nm} & {g_[1]} & {g_[2]} & {float(g_[3]):.1f} / {float(g_[4]):.1f} & {g_[5]} & {float(g_[6]):.2f} / {float(g_[7]):.2f} \\\\")
L.append(r"\bottomrule")
write("tab_undef_body", "\n".join(L))
st("[undef]", und)

# ---------------------------------------------------------------- redundancy at multiple steps
rd = load("mnist", "step4_redundancy.csv")
L = []
for setn in ["set_ii (69 cfgs)", "set_i (35 cfgs)"]:
    for mk, lab in [("h1_total", "$H_1$ total"), ("mag_t1", "Mag$(t{=}1)$"), ("pmag_fixed", "PMag (fixed window)"), ("h1_top5", "$H_1$ top-5")]:
        cells = []
        for step in [300, 1000, 5000, 15000]:
            r = [x for x in rd if x["set"] == setn and x["metric"] == mk and int(x["step"]) == step][0]
            cells.append(ci(fnum(r["rho"]), fnum(r["lo"]), fnum(r["hi"]), 2))
        L.append(f"{('69 configs' if setn.startswith('set_ii') else '35 configs') if mk == 'h1_total' else ''} & {lab} & " + " & ".join(cells) + r" \\")
    L.append(r"\midrule")
L[-1] = r"\bottomrule"
write("tab_redund_steps_body", "\n".join(L))

# ---------------------------------------------------------------- PMag windows across datasets
pw2 = []
for ln in open(f"{E2}/mnist/step4_pmag_windows.txt").read().splitlines():
    m = re.match(r"(mnist|fashion_mnist|cifar10)\s+orig vs (.+?)\s+rho=([+-][\d.]+) \[([+-][\d.]+),([+-][\d.]+)\] n_runs=(\d+) configs=(\d+)", ln)
    if m:
        pw2.append(m.groups())
tbl = {}
for ds, nm, rho, lo, hi, nr, nc in pw2:
    tbl[(ds, nm.strip())] = (float(rho), float(lo), float(hi))
L = []
for ds, dn in [("mnist", "MNIST"), ("fashion_mnist", "Fashion-MNIST"), ("cifar10", "CIFAR-10")]:
    cells = [ci(*tbl[(ds, k)], nd=2) for k in ["dataset-shared window (B)", "run's own final window (A)", "Wd", "U", "Mag(t=1)"]]
    L.append(f"{dn} & " + " & ".join(cells) + r" \\")
L.append(r"\bottomrule")
write("tab_pmag_rho_body", "\n".join(L))
st("[pmag-rho]", {k: v for k, v in tbl.items() if k[1].startswith("dataset-shared")})

open("/homes/lbastian/latent-dynamics/paper_v2_figs/paper_stats.txt", "w").write("\n".join(STATS) + "\n")

# ---------------------------------------------------------------- extra stats for text (F1, F5, Fashion)
S3 = read_csv(f"{EXT}/step3_results.csv")
for tag, tg in [("pre_covered", ["acc_pre", "gap_pre", "acc_preN", "gap_preN"])]:
    for t in tg:
        v = [(int(r["cutoff"]), fnum(r["delta"]), fnum(r["ci_lo"]), fnum(r["ci_hi"])) for r in S3 if r["tag"] == tag and r["target"] == t and r["model"] == "ridge" and r["kind"] == "increment" and r["set"] == "B1+G" and r["base"] == "B1"]
        st(f"[pre-slingshot] ridge {t}: " + " ".join(f"c{c}:{d:+.3f}" for c, d, lo, hi in sorted(v)))
    allv = [fnum(r["delta"]) for r in S3 if r["tag"] == tag and r["model"] == "ridge" and r["kind"] == "increment" and r["set"] == "B1+G" and r["base"] == "B1"]
    st(f"[pre-slingshot] ridge, all 4 targets x 5 cutoffs: min {min(allv):+.3f} max {max(allv):+.3f} (n={len(allv)})")
for ds in ["fashion"]:
    for model in ["ridge", "gbm"]:
        for pre in ["acc", "gap"]:
            cells = [(t, c) for t in TGT if t.startswith(pre) for c in CUTS if fnum(inc(ds, t, c, model)["lo95"]) > 0]
            cellsb = [(t, c) for t in TGT if t.startswith(pre) for c in CUTS if fnum(inc(ds, t, c, model)["hi95"]) < 0]
            st(f"[fashion] {model} {pre}: 95% CI wholly >0 at {cells}; wholly <0 at {cellsb}")
# MNIST B0 -> B0+G at cutoff 10 and G alone at 1000 (ridge)
for t in ["acc_15kw", "gap_15kw"]:
    r = S3_ = [x for x in S3 if x["tag"] == "main" and x["target"] == t and x["cutoff"] == "10" and x["model"] == "ridge" and x["kind"] == "increment" and x["set"] == "B0+G"][0]
    st(f"[B0->B0+G] {t} cut10 ridge {fnum(r['delta']):+.3f} [{fnum(r['ci_lo']):+.3f},{fnum(r['ci_hi']):+.3f}]")
    g = [x for x in S3 if x["tag"] == "main" and x["target"] == t and x["cutoff"] == "1000" and x["model"] == "ridge" and x["kind"] == "set" and x["set"] == "G"][0]
    st(f"[G alone] {t} cut1000 ridge R2 {fnum(g['r2']):.3f}")
open("/homes/lbastian/latent-dynamics/paper_v2_figs/paper_stats.txt", "w").write("\n".join(STATS) + "\n")
