"""Compact text table of step3_predict_ext results. usage: ANALYSIS_OUT=<name> python step3_report_ext.py"""
import common2 as C
import csv
rows = list(csv.DictReader(open(f"{C.OUT}/step3_results.csv")))
f = lambda x: float(x) if x not in ("", None) else float("nan")
out = []
for tag in dict.fromkeys(r["tag"] for r in rows):
    for t in dict.fromkeys(r["target"] for r in rows if r["tag"] == tag):
        for mdl in ("ridge", "gbm"):
            sub = [r for r in rows if r["tag"] == tag and r["target"] == t and r["model"] == mdl]
            if not sub: continue
            cuts = sorted({int(r["cutoff"]) for r in sub})
            n = sub[0]["n_runs"]
            def get(kind, name, base="", c=None):
                return next((r for r in sub if r["kind"] == kind and r["set"] == name and r["base"] == base and int(r["cutoff"]) == c), None)
            out.append(f"[{tag}] {t} {mdl} (n_runs={n})  cutoffs {cuts}")
            out.append("   R2 B1   " + " ".join(f"{f(get('set','B1',c=c)['r2']):+.2f}" for c in cuts))
            out.append("   R2 G    " + " ".join(f"{f(get('set','G',c=c)['r2']):+.2f}" for c in cuts))
            out.append("   R2 B1+G " + " ".join(f"{f(get('set','B1+G',c=c)['r2']):+.2f}" for c in cuts))
            for base, comp in (("B0", "B0+G"), ("B1", "B1+G")):
                out.append(f"   dR2 {base}->{comp} " + " ".join(
                    (lambda r: f"{f(r['delta']):+.3f}[{f(r['ci_lo']):+.3f},{f(r['ci_hi']):+.3f}]" + (f"p={f(r['perm_p']):.3f}" if r['perm_p'] != '' else ''))(get('increment', comp, base, c)) for c in cuts))
open(f"{C.OUT}/step3_summary.txt", "w").write("\n".join(out)); print("\n".join(out))
