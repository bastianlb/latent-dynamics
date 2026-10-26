#!/usr/bin/env python
"""Amendment 6 / E (post hoc): non-geometric placebo from data on disk. Weight/gradient norms are NOT logged for the confirmatory runs (or set (ii));
the only non-latent logged quantities are the run/* scalar curves (loss, accuracy) at every snapshot. Block CV18 = 6 curve quantities
(log train loss, log test loss, train acc, test acc, gap, log(test/train loss)) x 3 curve statistics computed from snapshots with step <= the G cutoff snapshot:
 (a) value at the last snapshot with step <= cutoff_step/10, (b) OLS slope vs log10(step+1) over the last <=5 snapshots (0 if <3), (c) std over all snapshots <= cutoff."""
import step8_common as H
import csv
import numpy as np
S, M, Z, D = H.S, H.M, H.Z, H.D
lg = lambda x: np.log(np.maximum(x, 1e-6))
new = H.load_new(); n = len(new)


def curve_block(r, cut):
    d = r["d"]; st = r["step"]; k = D.idx_le_mask(st, r["topo"], cut); sc = st[k]
    tl, te, tra, tea = d["run/train_loss"], d["run/test_loss"], d["run/train_acc"], d["run/test_acc"]
    qs = [lg(tl), lg(te), tra, tea, tra - tea, lg(te) - lg(tl)]
    ok = np.where(st <= sc)[0]; i10 = np.where(st <= sc / 10)[0]; i10 = i10[-1] if len(i10) else 0
    out = []
    for q in qs:
        v = q[ok]; x = np.log10(st[ok] + 1.0)
        last = slice(max(0, len(ok) - 5), len(ok))
        if len(ok) - max(0, len(ok) - 5) >= 3 and np.ptp(x[last]) > 0:
            sl = np.polyfit(x[last], v[last], 1)[0]
        else:
            sl = 0.0
        out += [q[i10], sl, v.std()]
    return np.array(out)


rows = []
for t in H.TARGETS:
    m = H.margin(t)
    for cut in H.CUTS:
        y, B1, G, names, g, B0 = H.cut_data(new, cut, t)
        c4 = M.gvariant_cols(names[2], "compact4")
        CV = np.array([curve_block(r, cut) for r in new])
        P = {"b": S.loco(B1, y, g, "ridge"), "F": S.loco(np.hstack([B1, G]), y, g, "ridge"), "C": S.loco(np.hstack([B1, G[:, c4]]), y, g, "ridge"),
             "cv": S.loco(np.hstack([B1, CV]), y, g, "ridge"), "cv4": S.loco(np.hstack([B1, CV[:, [0, 3, 6, 9]]]), y, g, "ridge")}
        st = M.CfgStats(y, P, g); W = st.draw(H.NB, np.random.default_rng(0)); W1 = np.ones((1, st.nc))
        def ci(a, b):
            d = st.d_r2(a, b, W); o = st.d_r2(a, b, W1)[0]; q = np.percentile(d, [2.5, 97.5, 5, 95])
            return dict(delta=o, lo95=q[0], hi95=q[1], lo90=q[2], hi90=q[3], p_neg=(np.sum(d >= 0) + 1) / (H.NB + 1), p_pos=(np.sum(d <= 0) + 1) / (H.NB + 1))
        for nm, a, b in (("CV18 vs B1", "b", "cv"), ("CV4(value@cut/10 of 4 curves) vs B1", "b", "cv4"), ("G vs B1", "b", "F"), ("compact4 vs B1", "b", "C"), ("G minus CV18", "cv", "F")):
            rows.append(dict(contrast=nm, target=t, cutoff=cut, margin=m, r2_B1=S.r2(y, P["b"]), **ci(a, b)))
        print(t, cut, flush=True)
for nm in {r["contrast"] for r in rows}:
    fam = [r for r in rows if r["contrast"] == nm]
    for r, h in zip(fam, H.K7.holm([r["p_neg"] for r in fam])): r["p_neg_holm"] = h
with open(f"{H.OUT}/E_curve_block_placebo.csv", "w", newline="") as f:
    w = csv.DictWriter(f, list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("done")
