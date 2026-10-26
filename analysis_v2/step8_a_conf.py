#!/usr/bin/env python
"""Amendment 6 / A on the confirmatory set (post hoc, exploratory): (i) within-architecture fits (14 configs each) + interaction-aware B1,
(ii) 1-2 dimensional geometry blocks (te_fisher; te_eff_rank_pr; both; in-fold PCA-1/PCA-2 of G; in-fold PCA-1 of compact4),
(iii) wider ridge alpha grid and fixed-alpha ridge. B1 as in PREREG (6 curve + log10_n, log10_wd, noise, log10_lr, is_cnn)."""
import step8_common as H
import csv
import numpy as np
S, M = H.S, H.M
new = H.load_new()
is_cnn = np.array([r["info"]["is_cnn"] for r in new], bool)
DEFAULT_ALPHAS = S.ALPHAS.copy()


def write(fn, rows):
    with open(f"{H.OUT}/{fn}", "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0].keys())); w.writeheader(); w.writerows(rows)


rows_i, rows_ii, rows_iii, rows_alpha = [], [], [], []
for t in H.TARGETS:
    m = H.margin(t)
    for cut in H.CUTS:
        y, B1, G, names, g, B0 = H.cut_data(new, cut, t)
        c4 = M.gvariant_cols(names[2], "compact4"); gn = names[2]
        ix = lambda n_: gn.index(n_)
        # ---------- (i) within architecture + interaction-aware B1
        for arch, mk in (("mlp", ~is_cnn), ("cnn", is_cnn)):
            yy, gg = y[mk], g[mk]
            sets = {"B1": B1[mk], "G": np.hstack([B1[mk], G[mk]]), "c4": np.hstack([B1[mk], G[mk][:, c4]])}
            P = {k: S.loco(X, yy, gg, "ridge") for k, X in sets.items()}
            for k, lab in (("G", "full G"), ("c4", "compact4")):
                rows_i.append(dict(fit=f"within_{arch}", target=t, cutoff=cut, X=lab, margin=m, **H.stat_ci(yy, P["B1"], P[k], gg, m)))
        cn = is_cnn.astype(float)[:, None]
        B1x = np.hstack([B1, cn * B0])            # is_cnn x curve-baseline interactions
        sets = {"B1x": B1x, "Gx": np.hstack([B1x, G]), "c4x": np.hstack([B1x, G[:, c4]]), "B1": B1}
        P = {k: S.loco(X, y, g, "ridge") for k, X in sets.items()}
        for k, lab in (("Gx", "full G"), ("c4x", "compact4")):
            rows_i.append(dict(fit="pooled_interaction_B1x", target=t, cutoff=cut, X=lab, margin=m, **H.stat_ci(y, P["B1x"], P[k], g, m)))
        rows_i.append(dict(fit="pooled_B1x_vs_B1", target=t, cutoff=cut, X="interaction terms only", margin=m, **H.stat_ci(y, P["B1"], P["B1x"], g, m)))
        # ---------- (ii) low-dimensional blocks
        pb = S.loco(B1, y, g, "ridge")
        def mk_cols(cols):
            return lambda tr, te: (np.hstack([B1[tr], G[tr][:, cols]]), np.hstack([B1[te], G[te][:, cols]]))
        def mk_pca(k, cols=None):
            cc = list(range(G.shape[1])) if cols is None else cols
            def f(tr, te):
                a_, b_ = M.pca_block(G[tr][:, cc], G[te][:, cc], k); return np.hstack([B1[tr], a_]), np.hstack([B1[te], b_])
            return f
        blocks = {"te_fisher": mk_cols([ix("te_fisher")]), "te_eff_rank_pr": mk_cols([ix("te_eff_rank_pr")]),
                  "te_mag_t1": mk_cols([ix("te_mag_t1")]), "te_h1_top5": mk_cols([ix("te_h1_top5")]),
                  "fisher+eff_rank": mk_cols([ix("te_fisher"), ix("te_eff_rank_pr")]),
                  "PCA1(G,infold)": mk_pca(1), "PCA2(G,infold)": mk_pca(2), "PCA1(compact4,infold)": mk_pca(1, c4)}
        for nm, bf in blocks.items():
            p = H.loco_fn(bf, y, g)
            rows_ii.append(dict(block=nm, target=t, cutoff=cut, margin=m, **H.stat_ci(y, pb, p, g, m)))
        # ---------- (iii) alpha grids
        S.ALPHAS = np.logspace(-4, 6, 21)
        PW = {"b": S.loco(B1, y, g, "ridge"), "G": S.loco(np.hstack([B1, G]), y, g, "ridge"), "c4": S.loco(np.hstack([B1, G[:, c4]]), y, g, "ridge")}
        for k, lab in (("G", "full G"), ("c4", "compact4")):
            rows_iii.append(dict(grid="wide_1e-4..1e6_nested", alpha="nested", target=t, cutoff=cut, X=lab, margin=m, **H.stat_ci(y, PW["b"], PW[k], g, m)))
        for al in (0.1, 1, 10, 100, 1e3, 1e4):
            S.ALPHAS = np.array([al])
            PF = {"b": S.loco(B1, y, g, "ridge"), "G": S.loco(np.hstack([B1, G]), y, g, "ridge"), "c4": S.loco(np.hstack([B1, G[:, c4]]), y, g, "ridge")}
            for k, lab in (("G", "full G"), ("c4", "compact4")):
                rows_iii.append(dict(grid="fixed", alpha=al, target=t, cutoff=cut, X=lab, margin=m, **H.stat_ci(y, PF["b"], PF[k], g, m)))
        S.ALPHAS = DEFAULT_ALPHAS
        # which alpha does the default nested CV pick (outer-fold training sets) for B1 vs B1+G?
        for nm, X in (("B1", B1), ("B1+G", np.hstack([B1, G]))):
            chosen = []
            for c in np.unique(g):
                te = g == c; Xtr, ytr, gtr = X[~te], y[~te], g[~te]
                ug = np.unique(gtr); k = min(5, len(ug)); perm = np.random.RandomState(0).permutation(ug)
                fo = {gg_: i % k for i, gg_ in enumerate(perm)}; f = np.array([fo[gg_] for gg_ in gtr]); sse = np.zeros(len(S.ALPHAS))
                for j in range(k):
                    tr_, va_ = f != j, f == j
                    sse += ((S._ridge_all(Xtr[tr_], ytr[tr_], Xtr[va_]) - ytr[va_][:, None]) ** 2).sum(0)
                chosen.append(int(np.argmin(sse)))
            ch = np.array(chosen)
            rows_alpha.append(dict(target=t, cutoff=cut, set=nm, median_log10_alpha=float(np.median(np.log10(S.ALPHAS[ch]))), frac_at_top=float(np.mean(ch == len(S.ALPHAS) - 1)),
                                   frac_at_bottom=float(np.mean(ch == 0))))
        print(t, cut, flush=True)
write("A_conf_i_within_arch.csv", rows_i); write("A_conf_ii_lowdim.csv", rows_ii); write("A_conf_iii_alpha.csv", rows_iii); write("A_conf_iii_alpha_selected.csv", rows_alpha)
print("done")
