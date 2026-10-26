"""Amendment 6 (review 4, post hoc) shared helpers. Import FIRST. Caps BLAS threads (step4_lib then pins them to 2). CPU only.
step7_confirm.py is imported, never modified."""
import os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "8")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step7_confirm as K7          # imports step6_lib -> step5_lib -> step4_lib (set (ii), mnist)
import numpy as np
Z, M, S, D, C = K7.Z, K7.M, K7.S, K7.D, K7.C
OUT = os.path.join(C.OUT_ROOT, "ext5"); os.makedirs(OUT, exist_ok=True)
CUTS, TARGETS, NB = K7.CUTS, K7.TARGETS, K7.NB
margin = K7.margin


def load_new():
    return K7.load_runs_from(K7.new_rids())


def cut_data(runs, cut, t):
    """returns y, B1 (B0 + design5), G (18), names, g (config ids), B0"""
    y, B0, DES3, Gm, g, names = K7.get(runs, cut, t)
    B1 = np.hstack([B0, K7.design5(runs, DES3)])
    return y, B1, Gm, names, g, B0


def loco_fn(build, y, g, fp=None):
    """generic LOCO. build(tr_mask, te_mask) -> (Xtr, Xte). fp default nested-alpha ridge."""
    fp = fp or S.ridge_fit_predict
    p = np.zeros(len(y))
    for c in np.unique(g):
        te = g == c; tr = ~te
        Xtr, Xte = build(tr, te)
        p[te] = fp(Xtr, y[tr], g[tr], Xte)
    return p


def stat_ci(y, pa, pb, g, m, nb=NB, seed=0):
    """dR2 = R2(pb)-R2(pa), config bootstrap; returns dict with delta, 90%CI, one-sided p's"""
    st = M.CfgStats(y, {"a": pa, "b": pb}, g)
    W = st.draw(nb, np.random.default_rng(seed)); d = st.d_r2("a", "b", W); obs = st.d_r2("a", "b", np.ones((1, st.nc)))[0]
    q = np.percentile(d, [2.5, 97.5, 5, 95])
    return dict(r2_a=S.r2(y, pa), r2_b=S.r2(y, pb), delta=obs, lo95=q[0], hi95=q[1], lo90=q[2], hi90=q[3], n_cfg=st.nc,
                p_neg=(np.sum(d >= 0) + 1) / (nb + 1), p_pos=(np.sum(d <= 0) + 1) / (nb + 1), tost_p=M.tost_p(d, m))
