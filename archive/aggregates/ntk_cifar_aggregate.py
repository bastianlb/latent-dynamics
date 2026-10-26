import os
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(k, "4")
import glob, json
import numpy as np
from scipy.stats import spearmanr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = '/vol/bitbucket/lbastian/latent-dynamics/dynamics_v2'; OUT = f'{BASE}/ntk_coupling/traintest/cifar'; DL = f'{BASE}/dense_latent'
RUNS = {'cifar10_probe_none_d512_s0_dl_rn18_n10000_ln0': 'CIFAR ln0', 'cifar10_probe_none_d512_s0_dl_rn18_n10000_ln0.2': 'CIFAR ln0.2'}
PH = {'early': (0, 50), 'mid': (50, 1500), 'late': (1500, 10**9)}


def load():
    R = {}
    for rid in RUNS:
        fs = sorted(glob.glob(f'{OUT}/{rid}/cf_*.json'))
        if fs: R[rid] = sorted((json.load(open(f)) for f in fs), key=lambda r: r['step'])
    return R


def idx(rs, p): lo, hi = PH[p]; return [r for r in rs if lo < r['step'] <= hi]


def med(rs, p, fn):
    v = []
    for r in idx(rs, p):
        try: v.append(fn(r))
        except (KeyError, TypeError): pass
    v = [x for x in v if x is not None and np.isfinite(x)]
    return float(np.median(v)) if v else np.nan


def pooled(rs, p, g, part, den):
    n = d = 0.0
    for r in idx(rs, p):
        x = r['drive'].get(g)
        if x is None: continue
        n += x[part]['p2']; d += x[den]
    return n / d if d > 0 else np.nan


def lossgap(rid, steps):
    d = f'{DL}/{rid}'; ra = f'{d}/../_run_artifacts/runs/{rid}'
    yte = np.load(f'{ra}/probe_index.npz')['labels']; out = {}
    for f in sorted(glob.glob(f'{d}/chunk_*.npz')):
        z = np.load(f); st = z['steps']; want = [i for i, s in enumerate(st) if int(s) in steps]
        if not want: continue
        LT = z['latents_test']; HW = z['head_w']; HB = z['head_b']; TL = z['train_probe_loss']
        for i in want:
            lg = LT[i].astype(np.float64) @ HW[i].astype(np.float64).T + HB[i]; lg -= lg.max(1, keepdims=True)
            out[int(st[i])] = (float((np.log(np.exp(lg).sum(1)) - lg[np.arange(len(yte)), yte]).mean()), float(TL[i].mean()))
    return out


def main():
    R = load(); md = '# CIFAR-10 ResNet-18 NTK coupling + train/test contrast (PCA16 frame; phase medians; early<=50, mid 50-1500, late>1500)\n'
    cols = [(rid, p) for rid in R for p in PH]
    def tab(title, rows, fmt='{:.3f}'):
        s = f'\n### {title}\n\n| quantity | ' + ' | '.join(f'{RUNS[r]} {p}' for r, p in cols) + ' |\n|---|' + '---|' * len(cols) + '\n'
        for lab, fn in rows: s += f'| {lab} | ' + ' | '.join(fmt.format(fn(R[r], p)) for r, p in cols) + ' |\n'
        return s
    A = lambda c, m, s='test': (lambda r: r['ab'][s][c][m])
    md += tab('(a)+(b) linearisation and attribution (test probes)', [
        ('lin R2', lambda rs, p: med(rs, p, A('lin', 'r2'))), ('lin cos', lambda rs, p: med(rs, p, A('lin', 'cos'))),
        ('lin R2 train probes', lambda rs, p: med(rs, p, A('lin', 'r2', 'trainprobe'))),
        ('energy H', lambda rs, p: med(rs, p, A('H', 'energy'))), ('energy Nb', lambda rs, p: med(rs, p, A('Nb', 'energy'))),
        ('energy W (weight decay)', lambda rs, p: med(rs, p, A('W', 'energy'))), ('energy Bbuf (BN stats)', lambda rs, p: med(rs, p, A('Bbuf', 'energy'))),
        ('R2 H', lambda rs, p: med(rs, p, A('H', 'r2'))), ('R2 Nb', lambda rs, p: med(rs, p, A('Nb', 'r2'))), ('cos Nb', lambda rs, p: med(rs, p, A('Nb', 'cos'))),
        ('energy Nd (drift est.)', lambda rs, p: med(rs, p, A('Nd', 'energy'))), ('R2 Nd', lambda rs, p: med(rs, p, A('Nd', 'r2'))), ('cos Nd', lambda rs, p: med(rs, p, A('Nd', 'cos'))),
        ('energy Nf', lambda rs, p: med(rs, p, A('Nf', 'energy'))), ('cos Nf', lambda rs, p: med(rs, p, A('Nf', 'cos'))),
        ('R2 H+Nd+W', lambda rs, p: med(rs, p, A('det', 'r2'))),
        ('update recon rel err', lambda rs, p: med(rs, p, lambda r: r['update_recon_relerr'])),
        ('proxy-batch-grad cos vs true g_b', lambda rs, p: med(rs, p, lambda r: r['proxy_cos_batchgrad'])),
        ('proxy-batch push cos vs true Nb push (z)', lambda rs, p: med(rs, p, lambda r: r['proxy_vs_true_Nb']['cos'])),
        ('obs rms', lambda rs, p: med(rs, p, lambda r: r['step_info']['obs_rms']))])
    D = lambda g, k: (lambda r: r['d_affine'][g][k])
    md += tab('(d) affine structure (test probes, PCA16 frame)', [
        ('obs glob', lambda rs, p: med(rs, p, lambda r: r['d_affine']['test']['obs']['glob'])), ('obs cls', lambda rs, p: med(rs, p, lambda r: r['d_affine']['test']['obs']['cls'])),
        ('obs cls_shuf', lambda rs, p: med(rs, p, lambda r: r['d_affine']['test']['obs']['cls_shuf'])), ('obs cls_share_of_resid', lambda rs, p: med(rs, p, lambda r: r['d_affine']['test']['obs']['cls_share_of_resid'])),
        ('Nd glob', lambda rs, p: med(rs, p, lambda r: r['d_affine']['test']['Nd']['glob'])), ('Nd cls_share_of_resid', lambda rs, p: med(rs, p, lambda r: r['d_affine']['test']['Nd']['cls_share_of_resid']))])
    KS = lambda k: (lambda r: r['kernel_test_stats'][k])
    KB = lambda a, b: (lambda r: r['kernel'][a][b])
    md += tab('(c) kernel structure (test points, raw eNTK of PCA16 outputs, count-sketch Gram)', [
        ('Fro within/between', lambda rs, p: med(rs, p, KS('fro_ratio_wb'))), ('cosn within', lambda rs, p: med(rs, p, KS('cosn_within'))), ('cosn between', lambda rs, p: med(rs, p, KS('cosn_between'))),
        ('KTA trace (centered)', lambda rs, p: med(rs, p, KS('kta_tr_c'))), ('erank diag blocks', lambda rs, p: med(rs, p, KS('erank_diag'))), ('erank within', lambda rs, p: med(rs, p, KS('erank_within'))),
        ('isotropy within', lambda rs, p: med(rs, p, KS('iso_within'))), ('isotropy diag', lambda rs, p: med(rs, p, KS('iso_diag')))])
    dv = lambda rel: (lambda r: r['kernel']['trtr_' + rel]['cosn'] - r['kernel']['trte_' + rel]['cosn'])
    fr = lambda rel: (lambda r: r['kernel']['trtr_' + rel]['fro'] / r['kernel']['trte_' + rel]['fro'])
    md += tab('(2.1) train vs test kernel blocks', [
        ('cosn trtr within', lambda rs, p: med(rs, p, KB('trtr_within', 'cosn'))), ('cosn trte within', lambda rs, p: med(rs, p, KB('trte_within', 'cosn'))),
        ('cosn tete within', lambda rs, p: med(rs, p, KB('tete_within', 'cosn'))), ('cosn trtr between', lambda rs, p: med(rs, p, KB('trtr_between', 'cosn'))),
        ('cosn trte between', lambda rs, p: med(rs, p, KB('trte_between', 'cosn'))),
        ('DIVERGENCE cosn(trtr)-cosn(trte) within', lambda rs, p: med(rs, p, dv('within'))), ('DIVERGENCE between', lambda rs, p: med(rs, p, dv('between'))),
        ('Fro ratio trtr/trte within', lambda rs, p: med(rs, p, fr('within'))), ('Fro ratio tete/trte within', lambda rs, p: med(rs, p, lambda r: r['kernel']['tete_within']['fro'] / r['kernel']['trte_within']['fro'])),
        ('diag train/test', lambda rs, p: med(rs, p, lambda r: r['kernel']['diag']['train_fro'] / r['kernel']['diag']['test_fro'])),
        ('KTA train (given)', lambda rs, p: med(rs, p, lambda r: r['kernel']['kta_train_given'])), ('KTA test', lambda rs, p: med(rs, p, lambda r: r['kernel']['kta_test']))])
    rows = []
    for g in ('test', 'train_out', 'train_in'):
        for part in ('self', 'same', 'other', 'proxy_tot'):
            if part == 'self' and g != 'train_in': continue
            rows.append((f'{g}: energy {part}/|Nb|^2 (Nb = exact)', lambda rs, p, g=g, part=part: pooled(rs, p, g, part, 'Nb2')))
        rows.append((f'{g}: n/pair', lambda rs, p, g=g: med(rs, p, lambda r: r['drive'][g]['n'])))
    md += tab('(2.2) drive decomposition; self/same/other are PROXY gradients (fresh augmentation); Nb exact', rows)
    # mislabelled (ln0.2)
    for rid in R:
        if 'ln0.2' not in rid: continue
        md += '\n### (2.3) mislabelled train pt -> test pts (raw eNTK, PCA16): cosn given / true / other; Fro given / true / other\n\n| step | epoch | cosn given | cosn true | cosn other | fro given | fro true | fro other |\n|---|---|---|---|---|---|---|---|\n'
        for r in R[rid]:
            m = r['kernel']['mislabelled']; md += f"| {r['step']} | {r['step'] / 79:.2f} | {m['given']['cosn']:.3f} | {m['true']['cosn']:.3f} | {m['other']['cosn']:.3f} | {m['given']['fro']:.3g} | {m['true']['fro']:.3g} | {m['other']['fro']:.3g} |\n"
    # gap
    md += '\n### (2.4) loss gap vs divergence\n\n| run | phase | test loss | train loss | gap | cosn divergence within | Fro ratio |\n|---|---|---|---|---|---|---|\n'; corr = []
    for rid, rs in R.items():
        GP = lossgap(rid, {r['step'] for r in rs})
        for p in PH:
            sub = idx(rs, p)
            if not sub: continue
            te = [GP[r['step']][0] for r in sub]; tr = [GP[r['step']][1] for r in sub]
            md += f"| {RUNS[rid]} | {p} | {np.median(te):.3f} | {np.median(tr):.3f} | {np.median(np.array(te) - np.array(tr)):.3f} | {np.median([dv('within')(r) for r in sub]):.4f} | {np.median([fr('within')(r) for r in sub]):.4f} |\n"
        sel = [r for r in rs if r['step'] >= 5]
        gap = np.array([GP[r['step']][0] - GP[r['step']][1] for r in sel]); d_ = np.array([dv('within')(r) for r in sel]); f_ = np.array([fr('within')(r) for r in sel]); st = np.array([r['step'] for r in sel])
        corr.append((RUNS[rid], spearmanr(gap, d_)[0], spearmanr(gap, f_)[0], spearmanr(st, gap)[0], len(sel)))
    md += '\nSpearman (steps>=5): rho(gap, cosn divergence), rho(gap, Fro ratio), rho(step, gap), n\n\n' + ''.join(f'- {c[0]}: {c[1]:.2f}, {c[2]:.2f}, {c[3]:.2f}, n={c[4]}\n' for c in corr)
    open(f'{OUT}/summary_cifar.md', 'w').write(md); print(md)


if __name__ == '__main__':
    main()
