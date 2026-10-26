"""Aggregate ntk_coupling pair JSONs -> phase tables (summary.md) + figures."""
import os
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(k, "4")
import glob, json
import sys
PFX = os.environ.get('NTK_PREFIX', 'mnist_probe_none_d16_s0_dl_')
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.environ.get('NTK_OUT', '/vol/bitbucket/lbastian/latent-dynamics/dynamics_v2/ntk_coupling')
PH = {'early': (0, 50), 'mid': (50, 2000), 'late': (2000, 10**9)}
TAGS = {'adam_n1000_wd0': 'Adam N=1000', 'adam_n4000_wd0': 'Adam N=4000', 'adam_n4000_wd1e-3': 'Adam N=4000 wd1e-3',
        'adam_nfull_wd0': 'Adam N=60000', 'adam_n4000_ln0.2': 'Adam N=4000 noise0.2', 'sgd_lr0.025_n4000_wd0': 'SGD N=4000'}


def load():
    R = {}
    for tag in TAGS:
        fs = sorted(glob.glob(f'{OUT}/{PFX}{tag}/pair_*.json'))
        if fs:
            rs = sorted((json.load(open(f)) for f in fs), key=lambda r: r['step'])
            for r in rs:
                f2 = f'{OUT}/{PFX}{tag}/resid_{r["step"] - 1:07d}.json'
                if os.path.exists(f2): r['resid'] = json.load(open(f2))['resid']
            R[tag] = rs
    return R


def get(r, path):
    for p in path:
        if r is None: return np.nan
        r = r.get(p) if isinstance(r, dict) else None
    return np.nan if r is None else r


def col(rs, path): return np.array([get(r, path) for r in rs], float)


def phase_mean(rs, path, ph):
    lo, hi = PH[ph]; st = np.array([r['step'] for r in rs]); v = col(rs, path)
    m = (st > lo) & (st <= hi) & ~np.isnan(v)
    return float(np.median(v[m])) if m.any() else np.nan


def table(R, title, rows, fmt='{:.3f}'):
    """rows: list of (label, path)."""
    s = f'\n### {title}\n\n| quantity | ' + ' | '.join(f'{TAGS[t]} {p}' for t in R for p in PH) + ' |\n|---|' + '---|' * (3 * len(R)) + '\n'
    for lab, path in rows:
        s += f'| {lab} | ' + ' | '.join(fmt.format(phase_mean(R[t], path, p)) for t in R for p in PH) + ' |\n'
    return s


def main():
    R = load(); md = '# NTK coupling summary (phase MEDIANS over checkpoint pairs; early step<=50, mid 50-2000, late >2000)\n'
    md += '\n(all per-run phase means; transposed tables per run are in the pair_*.json files)\n'
    ab = lambda comp, m, s='test': ['ab', s, comp, m]
    md += table(R, '(a) linearization J_i dtheta_actual vs observed dz (test probe)',
                [('R2', ab('lin', 'r2')), ('cos', ab('lin', 'cos')), ('mean per-sample cos', ab('lin', 'cos_mean')),
                 ('R2 (train probe)', ab('lin', 'r2', 'trainprobe')), ('rms |dz| obs', ['step_info', 'obs_rms'])])
    md += table(R, '(b) attribution (test probe). energy = |component|^2/|obs|^2',
                [('energy H (momentum history)', ab('H', 'energy')), ('energy Nb+W (new batch grad)', ab('newb', 'energy')),
                 ('energy Nd+W (full-batch drift)', ab('newd', 'energy')), ('energy Nf (fluct)', ab('Nf', 'energy')),
                 ('R2 H', ab('H', 'r2')), ('R2 Nb+W', ab('newb', 'r2')), ('cos Nb+W', ab('newb', 'cos')),
                 ('R2 Nd+W', ab('newd', 'r2')), ('cos Nd+W', ab('newd', 'cos')), ('cos Nf', ab('Nf', 'cos')),
                 ('R2 H+Nd+W (no-minibatch)', ab('det', 'r2')), ('cos H+Nd+W', ab('det', 'cos')),
                 ('R2 H+Nb+W (=recon)', ab('recon', 'r2')),
                 ('R2 Hunroll vs obs', ['ab', 'test', 'Hunroll', 'r2']), ('R2 Hunroll+Nb+W', ['ab', 'test', 'Hunroll_plus_new', 'r2']),
                 ('batch grad check cos', ['batch_check', 'cos']), ('update recon rel err', ['update_recon', 'rel_err']),
                 ('replay cos (torch opt vs ckpt)', ['replay_cos'])])
    ck = lambda k, nm='pre': ['c_kernel', nm, k]
    for nm, lab in (('pre', 'preconditioned kernel J D J^T (what Adam moves with; =raw for SGD)'), ('raw', 'raw eNTK J J^T')):
        if nm == 'raw' and all('SGD' in TAGS[t] for t in R): continue
        md += table(R, f'(c) kernel structure, {lab}',
                    [('|K| within/between (Frobenius ratio)', ck('fro_ratio_wb', nm)), ('cos-normalized within', ck('cosn_within', nm)),
                     ('cos-normalized between', ck('cosn_between', nm)), ('KTA trace-kernel (centered)', ck('kta_tr_c', nm)),
                     ('KTA Frobenius-kernel (centered)', ck('kta_fn_c', nm)), ('erank diag blocks K_ii', ck('erank_diag', nm)),
                     ('erank within-class K_ij', ck('erank_within', nm)), ('erank between-class K_ij', ck('erank_between', nm)),
                     ('isotropy (energy share of trI fit) diag', ck('iso_diag', nm)), ('isotropy within', ck('iso_within', nm)),
                     ('isotropy between', ck('iso_between', nm)), ('top-1 sv share, within', ck('top1_within', nm)),
                     ('coherence |mean K_ij|/mean|K_ij| within', ck('coher_within', nm))])
    da = lambda comp, m, s='test': ['d_affine', s, comp, m]
    md += table(R, '(d) affine structure of components (test probe; energy share explained)',
                [(f'{c} {m}', da(c, m)) for c in ('obs', 'Nd', 'Nf', 'Nb') for m in ('glob', 'cls', 'cls_shuf', 'cls_share_of_resid')]
                + [('class-block diag frac of drift energy', ['d_classblock', 'diag_frac']), ('row-diag mean', ['d_classblock', 'row_diag_mean'])])
    en = lambda comp, s='test': ['e_next', s, comp, 'cos']
    md += table(R, '(e) step-to-step anti-correlation (cos with next observed step dz(t+1->t+2))',
                [('cos(obs_t, obs_t+1) test', en('obs')), ('cos(obs_t, obs_t+1) train probe', en('obs', 'trainprobe')),
                 ('cos(H_t, obs_next)', en('H')), ('cos(Nd_t, obs_next)', en('Nd')), ('cos(Nf_t, obs_next)', en('Nf')),
                 ('cos(Nb_t, obs_next)', en('Nb')), ('cos(H+Nd+W, obs_next)', en('det')),
                 ('cos obs, sample in batch_t', ['e_membership', 'cos_obs_incur']), ('cos obs, sample not in batch_t', ['e_membership', 'cos_obs_notcur']),
                 ('cos obs, sample in batch_t+1', ['e_membership', 'cos_obs_innext']), ('cos obs, not in batch_t+1', ['e_membership', 'cos_obs_notnext']),
                 ('self-term energy / Nb (batch members)', ['self', 'self_energy_over_Nb']), ('self cos with obs', ['self', 'self_cos_obs'])])
    rb = lambda comp, m: ['resid', 'test', comp, m]
    md += table(R, '(b-resid) POST-GLOBAL-AFFINE residual of dz (test probe): how much do the components explain of it',
                [('residual energy frac of |obs|^2', ['resid', 'test', 'obs_energy_frac'])]
                + [(f'cos {c}', rb(c, 'cos')) for c in ('Nb', 'newb', 'Nd', 'newd', 'Nf', 'H', 'det', 'lin')]
                + [(f'R2 {c}', rb(c, 'r2')) for c in ('newb', 'newd', 'Nf', 'det', 'lin')])
    re_ = lambda comp, s='test': ['resid', s, 'e_next', comp, 'cos']
    md += table(R, '(e-resid) anti-correlation of POST-AFFINE residual steps: cos(resid_t, resid_t+1)',
                [('cos(resid obs_t, resid obs_t+1)', re_('obs')), ('  ... train probe', re_('obs', 'trainprobe')),
                 ('cos(resid H_t, resid obs_next)', re_('H')), ('cos(resid Nd_t, resid obs_next)', re_('Nd')),
                 ('cos(resid Nf_t, resid obs_next)', re_('Nf')), ('cos(resid Nb_t, resid obs_next)', re_('Nb')),
                 ('cos(resid det_t, resid obs_next)', re_('det'))])
    open(f'{OUT}/summary.md', 'w').write(md)
    # ---- figures
    os.makedirs(f'{OUT}/figures', exist_ok=True)
    cm = plt.get_cmap('tab10')

    def fig(name, panels, ncol=3):
        n = len(panels); nr = int(np.ceil(n / ncol))
        f, axs = plt.subplots(nr, ncol, figsize=(4.6 * ncol, 3.4 * nr), squeeze=False)
        for ax, (ttl, paths, ylab) in zip(axs.flat, panels):
            for i, t in enumerate(R):
                for j, p in enumerate(paths):
                    ax.plot([r['step'] for r in R[t]], col(R[t], p), color=cm(i), ls=['-', '--', ':'][j], lw=1.3,
                            label=TAGS[t] if j == 0 else None)
            ax.set_xscale('log'); ax.set_title(ttl, fontsize=9); ax.set_ylabel(ylab, fontsize=8); ax.set_xlabel('step', fontsize=8)
            ax.grid(alpha=.3)
        axs.flat[0].legend(fontsize=6)
        for ax in axs.flat[n:]: ax.axis('off')
        f.tight_layout(); f.savefig(f'{OUT}/figures/{name}.png', dpi=130); plt.close(f)
    fig('fig_ab_attribution', [('(a) linearization R2 (solid) / cos (dashed)', [ab('lin', 'r2'), ab('lin', 'cos')], ''),
                               ('(b) energy of new-batch term (solid), drift (dashed), fluct (dotted)', [ab('newb', 'energy'), ab('newd', 'energy'), ab('Nf', 'energy')], 'energy / |obs|^2'),
                               ('(b) cos with obs: Nb+W (solid), Nd+W (dashed), Nf (dotted)', [ab('newb', 'cos'), ab('newd', 'cos'), ab('Nf', 'cos')], 'cos'),
                               ('(b) R2: H (solid), H+Nd+W (dashed)', [ab('H', 'r2'), ab('det', 'r2')], 'R2'),
                               ('(b) R2 of new-batch term (solid), drift (dashed)', [ab('newb', 'r2'), ab('newd', 'r2')], 'R2')])
    fig('fig_c_kernel', [('(c) K within/between Frobenius ratio: pre (solid), raw (dashed)', [ck('fro_ratio_wb', 'pre'), ck('fro_ratio_wb', 'raw')], ''),
                         ('(c) kernel-target alignment (centered, trace-kernel): pre (solid), raw (dashed)', [ck('kta_tr_c', 'pre'), ck('kta_tr_c', 'raw')], 'KTA'),
                         ('(c) cos-norm within (solid) / between (dashed), pre', [ck('cosn_within'), ck('cosn_between')], ''),
                         ('(c) erank of K_ij (within): pre (solid), raw (dashed)', [ck('erank_within', 'pre'), ck('erank_within', 'raw')], 'effective rank /16'),
                         ('(c) isotropy share (within): pre (solid), raw (dashed)', [ck('iso_within', 'pre'), ck('iso_within', 'raw')], 'energy share of k*I fit'),
                         ('(c) erank of K_ii: pre (solid), raw (dashed)', [ck('erank_diag', 'pre'), ck('erank_diag', 'raw')], '')])
    fig('fig_d_affine', [('(d) class-wise affine energy: obs (solid), Nd (dashed), Nf (dotted)', [da('obs', 'cls'), da('Nd', 'cls'), da('Nf', 'cls')], 'energy explained'),
                         ('(d) global affine energy: obs, Nd, Nf', [da('obs', 'glob'), da('Nd', 'glob'), da('Nf', 'glob')], ''),
                         ('(d) share of post-global resid by class-wise: obs, Nd; (dotted) shuffled-label obs', [da('obs', 'cls_share_of_resid'), da('Nd', 'cls_share_of_resid'), da('obs', 'shuf_share_of_resid')], ''),
                         ('(d) class-block diag fraction of drift', [['d_classblock', 'diag_frac']], '')])
    fig('fig_e_anticorr', [('(e) POST-AFFINE residual cos(obs_t, obs_t+1)', [re_('obs'), re_('obs', 'trainprobe')], 'cos'),
                           ('(e) resid: cos with next resid step: Nd (solid), Nf (dashed), H (dotted)', [re_('Nd'), re_('Nf'), re_('H')], 'cos'),
                           ('(e) cos(obs_t, obs_t+1)', [en('obs'), en('obs', 'trainprobe')], 'cos'),
                           ('(e) cos with next step: Nd (solid), Nf (dashed), H (dotted)', [en('Nd'), en('Nf'), en('H')], 'cos'),
                           ('(e) obs cos: sample in batch_t (solid) vs not (dashed)', [['e_membership', 'cos_obs_incur'], ['e_membership', 'cos_obs_notcur']], 'cos')])
    print(md[:3000])


if __name__ == '__main__':
    main()
