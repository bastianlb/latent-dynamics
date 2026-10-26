import os
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(k, "4")
import glob, json
PFX = os.environ.get('NTK_PREFIX', 'mnist_probe_none_d16_s0_dl_')
import numpy as np
from scipy.stats import spearmanr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = '/vol/bitbucket/lbastian/latent-dynamics/dynamics_v2'
OUT = os.environ.get('NTK_OUT', f'{BASE}/ntk_coupling/traintest'); DL = f'{BASE}/dense_latent'
TAGS = {'adam_n1000_wd0': 'Adam N=1000', 'adam_n4000_wd0': 'Adam N=4000', 'adam_n4000_wd1e-3': 'Adam N=4000 wd1e-3',
        'adam_nfull_wd0': 'Adam N=60000', 'adam_n4000_ln0.2': 'Adam N=4000 noise0.2', 'sgd_lr0.025_n4000_wd0': 'SGD N=4000'}
PH = {'early': (0, 50), 'mid': (50, 2000), 'late': (2000, 10**9)}
SPE = {'adam_n1000_wd0': 1000 / 128, 'adam_nfull_wd0': 60000 / 128}


WANT = set()


def lossgap(tag):
    rid = f'{PFX}{tag}'; d = f'{DL}/{rid}'; ra = f'{DL}/_run_artifacts/runs/{rid}'
    yte = np.load(f'{ra}/probe_index.npz')['labels']; ytr_true = np.load(f'{ra}/probe_train_index.npz')['labels']
    out = {}
    for f in sorted(glob.glob(f'{d}/chunk_*.npz')):
        z0 = np.load(f); z = {k: z0[k] for k in ('steps', 'head_w', 'head_b', 'latents_test', 'latents_train', 'train_probe_loss')}
        for i, s in enumerate(z['steps']):
            if int(s) not in WANT: continue
            W, b = z['head_w'][i].astype(np.float64), z['head_b'][i].astype(np.float64)
            def ce(lat, y):
                lg = lat.astype(np.float64) @ W.T + b; lg -= lg.max(1, keepdims=True)
                return float((np.log(np.exp(lg).sum(1)) - lg[np.arange(len(y)), y]).mean())
            out[int(s)] = dict(test=ce(z['latents_test'][i], yte), train_given=float(z['train_probe_loss'][i].mean()),
                               train_true=ce(z['latents_train'][i], ytr_true))
    return out


def load():
    R = {}
    for t in TAGS:
        fs = sorted(glob.glob(f'{OUT}/{PFX}{t}/tt_*.json'))
        if fs: R[t] = sorted((json.load(open(f)) for f in fs), key=lambda r: r['step'])
    return R


def ph_idx(rs, ph):
    lo, hi = PH[ph]; return [i for i, r in enumerate(rs) if lo < r['step'] <= hi]


def kern(r, k, a, b):
    return r['kernel'][k][a][b]


def med(rs, ph, fn):
    v = []
    for i in ph_idx(rs, ph):
        try: v.append(fn(rs[i]))
        except KeyError: pass
    v = [x for x in v if x is not None and not np.isnan(x)]
    return float(np.median(v)) if v else np.nan


def pooled(rs, ph, fn_num, fn_den):
    n = d = 0.0
    for i in ph_idx(rs, ph):
        a, b = fn_num(rs[i]), fn_den(rs[i])
        if a is None or b is None: continue
        n += a; d += b
    return n / d if d > 0 else np.nan


def tab(R, title, rows, fmt='{:.3f}'):
    s = f'\n### {title}\n\n| quantity | ' + ' | '.join(f'{TAGS[t]} {p}' for t in R for p in PH) + ' |\n|---|' + '---|' * (3 * len(R)) + '\n'
    for lab, f in rows:
        s += f'| {lab} | ' + ' | '.join(fmt.format(f(R[t], p)) for t in R for p in PH) + ' |\n'
    return s


def main():
    R = load(); WANT.update(r['step'] for rs in R.values() for r in rs); GAP = {t: lossgap(t) for t in R}
    md = '# Train vs test NTK contrast (phase medians / pooled sums; early<=50, mid 50-2000, late>2000)\n'
    kk = lambda t: 'raw' if t.startswith('sgd') else 'pre'
    for kname in ('pre', 'raw'):
        rows = []
        for rel in ('within', 'between'):
            for ty in ('trtr', 'trte', 'tete'):
                rows.append((f'cosn {ty} {rel}', lambda rs, p, ty=ty, rel=rel, kname=kname: med(rs, p, lambda r: r['kernel'][kname if not rs[0]['run'].count('sgd') else 'raw'][f'{ty}_{rel}']['cosn'])))
        for rel in ('within', 'between'):
            rows.append((f'DIVERGENCE cosn(trtr)-cosn(trte) {rel}', lambda rs, p, rel=rel, kname=kname: med(rs, p, lambda r: (lambda K: K[f'trtr_{rel}']['cosn'] - K[f'trte_{rel}']['cosn'])(r['kernel'][kname if 'sgd' not in r['run'] else 'raw']))))
            rows.append((f'Fro ratio trtr/trte {rel}', lambda rs, p, rel=rel, kname=kname: med(rs, p, lambda r: (lambda K: K[f'trtr_{rel}']['fro'] / K[f'trte_{rel}']['fro'])(r['kernel'][kname if 'sgd' not in r['run'] else 'raw']))))
            rows.append((f'Fro ratio tete/trte {rel}', lambda rs, p, rel=rel, kname=kname: med(rs, p, lambda r: (lambda K: K[f'tete_{rel}']['fro'] / K[f'trte_{rel}']['fro'])(r['kernel'][kname if 'sgd' not in r['run'] else 'raw']))))
        rows += [('diag |K_ii| train/test', lambda rs, p, kname=kname: med(rs, p, lambda r: (lambda K: K['diag']['train_fro'] / K['diag']['test_fro'])(r['kernel'][kname if 'sgd' not in r['run'] else 'raw']))),
                 ('KTA train (given labels)', lambda rs, p, kname=kname: med(rs, p, lambda r: r['kernel'][kname if 'sgd' not in r['run'] else 'raw']['kta_train_given'])),
                 ('KTA test', lambda rs, p, kname=kname: med(rs, p, lambda r: r['kernel'][kname if 'sgd' not in r['run'] else 'raw']['kta_test']))]
        md += tab(R, f'(1) kernel blocks, {"preconditioned J D J^T (SGD columns: raw)" if kname == "pre" else "raw eNTK"}', rows)
    # drive
    def dfrac(g, part, what):
        def num(r):
            x = r['drive'][g]
            if x is None: return None
            return x[part]['p2'] if what == 'e_obs' else x[part]['p2']
        den = (lambda r: None if r['drive'][g] is None else r['drive'][g]['obs2']) if what == 'e_obs' else (lambda r: None if r['drive'][g] is None else r['drive'][g]['Nb2'])
        return lambda rs, p: pooled(rs, p, num, den)

    def dcos(g, part, pre=''):
        def f(rs, p):
            n = a = b = 0.0
            for i in ph_idx(rs, p):
                x = rs[i]['drive'][g]
                if x is None or (pre + part) not in x: continue
                n += x[pre + part]['p_obs']; a += x[pre + part]['p2']; b += x[pre + 'obs2'] if pre else x['obs2']
            return n / np.sqrt(a * b) if a > 0 and b > 0 else np.nan
        return f
    rows = []
    for g in ('test', 'train_out', 'train_in'):
        for part in ('self', 'same', 'other', 'Nb'):
            if part == 'self' and g != 'train_in': continue
            rows.append((f'{g}: energy {part}/|obs|^2', dfrac(g, part, 'e_obs')))
        for part in ('self', 'same', 'other'):
            if part == 'self' and g != 'train_in': continue
            rows.append((f'{g}: energy {part}/|Nb|^2', dfrac(g, part, 'e_Nb')))
        for part in ('self', 'same', 'other', 'Nb'):
            if part == 'self' and g != 'train_in': continue
            rows.append((f'{g}: cos({part}, obs)', dcos(g, part)))
        rows.append((f'{g}: n points/pair', lambda rs, p, g=g: med(rs, p, lambda r: None if r['drive'][g] is None else r['drive'][g]['n'])))
    md += tab(R, '(2) drive decomposition of N_batch (pooled over pairs in phase). same/other = batch samples of the same/other class as the probe point (train: training label); same excludes self', rows)
    rows = []
    for g in ('test', 'train_out', 'train_in'):
        for part in ('same', 'other', 'Nb'):
            def f(rs, p, g=g, part=part):
                n = a = b = 0.0
                for i in ph_idx(rs, p):
                    x = rs[i]['drive'][g]
                    if x is None or 'c_' + part not in x: continue
                    n += x['c_' + part]['p_obs']; a += x['c_' + part]['p2']; b += x['c_obs2']
                return n / np.sqrt(a * b) if a > 0 and b > 0 else np.nan
            rows.append((f'{g}: cos(centered {part}, centered obs)', f))
    md += tab(R, '(2b) same, after removing the group-mean displacement (non-uniform part)', rows)
    # noise run
    tn = 'adam_n4000_ln0.2'
    if tn in R:
        rs = [r for r in R[tn] if 'kernel' in r]; spe = 4000 / 128
        md += '\n### (3) label-noise run: mislabelled train pt -> test pts, mean cosine-normalised / Frobenius kernel (preconditioned)\n\n| step | epoch | cosn given-class | cosn true-class | cosn other | fro given | fro true | fro other | train-train: same given | given==my true | clean: own/other cosn |\n|---|---|---|---|---|---|---|---|---|---|---|\n'
        for r in rs:
            K = r['kernel']['pre']; m = K['mislabelled']; c = K['clean_train']
            md += f"| {r['step']} | {r['step'] / spe:.1f} | {m['given']['cosn']:.3f} | {m['true']['cosn']:.3f} | {m['other']['cosn']:.3f} | {m['given']['fro']:.0f} | {m['true']['fro']:.0f} | {m['other']['fro']:.0f} | {m['trtr_same_given']:.3f} | {m['trtr_given_eq_myTrue']:.3f} | {c['true']['cosn']:.3f}/{c['other']['cosn']:.3f} |\n"
        def win(lo, hi, key):
            v = [r['kernel']['pre']['mislabelled'][key]['cosn'] for r in rs if lo < r['step'] / spe <= hi]; return np.mean(v) if v else np.nan
        md += '\nWindow means of cosn(mislabelled, test of GIVEN class) / (TRUE class) / other: ' + '; '.join(
            f'{nm}: {win(lo, hi, "given"):.3f} / {win(lo, hi, "true"):.3f} / {win(lo, hi, "other"):.3f}' for nm, lo, hi in (('<2 ep', 0, 2), ('2-15 ep', 2, 15), ('>15 ep', 15, 1e9))) + '\n'
    # (4) gap vs divergence
    md += '\n### (4) train-test loss gap vs kernel divergence\n\n| run | phase | test loss | train loss (given) | gap | cosn divergence within | Fro ratio trtr/trte within |\n|---|---|---|---|---|---|---|\n'
    corr = {}
    for t, rs in R.items():
        rs = [r for r in rs if 'kernel' in r]
        if len(rs) < 3: continue
        kn = 'raw' if t.startswith('sgd') else 'pre'
        steps = np.array([r['step'] for r in rs])
        te = np.array([GAP[t][s]['test'] for s in steps]); tr = np.array([GAP[t][s]['train_given'] for s in steps])
        div = np.array([r['kernel'][kn]['trtr_within']['cosn'] - r['kernel'][kn]['trte_within']['cosn'] for r in rs])
        fr = np.array([r['kernel'][kn]['trtr_within']['fro'] / r['kernel'][kn]['trte_within']['fro'] for r in rs])
        gap = te - tr
        for p, (lo, hi) in PH.items():
            m = (steps > lo) & (steps <= hi)
            md += f'| {TAGS[t]} | {p} | {np.median(te[m]):.3f} | {np.median(tr[m]):.4f} | {np.median(gap[m]):.3f} | {np.median(div[m]):.4f} | {np.median(fr[m]):.4f} |\n'
        sel = steps >= 10
        c1 = spearmanr(gap[sel], div[sel]); c2 = spearmanr(gap[sel], fr[sel]); c3 = spearmanr(steps[sel], gap[sel])
        corr[t] = (c1[0], c2[0], c3[0], c1[1])
    md += '\nSpearman over checkpoints with step>=10: rho(gap, cosn divergence), rho(gap, Fro ratio), rho(step, gap) [step-trend baseline], p(gap,div)\n\n'
    for t, c in corr.items(): md += f'- {TAGS[t]}: {c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f}, p={c[3]:.2g}\n'
    open(f'{OUT}/summary_traintest.md', 'w').write(md)
    # figures
    os.makedirs(f'{OUT}/figures', exist_ok=True); cm = plt.get_cmap('tab10')
    f, axs = plt.subplots(2, 3, figsize=(15, 7))
    for i, (t, rs) in enumerate(R.items()):
        rs = [r for r in rs if 'kernel' in r]
        if len(rs) < 3: continue
        kn = 'raw' if t.startswith('sgd') else 'pre'; st = [r['step'] for r in rs]
        K = lambda a, b: np.array([r['kernel'][kn][a][b] for r in rs])
        axs[0, 0].plot(st, K('trtr_within', 'cosn') - K('trte_within', 'cosn'), color=cm(i), label=TAGS[t])
        axs[0, 1].plot(st, K('trtr_within', 'fro') / K('trte_within', 'fro'), color=cm(i))
        axs[0, 2].plot(st, K('trtr_between', 'cosn') - K('trte_between', 'cosn'), color=cm(i))
        g = np.array([GAP[t][s]['test'] - GAP[t][s]['train_given'] for s in st]); axs[1, 0].plot(st, g, color=cm(i))
        div = K('trtr_within', 'cosn') - K('trte_within', 'cosn'); m = np.array(st) >= 10
        axs[1, 1].scatter(g[m], div[m], color=cm(i), s=12)
        axs[1, 2].plot(st, [r['kernel'][kn]['kta_train_given'] - r['kernel'][kn]['kta_test'] for r in rs], color=cm(i))
    for a, tt in zip(axs.flat, ['cosn(train-train) - cosn(train-test), within-class', 'Fro ratio train-train / train-test, within', 'same, between-class', 'test - train loss (gap)', 'divergence (within) vs loss gap', 'KTA(train) - KTA(test)']):
        a.set_title(tt, fontsize=9); a.grid(alpha=.3)
        if a is not axs[1, 1]: a.set_xscale('log'); a.set_xlabel('step')
    axs[1, 1].set_xscale('log'); axs[1, 1].set_xlabel('loss gap'); axs[0, 0].legend(fontsize=6)
    f.tight_layout(); f.savefig(f'{OUT}/figures/fig_kernel_divergence_gap.png', dpi=130); plt.close(f)
    if tn in R:
        rs = [r for r in R[tn] if 'kernel' in r]; spe = 4000 / 128; ep = np.array([r['step'] / spe for r in rs])
        f, ax = plt.subplots(1, 2, figsize=(11, 3.8))
        for key, ls in (('given', '-'), ('true', '--'), ('other', ':')):
            ax[0].plot(ep, [r['kernel']['pre']['mislabelled'][key]['cosn'] for r in rs], ls, label=key + ' class (test pts)')
            ax[1].plot(ep, [r['kernel']['pre']['mislabelled'][key]['fro'] for r in rs], ls)
        for a in ax: a.axvspan(2, 15, color='gray', alpha=.15); a.set_xscale('log'); a.set_xlabel('epoch'); a.grid(alpha=.3)
        ax[0].set_title('mislabelled train pt -> test pts: cos-normalised kernel'); ax[1].set_title('Frobenius'); ax[0].legend(fontsize=7)
        f.tight_layout(); f.savefig(f'{OUT}/figures/fig_noise_given_vs_true.png', dpi=130); plt.close(f)
    # drive figure
    f, axs = plt.subplots(1, len(R), figsize=(3.4 * len(R), 3.6), squeeze=False)
    for a, (t, rs) in zip(axs[0], R.items()):
        xs = []; labs = []
        for gi, g in enumerate(('test', 'train_out', 'train_in')):
            for pi, p in enumerate(PH):
                e = {part: dfrac(g, part, 'e_Nb')(rs, p) for part in ('self', 'same', 'other')}
                x = gi * 4 + pi; b = 0
                for part, c in (('self', 'C3'), ('same', 'C0'), ('other', 'C1')):
                    v = e[part] if e[part] == e[part] else 0
                    a.bar(x, v, bottom=b, color=c, label=part if (gi == 0 and pi == 0) else None); b += v
                labs.append((x, f'{g[:6]}\n{p[:1]}'))
        a.set_xticks([x for x, _ in labs]); a.set_xticklabels([l for _, l in labs], fontsize=6); a.set_title(TAGS[t], fontsize=8)
        a.set_ylabel('energy / |Nb|^2 (parts not additive: cross terms)', fontsize=6)
    axs[0, 0].legend(fontsize=6); f.tight_layout(); f.savefig(f'{OUT}/figures/fig_drive_decomposition.png', dpi=130); plt.close(f)
    print(md)


if __name__ == '__main__':
    main()
