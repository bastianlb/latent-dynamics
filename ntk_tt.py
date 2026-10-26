"""Train-vs-test tangent-kernel contrast (kernel blocks on a mixed set, drive decomposition, label-noise coupling).
Reuses ntk_lib / ntk_pair / run_ntk. usage: run_tt.py <run_id>"""
import os, sys, json, time
import numpy as np, torch, torch.nn.functional as F
import ntk_lib as L, ntk_pair as Pm, run_ntk as R

OUT = os.environ.get('NTK_OUT', '/vol/bitbucket/lbastian/latent-dynamics/dynamics_v2/ntk_coupling/traintest')


def select_set(ctx, noisy, per_class=int(os.environ.get('NTK_KPC', 40))):
    """indices into Xp (test rows 0..nt-1, train-probe rows nt..): returns (idx, is_train bool, true, given)."""
    nt = ctx['ntest']; yp = ctx['yp'].cpu().numpy(); given_tp = ctx['lab_full'][torch.as_tensor(ctx['tp_idx']).to(L.DEV)].cpu().numpy()
    true_tr = yp[nt:]
    te = np.concatenate([np.nonzero(yp[:nt] == c)[0][:per_class] for c in range(10)])
    if noisy:  # 200 mislabelled + 200 clean train probes
        mis = np.nonzero(given_tp != true_tr)[0]; cl = np.nonzero(given_tp == true_tr)[0]
        tr = np.concatenate([mis[:5 * per_class], cl[:5 * per_class]])
    else:
        tr = np.concatenate([np.nonzero(true_tr == c)[0][:per_class] for c in range(10)])
    idx = np.concatenate([te, nt + tr])
    is_tr = np.concatenate([np.zeros(len(te), bool), np.ones(len(tr), bool)])
    true = yp[idx]; given = np.concatenate([yp[te], given_tp[tr]])
    return idx, is_tr, true, given


def kernel_blocks(G, is_tr, true, given, gen):
    n = len(true); dev = G.device
    Gr = G.view(n, 16, n, 16).double()
    Fro = (Gr ** 2).sum((1, 3)).sqrt(); Tr = torch.einsum('ikjk->ij', Gr)
    dg = Fro.diag(); Fn = Fro / (dg[:, None] * dg[None, :]).sqrt()
    tr = torch.as_tensor(is_tr).to(dev); y = torch.as_tensor(true).to(dev); yg = torch.as_tensor(given).to(dev)
    eye = torch.eye(n, dtype=torch.bool, device=dev)
    same = y[:, None] == y[None, :]
    out = {}
    for nm, (ri, ci) in dict(trtr=(tr, tr), tete=(~tr, ~tr), trte=(tr, ~tr)).items():
        pm = ri[:, None] & ci[None, :] & ~eye
        for rel, m in (('within', same), ('between', ~same)):
            mm = pm & m
            out[f'{nm}_{rel}'] = dict(fro=float(Fro[mm].mean()), cosn=float(Fn[mm].mean()), tr=float(Tr[mm].mean()))
    out['diag'] = dict(train_fro=float(dg[tr].mean()), test_fro=float(dg[~tr].mean()))
    # class x class matrices (true class) of cosine-normalised kernel, for each set pair
    oh = F.one_hot(y, 10).double()
    for nm, (ri, ci) in dict(trtr=(tr, tr), tete=(~tr, ~tr), trte=(tr, ~tr)).items():
        W = (ri[:, None] & ci[None, :] & ~eye).double()
        out[f'cm_{nm}'] = ((oh.T @ (Fn * W) @ oh) / (oh.T @ W @ oh).clamp_min(1)).cpu().numpy().tolist()
    # kernel-target alignment (centered, trace kernel) within train-only and test-only
    def kta(mask, lab):
        S = Tr[mask][:, mask]; S = 0.5 * (S + S.T); m = S.shape[0]
        Y = F.one_hot(lab[mask], 10).double(); Lm = Y @ Y.T
        H = torch.eye(m, device=dev, dtype=torch.double) - 1.0 / m
        S, Lm = H @ S @ H, H @ Lm @ H
        return float((S * Lm).sum() / (S.norm() * Lm.norm()))
    out['kta_train_given'] = kta(tr, yg); out['kta_test'] = kta(~tr, y); out['kta_train_true'] = kta(tr, y)
    # mislabelled analysis (noise run): per train point i, mean over test j of Fn/Fro by class relation
    mis = tr & (yg != y)
    if mis.any():
        def rel_stats(rows, nm):
            te = ~tr
            res = {}
            for key, cond in (('given', lambda i: te & (y == yg[i])), ('true', lambda i: te & (y == y[i])),
                              ('other', lambda i: te & (y != yg[i]) & (y != y[i]))):
                a, b = [], []
                for i in rows.nonzero().flatten().tolist():
                    mk = cond(i)
                    if mk.any():
                        a.append(Fn[i][mk].mean()); b.append(Fro[i][mk].mean())
                res[key] = dict(cosn=float(torch.stack(a).mean()), fro=float(torch.stack(b).mean()))
            # train-train by given labels
            a2, a3 = [], []
            for i in rows.nonzero().flatten().tolist():
                mk1 = tr & (yg == yg[i]) & ~eye[i]; mk2 = tr & (yg == y[i]) & (y != yg[i]) & ~eye[i]
                if mk1.any(): a2.append(Fn[i][mk1].mean())
                if mk2.any(): a3.append(Fn[i][mk2].mean())
            res['trtr_same_given'] = float(torch.stack(a2).mean()) if a2 else None
            res['trtr_given_eq_myTrue'] = float(torch.stack(a3).mean()) if a3 else None
            return res
        out['mislabelled'] = rel_stats(mis, 'mis'); out['clean_train'] = rel_stats(tr & ~mis, 'clean')
        out['n_mis'] = int(mis.sum())
    return out


def drive(P, ctx, d, batch_idx, obs, dz_nb, z0):
    """Split N_batch into self / same-class batch / other-class batch for each probe point; group sums."""
    nt = ctx['ntest']; Xp = ctx['Xp']; B = len(batch_idx)
    bi = torch.as_tensor(batch_idx).to(L.DEV); Xb, yb = ctx['Xfull'][bi], ctx['lab_full'][bi]
    kn, Dp = d['kn'], d['Dp']
    mk = lambda g: L.scale(L.mul(Dp, g), -kn)
    dzc = []
    for c in range(10):
        m = yb == c
        dzc.append(L.jvp_z(P, Xp, mk(L.grad_enc(P, Xb[m], yb[m], B))) if m.any() else torch.zeros_like(obs))
    dzc = torch.stack(dzc)  # [10,n,16]
    tot = dzc.sum(0)
    # class of each probe point for "same/other": test -> true class; train -> given (training) label
    tp_given = ctx['lab_full'][torch.as_tensor(ctx['tp_idx']).to(L.DEV)]
    cls = torch.cat([ctx['yp'][:nt], tp_given])
    same = dzc[cls, torch.arange(len(cls))]
    other = tot - same
    inb_tr = torch.as_tensor(np.isin(ctx['tp_idx'], batch_idx)).to(L.DEV)
    selfz = torch.zeros_like(obs)
    for j in inb_tr.nonzero().flatten().tolist():
        fi = int(ctx['tp_idx'][j])
        gi = L.grad_enc(P, ctx['Xfull'][fi][None], ctx['lab_full'][fi][None], B)
        selfz[nt + j] = L.jvp_z(P, Xp[nt + j][None], mk(gi))[0]
    same_ex = same - selfz
    inb = torch.cat([torch.zeros(nt, dtype=torch.bool, device=L.DEV), inb_tr])
    istr = torch.cat([torch.zeros(nt, dtype=torch.bool, device=L.DEV), torch.ones(len(inb_tr), dtype=torch.bool, device=L.DEV)])
    groups = dict(test=~istr, train_out=istr & ~inb, train_in=inb)
    parts = dict(self=selfz, same=same_ex, other=other, Nb=tot)
    res = {}
    for gname, gm in groups.items():
        n = int(gm.sum())
        if n == 0: res[gname] = None; continue
        o = obs[gm].double(); nb = tot[gm].double()
        r = dict(n=n, obs2=float((o ** 2).sum()), Nb2=float((nb ** 2).sum()), Nb_obs=float((nb * o).sum()))
        for pn, pv in parts.items():
            p = pv[gm].double()
            r[pn] = dict(p2=float((p ** 2).sum()), p_obs=float((p * o).sum()), p_Nb=float((p * nb).sum()))
        # centered (group-mean removed) variants: non-uniform component
        if n >= 10:
            co = o - o.mean(0); cn = nb - nb.mean(0)
            r['c_obs2'] = float((co ** 2).sum()); r['c_Nb2'] = float((cn ** 2).sum())
            for pn, pv in parts.items():
                p = pv[gm].double(); cp = p - p.mean(0)
                r['c_' + pn] = dict(p2=float((cp ** 2).sum()), p_obs=float((cp * co).sum()), p_Nb=float((cp * cn).sum()))
        res[gname] = r
    res['sum_check'] = float((((selfz + same_ex + other) - tot) ** 2).sum() / (tot ** 2).sum())
    return res


def main():
    rid = sys.argv[1]
    only = [int(a) for a in sys.argv[2:] if a.lstrip('-').isdigit()]
    d_ = f'{R.DL}/{rid}'; ra = f'{R.DL}/_run_artifacts/runs/{rid}'
    meta = json.load(open(f'{d_}/meta.json')); cs = meta['ckpt_steps']
    pool = np.load(f'{d_}/probe_info.npz')['pool_indices']
    bidx = L.true_batches(0, pool, meta['batch_size'], meta['total_steps'])
    if os.environ.get('NTK_BATCH') == 'logged':
        import glob as _g
        bidx = {}
        for _f in sorted(_g.glob(f'{d_}/chunk_*.npz')):
            _z = np.load(_f)
            for _s, _b in zip(_z['steps'], _z['batch_idx']): bidx[int(_s)] = _b[_b >= 0].astype(np.int64)
    ctx = Pm.make_ctx(R.MNIST, f'{ra}/probe_index.npz', f'{ra}/probe_train_index.npz', f'{ra}/train_index.npz')
    noisy = 'ln' in rid
    idx, is_tr, true, given = select_set(ctx, noisy)
    Xk = ctx['Xp'][torch.as_tensor(idx).to(L.DEV)]
    od = f'{OUT}/{rid}'; os.makedirs(od, exist_ok=True)
    gen = torch.Generator().manual_seed(0)
    pairs = [s for s in cs if s + 1 in set(cs) and s + 1 <= meta['total_steps']]
    if only: pairs = [s for s in pairs if s in only]
    for s in pairs:
        fn = f'{od}/tt_{s:07d}.json'
        if os.path.exists(fn) and not only and not os.environ.get('NTK_REDO'): continue
        t0 = time.time()
        prev, _ = R.load_state(f'{d_}/ckpt/step_{s:07d}.pt'); nxt, _ = R.load_state(f'{d_}/ckpt/step_{s + 1:07d}.pt')
        P = prev['params']; dd = L.decompose(prev, nxt)
        Xp = ctx['Xp']
        z0, z1 = L.z_of(P, Xp), L.z_of(nxt['params'], Xp); obs = z1 - z0
        r = dict(step=s + 1, run=rid)
        r['drive'] = drive(P, ctx, dd, bidx[s + 1], obs, None, z0)
        if os.environ.get('NTK_NOKERNEL'):
            json.dump(r, open(fn, 'w')); print(rid, s, 'drive-only', flush=True); continue
        sq = None if nxt['kind'] == 'sgd' else [t.sqrt() for t in dd['Dp']]
        Gs = L.gram(P, Xk, sqrtD=sq)
        r['kernel'] = {nm: kernel_blocks(G, is_tr, true, given, gen) for nm, G in Gs.items()}
        del Gs; torch.cuda.empty_cache()
        json.dump(r, open(fn, 'w'))
        print(rid, s, f'{time.time() - t0:.1f}s dev {L.DEV}', flush=True)


if __name__ == '__main__':
    main()
